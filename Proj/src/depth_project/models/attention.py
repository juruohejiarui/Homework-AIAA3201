"""Optional SDPA/FlashAttention inference adapters for the upstream models."""
import importlib

import torch
import torch.nn.functional as functional


def attend(query, key, value, backend, scale, dropout=0, mask=None):
    use_flash = backend == "flash" and query.is_cuda and mask is None
    use_flash &= query.dtype in (torch.float16, torch.bfloat16)
    if use_flash:
        from flash_attn import flash_attn_func

        outputs = []
        for start in range(0, len(query), 8192):
            chunk = slice(start, start + 8192)
            outputs.append(flash_attn_func(query[chunk], key[chunk], value[chunk],
                                          dropout_p=dropout, softmax_scale=scale))
        return torch.cat(outputs)
    result = functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attn_mask=mask, dropout_p=dropout, scale=scale,
    )
    return result.transpose(1, 2)


def configure(package, backend):
    if backend == "upstream":
        return
    attention = importlib.import_module(f"{package}.dinov2_layers.attention")

    def spatial_forward(self, inputs, attn_bias=None):
        if attn_bias is not None:
            raise NotImplementedError("This adapter supports dense inference only")
        batch, length, channels = inputs.shape
        projected = self.qkv(inputs).reshape(batch, length, 3, self.num_heads, channels // self.num_heads)
        dropout = self.attn_drop.p if self.training else 0
        output = attend(projected[:, :, 0], projected[:, :, 1], projected[:, :, 2],
                        backend, self.scale, dropout)
        return self.proj_drop(self.proj(output.reshape(batch, length, channels)))

    attention.Attention.forward = spatial_forward
    attention.MemEffAttention.forward = spatial_forward
    if package != "video_depth_anything":
        return
    temporal = importlib.import_module(f"{package}.motion_module.attention")

    def temporal_forward(self, query, key, value, attention_mask=None):
        batch_heads, length, channels = query.shape
        batch = batch_heads // self.heads
        query = query.reshape(batch, self.heads, length, channels).transpose(1, 2)
        key = key.reshape(batch, self.heads, key.shape[1], channels).transpose(1, 2)
        value = value.reshape(batch, self.heads, value.shape[1], channels).transpose(1, 2)
        dtype = value.dtype
        if self.upcast_attention or self.upcast_softmax:
            query = query.float()
            key = key.float()
            value = value.float()
        mask = None
        if attention_mask is not None:
            mask = attention_mask.reshape(batch, self.heads, *attention_mask.shape[-2:])
        output = attend(query, key, value, backend, self.scale, mask=mask)
        return output.to(dtype).reshape(batch, length, self.heads * channels)

    temporal.CrossAttention._attention = temporal_forward
    motion = importlib.import_module(f"{package}.motion_module.motion_module")
    motion.XFORMERS_AVAILABLE = False
