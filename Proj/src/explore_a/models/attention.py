"""Flash-preferred timm attention core; retain Q/K normalization, gating and projections."""
from collections import Counter
from types import MethodType

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


class AttentionPolicy:
    def __init__(self):
        self.counts, self.reasons = Counter(), {}
        self.disabled_external = set()
        try:
            from flash_attn import flash_attn_func
            self.external = flash_attn_func
        except (ImportError, OSError, RuntimeError) as error:
            self.external = None
            self.reasons["extension"] = str(error)

    def __call__(self, q, k, v, mask, dropout, causal, scale):
        compatible = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and mask is None
        key = str(q.device), str(q.dtype), q.shape[-1], causal
        if compatible and self.external is not None and key not in self.disabled_external:
            try:
                result = self.external(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       dropout_p=dropout, softmax_scale=scale, causal=causal)
                self.counts["flash_attn_2"] += 1
                return result.transpose(1, 2)
            except RuntimeError as error:
                text = str(error).lower()
                if not any(word in text for word in ("no kernel", "not supported", "unsupported",
                                                     "invalid device function")):
                    raise
                self.disabled_external.add(key)
                self.reasons[str(key)] = str(error)
        if compatible:
            parameters = torch.backends.cuda.SDPAParams(q, k, v, mask, dropout, causal, False)
            if torch.backends.cuda.can_use_flash_attention(parameters):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    result = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout, is_causal=causal, scale=scale)
                self.counts["torch_sdpa_flash"] += 1
                return result
        self.counts["native_sdpa_fallback"] += 1
        self.reasons.setdefault("fallback", "dtype/mask/device/kernel not Flash-compatible")
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=causal, scale=scale)

    def report(self):
        return {"calls": dict(self.counts), "reasons": self.reasons,
                "mftiq_dinov2": "FP32 retained; no low-precision Flash conversion"}


def configure_depth_attention(model):
    policy = AttentionPolicy()
    count = 0
    for module in model.modules():
        if not module.__class__.__module__.startswith("timm."):
            continue
        fields = ("qkv", "num_heads", "head_dim", "q_norm", "k_norm", "scale", "norm",
                  "attn_dim", "proj", "proj_drop", "attn_drop")
        if not all(hasattr(module, field) for field in fields):
            continue

        def forward(self, x, attn_mask=None, is_causal=False):
            batch, tokens, _ = x.shape
            gate = self.gate(x).sigmoid() if getattr(self, "gate", None) is not None else None
            packed = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            value = policy(q, k, v, attn_mask, self.attn_drop.p if self.training else 0.,
                           is_causal, self.scale)
            value = self.norm(value.transpose(1, 2).reshape(batch, tokens, self.attn_dim))
            if gate is not None:
                value = value * gate
            return self.proj_drop(self.proj(value))

        module.forward = MethodType(forward, module)
        count += 1
    policy.reasons["adapted_timm_modules"] = count
    return policy
