"""Bounded-memory implementation of VDA's upstream 32-frame/keyframe inference algorithm."""
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def infer(adapter, frames):
    from torchvision.transforms import Compose
    from video_depth_anything.video_depth import INFER_LEN, OVERLAP, KEYFRAMES, INTERP_LEN
    from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from utils.util import compute_scale_and_shift, get_interpolate_frames

    count = len(frames)
    if not count:
        raise ValueError("Empty video")
    height, width = frames[0].shape[:2]
    ratio, size = max(height,width)/min(height,width), adapter.input_size
    if ratio > 1.78:
        size = round(int(size*1.777/ratio)/14)*14
    transform = Compose([Resize(width=size,height=size,resize_target=False,keep_aspect_ratio=True,
                        ensure_multiple_of=14,resize_method="lower_bound",image_interpolation_method=cv2.INTER_CUBIC),
                        NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()])
    if getattr(adapter,"_temporary",None) is not None:
        adapter._temporary.cleanup()
    adapter._temporary = TemporaryDirectory(prefix="vda-prediction-")
    step, align_len = INFER_LEN-OVERLAP, OVERLAP-INTERP_LEN
    length = INFER_LEN + max(0, (count-1)//step)*step
    output = np.lib.format.open_memmap(Path(adapter._temporary.name)/"depth.npy",mode="w+",
                                      dtype=np.float32,shape=(length,height,width))
    previous, references, filled = None, [], 0
    for offset in range(0,count,step):
        tensors = [torch.from_numpy(transform({"image":frames[min(offset+i,count-1)].astype(np.float32)/255.})["image"])[None,None]
                   for i in range(INFER_LEN)]
        inputs = torch.cat(tensors,dim=1).cuda()
        if previous is not None:
            inputs[:,:OVERLAP] = previous[:,KEYFRAMES]
        with torch.no_grad(), torch.autocast("cuda",enabled=not adapter.fp32):
            value = adapter.model(inputs)
        value = F.interpolate(value.to(inputs.dtype).flatten(0,1)[:,None],size=(height,width),
                              mode="bilinear",align_corners=True)[:,0].float().cpu().numpy()
        if not filled:
            output[:INFER_LEN] = value
            filled = INFER_LEN
            references = [value[k].copy() for k in KEYFRAMES[:align_len]]
        else:
            if adapter.kind == "metric_m":
                scale,shift = 1.,0.
            else:
                scale,shift = compute_scale_and_shift(np.concatenate(value[:align_len]),
                    np.concatenate(references),np.concatenate(np.ones_like(references)==1))
            after = [np.maximum(v*scale+shift,0) for v in value[align_len:OVERLAP]]
            output[filled-INTERP_LEN:filled] = get_interpolate_frames(
                list(output[filled-INTERP_LEN:filled]),after)
            output[filled:filled+step] = np.maximum(value[OVERLAP:]*scale+shift,0)
            filled += step
            references = references[:1] + [np.maximum(value[k]*scale+shift,0)
                                           for k in KEYFRAMES[1:align_len]]
        previous = inputs
    return output[:count]
