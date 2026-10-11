"""Install this directory under ComfyUI/custom_nodes. ComfyUI supplies torch."""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def root():
    return Path(os.environ.get("LIVESWAP_DATA_DIR", "/data/liveswap"))


def check_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}-[0-9]+", job_id):
        raise ValueError("invalid job identifier")


class LiveSwapLoadWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_id": ("STRING",), "garment_hash": ("STRING",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    FUNCTION = "load"
    CATEGORY = "LiveSwap"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, job_id, garment_hash):
        check_job(job_id)
        if not re.fullmatch(r"[a-f0-9]{64}", garment_hash):
            raise ValueError("invalid garment hash")
        folder = root() / "jobs" / job_id
        count = json.loads((folder / "input.json").read_text())["count"]
        if not 5 <= count <= 81:
            raise ValueError("invalid frame count")
        frames, masks = [], []
        for index in range(count):
            with Image.open(folder / f"frame-{index:03}.png") as image:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
            with Image.open(folder / f"mask-{index:03}.png") as image:
                masks.append(np.asarray(image.convert("L"), dtype=np.float32) / 255)
        with Image.open(root() / "assets" / f"{garment_hash}.png") as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(np.stack(frames)), torch.from_numpy(np.stack(masks)), torch.from_numpy(reference[None])


class LiveSwapSaveFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "job_id": ("STRING",), "quality": ("INT", {"default": 94, "min": 80, "max": 100})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LiveSwap"

    def save(self, images, job_id, quality):
        check_job(job_id)
        # Return files through ComfyUI /view; the backend deletes them after download.
        import folder_paths
        folder = Path(folder_paths.get_output_directory()) / "liveswap" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        output = []
        for index, tensor in enumerate(images):
            filename = f"{index:03}.jpg"
            array = np.clip(tensor.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(folder / filename, quality=quality)
            output.append({"filename": filename, "subfolder": f"liveswap/{job_id}", "type": "output"})
        return {"ui": {"images": output}}


NODE_CLASS_MAPPINGS = {"LiveSwapLoadWindow": LiveSwapLoadWindow, "LiveSwapSaveFrames": LiveSwapSaveFrames}
