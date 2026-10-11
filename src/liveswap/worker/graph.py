from liveswap.contracts import Profile


def build_graph(profile: Profile, job_id: str, garment_hash: str) -> dict:
    """Native ComfyUI API graph; no client-supplied arbitrary graph execution."""
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": profile.model, "weight_dtype": profile.weight_dtype}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": profile.text_encoder, "type": "wan", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": profile.vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": profile.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": profile.negative_prompt}},
        "6": {"class_type": "LiveSwapLoadWindow", "inputs": {"job_id": job_id, "garment_hash": garment_hash}},
        "7": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0],
            "width": profile.width, "height": profile.height, "length": profile.length,
            "batch_size": 1, "strength": 1.0, "control_video": ["6", 0],
            "control_masks": ["6", 1], "reference_image": ["6", 2],
        }},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["13", 0] if profile.lora else ["1", 0], "shift": profile.shift}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["8", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2],
            "seed": profile.seed, "steps": profile.steps, "cfg": profile.cfg,
            "sampler_name": profile.sampler, "scheduler": profile.scheduler, "denoise": 1.0,
        }},
        "10": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["9", 0], "trim_amount": ["7", 3]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "LiveSwapSaveFrames", "inputs": {"images": ["11", 0], "job_id": job_id, "quality": profile.jpeg_quality}},
    }
    if profile.lora:
        graph["13"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": profile.lora, "strength_model": profile.lora_strength,
        }}
    return graph
