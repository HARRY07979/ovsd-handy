import re
import torch
from optimum.intel.openvino import (
    OVStableDiffusionPipeline,
    OVStableDiffusionImg2ImgPipeline,
    OVStableDiffusionInpaintPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLPipeline,
)
from typing import Optional, Tuple, Union
from huggingface_hub import model_info

def built_in_detect_nsfw(prompt: str) -> bool:
    NSFW_HIGH = {
        "nude", "naked", "sex", "porn", "xxx", "fuck", "dick", "cock", "pussy", "vagina", "penis",
        "boobs", "tits", "breasts", "bra", "panties", "underwear", "lingerie", "orgasm", "cum",
        "blowjob", "handjob", "masturbate", "rape", "gangbang", "incest", "hentai", "lewd",
        "erotic", "kinky", "bondage", "bdsm", "squirt", "creampie", "threesome", "orgy", "yaoi",
        "yuri", "futanari", "cunnilingus", "fellatio", "anal", "paizuri", "bukkake", "guro",
        "vore", "tentacle", "netorare", "cuckold", "exhibitionism", "voyeurism", "poop", "pee",
        "poo", "shit", "piss", "scat", "diarrhea", "vomit", "gore", "blood", "murder",
        "torture", "suicide", "decapitation", "mutilation", "drugs", "cocaine", "heroin",
        "lsd", "ecstasy", "vlxx"
    }

    NSFW_MEDIUM = {
        "bikini", "swimwear", "sexy", "succubus", "leather", "latex", "stockings", "miniskirt",
        "cleavage", "thighs", "ass", "butt", "skirt", "dress", "topless", "wet", "moaning",
        "spread", "legs apart", "tight", "revealing", "provocative", "suggestive", "flirty"
    }

    NSFW_PHRASES = {
        "spreading legs", "removing bra", "pulling panties", "sucking dick", "licking pussy",
        "penetrating", "fucking scene", "hard cock", "wet pussy", "big tits", "exposed breasts",
        "nipples visible", "ass spread", "thigh gap", "camel toe", "pussy lips", "cum on face",
        "blowjob scene", "anal sex", "titty fuck", "gang rape", "group sex", "public sex",
        "hidden camera", "peeing girl", "pooping girl", "covered in blood", "cutting flesh",
        "snorting cocaine", "injecting heroin", "hallucinating", "smoking weed"
    }

    SENSITIVE_CONTEXT = {
        "spread", "removing", "pulling", "sucking", "licking", "penetrating",
        "fucking", "hard", "wet", "exposed", "visible", "ass", "tight", "revealing"
    }
    
    prompt_lower = prompt.lower()
    words = set(re.findall(r'\b\w+\b', prompt_lower))

    if words & NSFW_HIGH:
        return True
    
    for phrase in NSFW_PHRASES:
        if phrase in prompt_lower:
            return True
    
    medium_matches = words & NSFW_MEDIUM
    if medium_matches and (words & SENSITIVE_CONTEXT):
        return True
    
    return False


# Global pipeline
pipe = None
current_model_id = None

def detect_pipeline_class(model_id: str):
    """Tự động detect loại pipeline cần dùng."""
    info = model_info(model_id)
    tags = " ".join(info.tags).lower() if hasattr(info, "tags") else ""

    if "inpainting" in model_id.lower() or "inpaint" in tags:
        return OVStableDiffusionInpaintPipeline
    elif "img2img" in model_id.lower() or "image-to-image" in tags:
        return OVStableDiffusionImg2ImgPipeline
    elif "sdxl" in model_id.lower() or "xl" in tags:
        # SDXL có cả base và img2img
        if "img2img" in model_id.lower():
            return OVStableDiffusionXLImg2ImgPipeline
        return OVStableDiffusionXLPipeline
    return OVStableDiffusionPipeline


def CustomOVStableDiffusionPipeline(model_id: str):
    global pipe, current_model_id

    if not isinstance(model_id, str):
        raise ValueError("Var model_id must be a string.")
    
    if pipe is not None and current_model_id == model_id:
        return  # không reload nếu đang xài cùng model

    pipeline_cls = detect_pipeline_class(model_id)
    pipe = pipeline_cls.from_pretrained(
        model_id,
        dtype=torch.float32,
        feature_extractor=None,
        safety_checker=None
    )
    pipe.to("cpu")
    pipe.compile()
    current_model_id = model_id


def CustomOVStableDiffusionHandyPipeline(
    prompt: str,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    width: int = 512,
    height: int = 512,
    seed: Optional[Union[int, str]] = None,
    model_id: str = "HARRY07979/stable-diffusion-v1-5-openvino"
) -> Tuple[torch.Tensor, int]:
    global pipe
    
    if built_in_detect_nsfw(prompt):
        raise ValueError("NSFW content detected in prompt. Please modify your prompt.")
    
    if pipe is None or current_model_id != model_id:
        CustomOVStableDiffusionPipeline(model_id)
    
    if seed is None or seed == "":
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        try:
            seed = int(seed)
        except Exception:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
    seed = seed % (2**32)

    generator = torch.Generator().manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        )

    return result.images[0], seed
