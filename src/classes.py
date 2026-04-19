from enum import Enum, auto
import os
import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms
import clip


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def load_clip_model_and_preprocess(dataset_path: str, clip_type: str, device: torch.device, jit: bool = False):
    print(f'Loading CLIP {clip_type}...', flush=True)

    openclip_types = ['ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']

    if clip_type in openclip_types:
        pretraining = {
            'ViT-B-32': 'laion2b_s34b_b79k',
            'ViT-B-16': 'laion2b_s34b_b88k',
            'ViT-L-14': 'laion2b_s32b_b82k',
            'ViT-H-14': 'laion2b_s32b_b79k',
            'ViT-g-14': 'laion2b_s34b_b88k',
            'ViT-bigG-14': 'laion2b_s39b_b160k'
        }

        weight_root = os.path.abspath(os.path.join(dataset_path, '..', 'weights', 'open_clip'))
        os.makedirs(weight_root, exist_ok=True)

        local_model_dir = os.path.join(weight_root, clip_type)
        local_weight_path = os.path.join(local_model_dir, 'open_clip_pytorch_model.bin')

        print(f'[OpenCLIP] weight_root: {weight_root}', flush=True)
        print(f'[OpenCLIP] local_weight_path: {local_weight_path}', flush=True)

        if os.path.isfile(local_weight_path):
            print(f'[OpenCLIP] Load from local file: {local_weight_path}', flush=True)
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                clip_type,
                pretrained=local_weight_path
            )
        else:
            print(f'[OpenCLIP] Local file not found, fallback to cache/tag loading.', flush=True)
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                clip_type,
                pretrained=pretraining[clip_type],
                cache_dir=weight_root
            )

        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer(clip_type)
        clip_model.tokenizer = tokenizer

    else:
        clip_model, clip_preprocess = clip.load(clip_type, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False).to(device)

    print(f'CLIP {clip_type} loaded.', flush=True)
    return clip_model, clip_preprocess
