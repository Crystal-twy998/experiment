#!/usr/bin/env python3
"""
Minimal BAGEL demo for CIRR-style image editing.

This script follows the official BAGEL GitHub loading path and inference flow,
then adds two simple steps for your test case:
1) Generate a target-image description from the reference image + edit instruction.
2) Generate the edited target image.

Expected usage:
    python demo_bagel_cirr_edit.py \
        --repo_root /path/to/BAGEL_repo \
        --model_path /nativemm/share/cpfs/tangwenyue/models/BAGEL \
        --image_path /nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRR/test1/test1-147-1-img1.png

Notes:
- This script assumes you have already cloned the official BAGEL repository:
  https://github.com/ByteDance-Seed/BAGEL
- It also assumes the checkpoint directory contains files like:
  ema.safetensors, ae.safetensors, llm_config.json, vit_config.json, config.json, etc.
- For a first smoke test, mode=2 (NF4) is usually easier on VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple BAGEL demo for CIRR-style editing")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Path to the official BAGEL GitHub repo root. If omitted, current working directory is used.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/nativemm/share/cpfs/tangwenyue/models/BAGEL",
        help="Local BAGEL checkpoint directory.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/nativemm/share/cpfs/tangwenyue/Reasoning/Datasets/CIRCO/COCO2017_unlabeled/unlabeled2017/000000271520.jpg",
        help="Reference image path.",
    )
    parser.add_argument(
        "--image_caption",
        type=str,
        default="A performer in traditional costume holds an orange parasol on a dimly lit stage.",
        help="Reference image caption.",
    )
    parser.add_argument(
        "--edit_instruction",
        type=str,
        default="shows two people and has a more colorful background",
        help="Edit instruction / relative caption.",
    )
    parser.add_argument(
        "--reference_id",
        type=str,
        default="test1-147-1-img1",
        help="Reference image ID for bookkeeping.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bagel_demo_outputs",
        help="Directory to save generated text and image.",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="1=bf16 full weights, 2=NF4 quantized, 3=INT8 quantized.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default="40GiB",
        help="Max memory string passed to accelerate device_map, e.g. 24GiB / 40GiB / 80GiB.",
    )
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval", type=float, default=0.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument(
        "--cfg_renorm_type",
        type=str,
        default="text_channel",
        choices=["global", "channel", "text_channel"],
    )
    parser.add_argument(
        "--also_text2image",
        action="store_true",
        help="Also generate a pure text-to-image sample from the generated target description.",
    )
    return parser.parse_args()


def import_bagel_modules(repo_root: str | None):
    """Import BAGEL modules from the official repo root, avoiding name collisions.

    Important: when this script is launched inside another project (e.g. WISER),
    Python may otherwise import that project's local `data/` or `modeling/`
    packages instead of BAGEL's own modules.
    """
    if repo_root is None:
        repo_root = os.getcwd()
    repo_root = os.path.abspath(repo_root)

    required = [
        os.path.join(repo_root, 'inferencer.py'),
        os.path.join(repo_root, 'data'),
        os.path.join(repo_root, 'modeling'),
    ]
    missing = [x for x in required if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError(
            'repo_root does not look like the official BAGEL repo root. Missing: ' + ', '.join(missing)
        )

    # Make BAGEL repo highest priority.
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)

    # Remove current working directory / script directory if they point to another
    # project that may also contain `data` or `modeling` packages.
    script_dir = os.path.abspath(os.path.dirname(__file__))
    cwd = os.path.abspath(os.getcwd())
    cleaned = []
    for p in sys.path[1:]:
        ap = os.path.abspath(p or cwd)
        if ap in {script_dir, cwd} and ap != repo_root:
            continue
        cleaned.append(p)
    sys.path[:] = [repo_root] + cleaned

    try:
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
        from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel.qwen2_navit import NaiveCache  # noqa: F401  # imported for side effects / parity
        from modeling.bagel import (
            BagelConfig,
            Bagel,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
    except Exception as e:
        raise ImportError(
            "Failed to import BAGEL official repo modules. "
            "Please clone https://github.com/ByteDance-Seed/BAGEL and pass --repo_root /path/to/BAGEL, "
            "or place this script under the BAGEL repo root before running. "
            f"Original error: {e}"
        ) from e

    return {
        "infer_auto_device_map": infer_auto_device_map,
        "load_checkpoint_and_dispatch": load_checkpoint_and_dispatch,
        "init_empty_weights": init_empty_weights,
        "BnbQuantizationConfig": BnbQuantizationConfig,
        "load_and_quantize_model": load_and_quantize_model,
        "add_special_tokens": add_special_tokens,
        "pil_img2rgb": pil_img2rgb,
        "ImageTransform": ImageTransform,
        "InterleaveInferencer": InterleaveInferencer,
        "load_ae": load_ae,
        "BagelConfig": BagelConfig,
        "Bagel": Bagel,
        "Qwen2Config": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2ForCausalLM,
        "SiglipVisionConfig": SiglipVisionConfig,
        "SiglipVisionModel": SiglipVisionModel,
        "Qwen2Tokenizer": Qwen2Tokenizer,
    }


def build_inferencer(args: argparse.Namespace, mods: Dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL demo currently expects at least one CUDA GPU.")

    model_path = args.model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    Qwen2Config = mods["Qwen2Config"]
    SiglipVisionConfig = mods["SiglipVisionConfig"]
    BagelConfig = mods["BagelConfig"]
    Qwen2ForCausalLM = mods["Qwen2ForCausalLM"]
    SiglipVisionModel = mods["SiglipVisionModel"]
    Bagel = mods["Bagel"]
    load_ae = mods["load_ae"]
    Qwen2Tokenizer = mods["Qwen2Tokenizer"]
    add_special_tokens = mods["add_special_tokens"]
    ImageTransform = mods["ImageTransform"]
    InterleaveInferencer = mods["InterleaveInferencer"]
    infer_auto_device_map = mods["infer_auto_device_map"]
    load_checkpoint_and_dispatch = mods["load_checkpoint_and_dispatch"]
    init_empty_weights = mods["init_empty_weights"]
    BnbQuantizationConfig = mods["BnbQuantizationConfig"]
    load_and_quantize_model = mods["load_and_quantize_model"]

    print(f"[BAGEL] Loading from: {model_path}")
    print(f"[BAGEL] CUDA device count: {torch.cuda.device_count()}")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    max_memory = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device

    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map=device_map,
            offload_buffers=True,
            offload_folder=os.path.join(args.output_dir, "offload"),
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif args.mode == 2:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model = load_and_quantize_model(
            model,
            weights_location=checkpoint_path,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder=os.path.join(args.output_dir, "offload"),
        ).eval()
    else:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True,
            torch_dtype=torch.float32,
        )
        model = load_and_quantize_model(
            model,
            weights_location=checkpoint_path,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder=os.path.join(args.output_dir, "offload"),
        ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def generate_target_description(inferencer, ref_image: Image.Image, edit_instruction: str) -> str:
    prompt = (
        "Given the reference image and the editing instruction, describe the FINAL target image in one concise sentence. "
        "Only describe the edited result, not the original image, and do not explain your reasoning.\n"
        f"Editing instruction: {edit_instruction}"
    )

    result = inferencer(
        image=ref_image,
        text=prompt,
        think=False,
        understanding_output=True,
        do_sample=False,
        text_temperature=0.3,
        max_think_token_n=128,
    )
    text = (result.get("text") or "").strip()
    return text


def edit_target_image(inferencer, ref_image: Image.Image, edit_instruction: str, target_description: str, args: argparse.Namespace):
    prompt = (
        "Edit this image according to the instruction below. "
        "Preserve the scene as much as possible except for the requested changes.\n"
        f"Instruction: {edit_instruction}\n"
        f"Desired final image: {target_description}"
    )

    result = inferencer(
        image=ref_image,
        text=prompt,
        think=False,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[args.cfg_interval, 1.0],
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )
    return result.get("image")


def text_to_image_from_description(inferencer, target_description: str) -> Image.Image | None:
    result = inferencer(
        text=target_description,
        think=False,
        cfg_text_scale=4.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    )
    return result.get("image")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "offload"), exist_ok=True)

    set_seed(args.seed)

    mods = import_bagel_modules(args.repo_root)
    pil_img2rgb = mods["pil_img2rgb"]

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    ref_image = pil_img2rgb(Image.open(image_path).convert("RGB"))
    inferencer = build_inferencer(args, mods)

    print("\n[Step 1] Generating target-image description...")
    target_description = generate_target_description(inferencer, ref_image, args.edit_instruction)
    print(f"[Target Description] {target_description}")

    print("\n[Step 2] Generating edited target image...")
    edited_image = edit_target_image(inferencer, ref_image, args.edit_instruction, target_description, args)
    if edited_image is None:
        raise RuntimeError("Image editing returned None. Please check model loading and inference logs.")

    description_path = os.path.join(args.output_dir, f"{args.reference_id}_target_description.txt")
    edited_image_path = os.path.join(args.output_dir, f"{args.reference_id}_edited.png")

    with open(description_path, "w", encoding="utf-8") as f:
        f.write(target_description + "\n")
    edited_image.save(edited_image_path)

    metadata = {
        "reference": args.reference_id,
        "image_path": args.image_path,
        "caption": args.edit_instruction,
        "generated_target_description": target_description,
        "edited_image_path": edited_image_path,
        "mode": args.mode,
        "seed": args.seed,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": args.cfg_interval,
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
    }

    metadata_path = os.path.join(args.output_dir, f"{args.reference_id}_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[Saved]")
    print(f"- target text : {description_path}")
    print(f"- edited image: {edited_image_path}")
    print(f"- metadata    : {metadata_path}")

    if args.also_text2image:
        print("\n[Step 3] Generating an extra text-to-image sample from the target description...")
        t2i_image = text_to_image_from_description(inferencer, target_description)
        if t2i_image is not None:
            t2i_path = os.path.join(args.output_dir, f"{args.reference_id}_t2i.png")
            t2i_image.save(t2i_path)
            print(f"- text-to-image sample: {t2i_path}")


if __name__ == "__main__":
    main()
