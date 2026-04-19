import os
import argparse
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import logging
from transformers.utils import logging as hf_logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.models.blip_2.modeling_blip_2").setLevel(logging.ERROR)

from transformers import Blip2Processor, Blip2ForConditionalGeneration

torch.multiprocessing.set_sharing_strategy("file_system")

def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    """
    For single-GPU loading, returns that GPU.
    For device_map='auto', returns the first non-meta parameter device.
    """
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def load_model_and_processor(
    captioner: str,
    model_path: str,
    mode: str = "single",
    dtype: str = "fp32",
    device: str = "cuda",
):
    """
    Maximum alignment with original WISER:
    - default dtype is fp32
    - default mode is single
    - no sampling tricks
    """
    if captioner != "blip2_t5":
        raise ValueError(f"Unsupported model type: {captioner}")

    torch_dtype = resolve_torch_dtype(dtype)

    processor = Blip2Processor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    common_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if mode == "single":
        model = Blip2ForConditionalGeneration.from_pretrained(
            **common_kwargs,
            device_map=None,
        )
        model = model.to(device)
    elif mode == "auto":
        model = Blip2ForConditionalGeneration.from_pretrained(
            **common_kwargs,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Keep behavior deterministic / inference-only
    model.eval()

    # Align with original WISER's model.float() when dtype=fp32
    # If user explicitly chooses fp16/bf16, respect that choice.
    if dtype in ["fp32", "float32"]:
        model = model.float()

    return processor, model


def generate_captions_batch(
    model_type: str,
    model,
    processor,
    image_paths: List[str],
    prompt: str,
    max_new_tokens: Optional[int] = None,
):
    if model_type != "blip2_t5":
        raise ValueError(f"Unsupported model type: {model_type}")

    images = [Image.open(p).convert("RGB") for p in image_paths]

    inputs = processor(
        images=images,
        text=[prompt] * len(images),
        return_tensors="pt",
        padding=True,
    )

    input_device = get_model_input_device(model)
    if input_device.type != "cpu":
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

    generate_kwargs = dict(
        **inputs,
        do_sample=False,
    )

    # To stay maximally aligned with original WISER, do NOT force max_new_tokens
    # unless the user explicitly sets it.
    if max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        generated_ids = model.generate(**generate_kwargs)

    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]


def generate_captions(
    img_dir: str,
    output_csv: str,
    captioner: str,
    model_path: str,
    batch_size: int = 1,
    mode: str = "single",
    dtype: str = "fp32",
    device: str = "cuda",
    max_new_tokens: Optional[int] = None,
):
    processor, model = load_model_and_processor(
        captioner=captioner,
        model_path=model_path,
        mode=mode,
        dtype=dtype,
        device=device,
    )

    # Keep exactly the same prompt as original WISER
    prompt = (
        "Describe the image in complete detail. "
        "You must especially focus on all the objects in the image."
    )

    img_file_list = sorted(
        [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
    )

    image_id_list = []
    generated_text_list = []

    for i in tqdm(range(0, len(img_file_list), batch_size)):
        batch_files = img_file_list[i: i + batch_size]
        batch_paths = [os.path.join(img_dir, img_file) for img_file in batch_files]

        try:
            captions = generate_captions_batch(
                model_type=captioner,
                model=model,
                processor=processor,
                image_paths=batch_paths,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            captions = [f"Error: {e}"] * len(batch_files)

        for img_file, caption in zip(batch_files, captions):
            image_id_list.append(os.path.splitext(img_file)[0])
            generated_text_list.append(caption)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(
        {
            "image_id": image_id_list,
            "generated_text": generated_text_list,
        }
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Finished, save in: {output_csv}")
    print(f"Total images processed: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator")

    parser.add_argument("--dataset_name", type=str, default="CIRCO", choices=["FASHIONIQ", "CIRCO", "CIRR"],)
    parser.add_argument("--captioner", type=str, default="blip2_t5", choices=["blip2_t5"],)

    # Add xl / xxl local checkpoints
    parser.add_argument("--model_variant", type=str, default="xxl", choices=["xl", "xxl"], help="Default is xxl to align with original WISER.",)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1, help="Default 1 to align with original WISER.",)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "auto"], help="Default single to align with original WISER. Use auto only if OOM.",)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Default fp32 to align with original WISER's model.float().",)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Used only when mode=single.",)
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Default None for maximum alignment with original WISER.",)

    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)

    args = parser.parse_args()

    default_model_paths = {
        "xl": "/data1/tangwenyue/Models/blip2-flan-t5-xl",
        "xxl": "/data1/tangwenyue/Models/blip2-flan-t5-xxl",
    }

    dataset_image_path = {
        "FASHIONIQ": "/data1/tangwenyue/Dataset/FashionIQ/images",
        "CIRCO": "/data1/tangwenyue/Dataset/CIRCO/COCO2017_unlabeled/unlabeled2017",
        "CIRR": "/data1/tangwenyue/Dataset/CIRR/test1",
    }

    model_path = args.model_path or default_model_paths[args.model_variant]
    img_dir = args.img_dir or dataset_image_path[args.dataset_name]

    output_csv = args.output_csv
    if output_csv is None:
        output_name = f"{args.dataset_name}_{args.captioner}_{args.model_variant}_captions.csv"
        output_csv = os.path.join(
            "/data1/tangwenyue/Dataset",
            args.dataset_name,
            "preload/image_captions",
            output_name,
        )

    print(f"captioner      : {args.captioner}")
    print(f"model_variant  : {args.model_variant}")
    print(f"model_path     : {model_path}")
    print(f"mode           : {args.mode}")
    print(f"dtype          : {args.dtype}")
    print(f"device         : {args.device}")
    print(f"batch_size     : {args.batch_size}")
    print(f"max_new_tokens : {args.max_new_tokens}")
    print(f"img_dir        : {img_dir}")
    print(f"output_csv     : {output_csv}")

    generate_captions(
        img_dir=img_dir,
        output_csv=output_csv,
        captioner=args.captioner,
        model_path=model_path,
        batch_size=args.batch_size,
        mode=args.mode,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()