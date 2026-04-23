#!/usr/bin/env python3
"""
Single-sample BAGEL demo aligned with the current experiment/src codebase.

What this script does:
1. Loads BAGEL through experiment/src/bagel_inference.py (BagelImageEditor).
2. Loads unified prompts from experiment/src/prompts.py.
3. Runs four stage-1 prompt modes on one sample:
   - structural_modifier_prompt                (caption + instruction)
   - mllm_structural_predictor_prompt_CoT      (image + instruction)
   - image_mllm_structural_predictor_prompt_CoT(image + instruction)
   - mllm_structural_predictor_prompt_CoT_multi(image + instruction)
4. Saves raw outputs, parsed descriptions, and optional edited images for analysis.

This is intentionally a single-sample analysis tool rather than the batch pipeline in utils.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch


PROMPT_MODE_TO_NAME = {
    "structural_modifier": "structural_modifier_prompt",
    "mllm_cot": "mllm_structural_predictor_prompt_CoT",
    "image_mllm_cot": "image_mllm_structural_predictor_prompt_CoT",
    "mllm_cot_multi": "mllm_structural_predictor_prompt_CoT_multi",
}

MULTI_KEY_ALIASES = {
    "conservative": ["Conservative Query"],
    "balanced": ["Balanced Query"],
    "reasoning": ["Reasoning Enhanced Query", "Reasoning-Enhanced Query", "Reasoning Query"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-sample BAGEL demo aligned to experiment/src")
    parser.add_argument("--repo_src_root", type=str, required=True,
                        help="Path to experiment/src directory.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local BAGEL checkpoint directory.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Reference image path.")
    parser.add_argument("--image_caption", type=str, default="",
                        help="Reference image caption for structural_modifier mode.")
    parser.add_argument("--edit_instruction", type=str, required=True,
                        help="Relative caption / edit instruction.")
    parser.add_argument("--reference_id", type=str, default="sample",
                        help="Sample id for bookkeeping.")
    parser.add_argument("--output_dir", type=str, default="./bagel_demo_outputs_latest")
    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=["structural_modifier", "mllm_cot", "image_mllm_cot", "mllm_cot_multi"],
        choices=list(PROMPT_MODE_TO_NAME.keys()),
    )
    parser.add_argument("--multi_query_choice", type=str, default="conservative",
                        choices=["conservative", "balanced", "reasoning"],
                        help="Which multi-query branch to use when optionally generating the edited image.")
    parser.add_argument("--generate_images", action="store_true",
                        help="Also generate edited images using edit_image_no_think.")
    parser.add_argument(
        "--image_edit_prompt_mode",
        type=str,
        default="instruction_plus_target",
        choices=["instruction_only", "target_text_only", "instruction_plus_target"],
        help="Prompt composition for the optional edited-image generation step.",
    )
    parser.add_argument("--max_think_token_n", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--offload_folder", type=str, default="/tmp/offload_bagel_demo")
    parser.add_argument("--device", type=str, default=None,
                        help="Optional device override, e.g. cuda:0")
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_start", type=float, default=0.0)
    parser.add_argument("--cfg_interval_end", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel",
                        choices=["global", "channel", "text_channel"])
    return parser.parse_args()


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_repo_modules(repo_src_root: Path):
    if not repo_src_root.exists():
        raise FileNotFoundError(f"repo_src_root not found: {repo_src_root}")
    sys.path.insert(0, str(repo_src_root))
    bagel_inference = load_module_from_path("bagel_inference_local", repo_src_root / "bagel_inference.py")
    prompts_module = load_module_from_path("prompts_local", repo_src_root / "prompts.py")
    return bagel_inference, prompts_module


def ensure_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    return str(value)


def extract_json_substring(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_single_description(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    result: Dict[str, Any] = {
        "raw_text": text,
        "parsed_description": "",
        "parser": "fallback",
    }
    if not text:
        return result

    for key in ["Edited Description", "Target Image Description"]:
        pattern = rf'{re.escape(key)}\s*[:：]\s*(.+)'
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            result["parsed_description"] = m.group(1).strip().strip('"')
            result["parser"] = f"line_prefix:{key}"
            return result

    json_block = extract_json_substring(text)
    if json_block:
        try:
            obj = json.loads(json_block)
            for key in ["Edited Description", "Target Image Description"]:
                if isinstance(obj, dict) and key in obj and isinstance(obj[key], str):
                    result["parsed_description"] = obj[key].strip()
                    result["parser"] = f"json:{key}"
                    result["json"] = obj
                    return result
        except Exception:
            pass

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        result["parsed_description"] = lines[-1]
        result["parser"] = "last_nonempty_line"
    return result


def parse_multi_queries(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    result: Dict[str, Any] = {
        "raw_text": text,
        "queries": {},
        "parser": "unparsed",
    }
    if not text:
        return result

    json_block = extract_json_substring(text)
    if json_block:
        try:
            obj = json.loads(json_block)
            if isinstance(obj, dict):
                result["queries"] = obj
                result["parser"] = "json"
                return result
        except Exception:
            pass

    # Regex fallback for semi-structured outputs.
    patterns = {
        "Conservative Query": r'Conservative Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?rationale\s*[:：]\s*"?([^"\n]+)"?',
        "Balanced Query": r'Balanced Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?rationale\s*[:：]\s*"?([^"\n]+)"?',
        "Reasoning Enhanced Query": r'Reasoning[ -]?Enhanced Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?rationale\s*[:：]\s*"?([^"\n]+)"?',
    }
    queries = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            queries[key] = {
                "description": m.group(1).strip(),
                "rationale": m.group(2).strip(),
            }
    if queries:
        result["queries"] = queries
        result["parser"] = "regex"
    return result


def get_prompt_text(prompts_module, prompt_name: str) -> str:
    if not hasattr(prompts_module, prompt_name):
        raise AttributeError(f"Prompt {prompt_name} not found in prompts.py")
    prompt = getattr(prompts_module, prompt_name)
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt {prompt_name} is not a string")
    return prompt


def build_caption_prompt(base_prompt: str, image_caption: str, edit_instruction: str) -> str:
    return base_prompt + "\n" + f"Image Content: {image_caption}" + "\n" + f"Instruction: {edit_instruction}"


def build_image_understanding_prompt(base_prompt: str, edit_instruction: str) -> str:
    return base_prompt + "\n" + f"Modification Text: {edit_instruction}"


def build_image_edit_prompt(mode: str, instruction: str, target_text: str) -> str:
    if mode == "instruction_only":
        return instruction
    if mode == "target_text_only":
        return target_text
    return (
        "Edit this reference image according to the modification text and the desired final target description.\n"
        f"Modification Text: {instruction}\n"
        f"Target Image Description: {target_text}"
    )


def run_text_only_prompt(editor, prompt_text: str, max_think_token_n: int, do_sample: bool) -> str:
    return editor.generate_caption(prompt_text, max_think_token_n=max_think_token_n, do_sample=do_sample)


def run_image_conditioned_prompt(editor, image: Image.Image, prompt_text: str,
                                 max_think_token_n: int, do_sample: bool) -> str:
    output_dict = editor.inferencer(
        image=image,
        text=prompt_text,
        understanding_output=True,
        max_think_token_n=max_think_token_n,
        do_sample=do_sample,
    )
    return (output_dict.get("text") or "").strip()


def select_multi_query(multi_result: Dict[str, Any], choice: str) -> Optional[Dict[str, Any]]:
    queries = multi_result.get("queries") or {}
    for key in MULTI_KEY_ALIASES[choice]:
        if key in queries and isinstance(queries[key], dict):
            desc = queries[key].get("description", "")
            rationale = queries[key].get("rationale", "")
            return {"key": key, "description": desc, "rationale": rationale}
    return None


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_src_root = Path(args.repo_src_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bagel_inference, prompts_module = import_repo_modules(repo_src_root)
    BagelImageEditor = bagel_inference.BagelImageEditor

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    ref_image = Image.open(image_path).convert("RGB")

    editor = BagelImageEditor(
        model_path=args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=args.offload_folder,
        device=args.device,
        use_multi_gpu=args.use_multi_gpu,
    )

    summary: Dict[str, Any] = {
        "reference_id": args.reference_id,
        "image_path": str(image_path),
        "image_caption": args.image_caption,
        "edit_instruction": args.edit_instruction,
        "prompt_modes": args.prompt_modes,
        "results": {},
    }

    for mode in args.prompt_modes:
        prompt_name = PROMPT_MODE_TO_NAME[mode]
        base_prompt = get_prompt_text(prompts_module, prompt_name)
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        if mode == "structural_modifier":
            if not args.image_caption.strip():
                raise ValueError("--image_caption is required for structural_modifier mode.")
            final_prompt = build_caption_prompt(base_prompt, args.image_caption, args.edit_instruction)
            raw_output = run_text_only_prompt(editor, final_prompt, args.max_think_token_n, args.do_sample)
            parsed = parse_single_description(raw_output)
            chosen_target = parsed.get("parsed_description", "")
            mode_result: Dict[str, Any] = {
                "mode": mode,
                "prompt_name": prompt_name,
                "prompt_type": "text_only_caption_based",
                "parsed": parsed,
                "chosen_target_description": chosen_target,
            }
        elif mode == "mllm_cot_multi":
            final_prompt = build_image_understanding_prompt(base_prompt, args.edit_instruction)
            raw_output = run_image_conditioned_prompt(editor, ref_image, final_prompt, args.max_think_token_n, args.do_sample)
            parsed_multi = parse_multi_queries(raw_output)
            selected = select_multi_query(parsed_multi, args.multi_query_choice)
            chosen_target = selected["description"] if selected else ""
            mode_result = {
                "mode": mode,
                "prompt_name": prompt_name,
                "prompt_type": "image_conditioned_multi_query",
                "parsed": parsed_multi,
                "selected_query": selected,
                "chosen_target_description": chosen_target,
            }
        else:
            final_prompt = build_image_understanding_prompt(base_prompt, args.edit_instruction)
            raw_output = run_image_conditioned_prompt(editor, ref_image, final_prompt, args.max_think_token_n, args.do_sample)
            parsed = parse_single_description(raw_output)
            chosen_target = parsed.get("parsed_description", "")
            mode_result = {
                "mode": mode,
                "prompt_name": prompt_name,
                "prompt_type": "image_conditioned_single_query",
                "parsed": parsed,
                "chosen_target_description": chosen_target,
            }

        save_text(mode_dir / "request_prompt.txt", final_prompt)
        save_text(mode_dir / "raw_output.txt", raw_output)

        if args.generate_images and chosen_target:
            image_edit_prompt = build_image_edit_prompt(
                args.image_edit_prompt_mode,
                args.edit_instruction,
                chosen_target,
            )
            image_result = editor.edit_image_no_think(
                image_path=str(image_path),
                prompt=image_edit_prompt,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                cfg_interval=[args.cfg_interval_start, args.cfg_interval_end],
                timestep_shift=args.timestep_shift,
                num_timesteps=args.num_timesteps,
                cfg_renorm_min=args.cfg_renorm_min,
                cfg_renorm_type=args.cfg_renorm_type,
            )
            edited_image = image_result.get("image")
            if edited_image is not None:
                edited_path = mode_dir / f"edited_{args.reference_id}.png"
                edited_image.save(edited_path)
                mode_result["edited_image_path"] = str(edited_path)
                save_text(mode_dir / "image_edit_prompt.txt", image_edit_prompt)

        result_json_path = mode_dir / "result.json"
        result_json_path.write_text(json.dumps(ensure_serializable(mode_result), indent=2, ensure_ascii=False), encoding="utf-8")
        summary["results"][mode] = mode_result

    (output_dir / "summary.json").write_text(
        json.dumps(ensure_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(ensure_serializable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
