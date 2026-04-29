#!/usr/bin/env python3
"""
Single-sample BAGEL demo aligned with the current experiment/src codebase.

New in this v2 version:
1. Keeps the original four prompt modes for single-sample analysis.
2. Adds a lightweight image-branch refinement demo:
   - direct draft generation
   - planned draft generation
   - self-critique using BAGEL (via side-by-side comparison image)
   - 1~N refinement rounds
   - select the best proxy image for later retrieval use

This is still a single-sample analysis/debug tool, not the full batch pipeline.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
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
    "reasoning": [
        "Reasoning Enhanced Query",
        "Reasoning-Enhanced Query",
        "Reasoning Query",
    ],
}


# =========================
# Basic utils
# =========================
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
    parser = argparse.ArgumentParser(
        description="Single-sample BAGEL demo aligned to experiment/src"
    )
    parser.add_argument(
        "--repo_src_root",
        type=str,
        required=True,
        help="Path to experiment/src directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local BAGEL checkpoint directory.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Reference image path.",
    )
    parser.add_argument(
        "--image_caption",
        type=str,
        default="",
        help="Reference image caption for structural_modifier mode.",
    )
    parser.add_argument(
        "--edit_instruction",
        type=str,
        required=True,
        help="Relative caption / edit instruction.",
    )
    parser.add_argument(
        "--reference_id",
        type=str,
        default="sample",
        help="Sample id for bookkeeping.",
    )
    parser.add_argument("--output_dir", type=str, default="./bagel_demo_outputs_latest")

    parser.add_argument(
        "--prompt_modes",
        nargs="+",
        default=[
            "structural_modifier",
            "mllm_cot",
            "image_mllm_cot",
            "mllm_cot_multi",
        ],
        choices=list(PROMPT_MODE_TO_NAME.keys()),
    )
    parser.add_argument(
        "--multi_query_choice",
        type=str,
        default="conservative",
        choices=["conservative", "balanced", "reasoning"],
        help="Which multi-query branch to use when optionally generating edited image.",
    )

    # original optional image generation for each prompt mode
    parser.add_argument(
        "--generate_images",
        action="store_true",
        help="Also generate edited images using edit_image_no_think for each prompt mode.",
    )
    parser.add_argument(
        "--image_edit_prompt_mode",
        type=str,
        default="instruction_plus_target",
        choices=["instruction_only", "target_text_only", "instruction_plus_target"],
        help="Prompt composition for the optional edited-image generation step.",
    )

    # new image-branch refine demo
    parser.add_argument(
        "--run_image_branch_refine",
        action="store_true",
        help="Run the new direct/planned/refine image-branch demo.",
    )
    parser.add_argument(
        "--planning_mode",
        type=str,
        default="image_mllm_cot",
        choices=["structural_modifier", "mllm_cot", "image_mllm_cot", "mllm_cot_multi"],
        help="Which prompt mode result to use as the detailed planning description.",
    )
    parser.add_argument(
        "--planning_multi_choice",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "reasoning"],
        help="If planning_mode is mllm_cot_multi, which branch to use for planning.",
    )
    parser.add_argument(
        "--critique_max_think_token_n",
        type=int,
        default=256,
        help="max_think_token_n for BAGEL self-critique.",
    )
    parser.add_argument(
        "--refine_rounds",
        type=int,
        default=1,
        help="How many refinement rounds to run after selecting the best initial draft.",
    )
    parser.add_argument(
        "--refine_from",
        type=str,
        default="reference",
        choices=["reference", "best_initial"],
        help=(
            "If reference: refine by re-editing from the original reference image. "
            "If best_initial: refine by editing the selected best draft itself."
        ),
    )
    parser.add_argument(
        "--save_compare_canvas",
        action="store_true",
        help="Save side-by-side comparison images used for critique.",
    )

    parser.add_argument("--max_think_token_n", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--offload_folder", type=str, default="/tmp/offload_bagel_demo")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda:0",
    )
    parser.add_argument("--use_multi_gpu", action="store_true")

    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_start", type=float, default=0.0)
    parser.add_argument("--cfg_interval_end", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument(
        "--cfg_renorm_type",
        type=str,
        default="text_channel",
        choices=["global", "channel", "text_channel"],
    )

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
    bagel_inference = load_module_from_path(
        "bagel_inference_local", repo_src_root / "bagel_inference.py"
    )
    prompts_module = load_module_from_path(
        "prompts_local", repo_src_root / "prompts.py"
    )
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
                return text[start : i + 1]
    return None


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(ensure_serializable(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =========================
# Parsing outputs
# =========================
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
        pattern = rf"{re.escape(key)}\s*[:：]\s*(.+)"
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

    patterns = {
        "Conservative Query": (
            r'Conservative Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?'
            r'rationale\s*[:：]\s*"?([^"\n]+)"?'
        ),
        "Balanced Query": (
            r'Balanced Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?'
            r'rationale\s*[:：]\s*"?([^"\n]+)"?'
        ),
        "Reasoning Enhanced Query": (
            r'Reasoning[ -]?Enhanced Query.*?description\s*[:：]\s*"?([^"\n]+)"?.*?'
            r'rationale\s*[:：]\s*"?([^"\n]+)"?'
        ),
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


def parse_critique_result(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    result: Dict[str, Any] = {
        "raw_text": text,
        "edit_fidelity": 0.0,
        "preserve_fidelity": 0.0,
        "main_errors": [],
        "refinement_instruction": "",
        "brief_reason": "",
        "parser": "fallback",
    }

    if not text:
        return result

    json_block = extract_json_substring(text)
    if json_block:
        try:
            obj = json.loads(json_block)
            result["edit_fidelity"] = float(obj.get("edit_fidelity", 0.0))
            result["preserve_fidelity"] = float(obj.get("preserve_fidelity", 0.0))
            result["main_errors"] = obj.get("main_errors", []) or []
            result["refinement_instruction"] = obj.get("refinement_instruction", "") or ""
            result["brief_reason"] = obj.get("brief_reason", "") or ""
            result["parser"] = "json"
            result["json"] = obj
            return result
        except Exception:
            pass

    # fallback: very loose parsing
    score_keys = {
        "edit_fidelity": r"edit_fidelity\s*[:：]\s*([0-5](?:\.\d+)?)",
        "preserve_fidelity": r"preserve_fidelity\s*[:：]\s*([0-5](?:\.\d+)?)",
        "refinement_instruction": r"refinement_instruction\s*[:：]\s*(.+)",
        "brief_reason": r"brief_reason\s*[:：]\s*(.+)",
    }
    for key, pattern in score_keys.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            if key in ["edit_fidelity", "preserve_fidelity"]:
                result[key] = float(m.group(1))
            else:
                result[key] = m.group(1).strip()

    err_match = re.search(r"main_errors\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if err_match:
        raw_err = err_match.group(1).strip()
        result["main_errors"] = [x.strip() for x in re.split(r"[;,]", raw_err) if x.strip()]

    return result


def critique_score(parsed: Dict[str, Any]) -> float:
    edit_f = float(parsed.get("edit_fidelity", 0.0))
    preserve_f = float(parsed.get("preserve_fidelity", 0.0))
    err_penalty = 0.2 * len(parsed.get("main_errors", []) or [])
    return 0.7 * edit_f + 0.3 * preserve_f - err_penalty


# =========================
# Prompt builders
# =========================
def get_prompt_text(prompts_module, prompt_name: str) -> str:
    if not hasattr(prompts_module, prompt_name):
        raise AttributeError(f"Prompt {prompt_name} not found in prompts.py")
    prompt = getattr(prompts_module, prompt_name)
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt {prompt_name} is not a string")
    return prompt


def build_caption_prompt(base_prompt: str, image_caption: str, edit_instruction: str) -> str:
    return (
        base_prompt
        + "\n"
        + f"Image Content: {image_caption}"
        + "\n"
        + f"Instruction: {edit_instruction}"
    )


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


def build_instruction_plus_plan_prompt(instruction: str, detailed_plan: str) -> str:
    detailed_plan = (detailed_plan or "").strip()
    if not detailed_plan:
        return instruction
    return (
        "Edit this reference image according to the modification text.\n"
        f"Modification Text: {instruction}\n"
        f"Detailed Edit Plan: {detailed_plan}\n"
        "Please preserve irrelevant content and only make the requested changes."
        # "Generate one plausible target image that satisfies the user's intent given the reference image and the modification text.\n"
        # "Keep the shared concepts that remain compatible with the request.\n"
        # "Modify, remove, or replace concepts that conflict with the requested target.\n"
        # "Do not preserve details merely because they appear in the reference image if they are irrelevant to the intended target.\n"
        # "The result should be visually coherent, commonsense, and faithful to the intended composition."
    )

# def build_instruction_plus_plan_prompt(instruction: str, detailed_plan: str) -> str:
#     detailed_plan = (detailed_plan or "").strip()
#     detail_block = f"Detailed Target Hypothesis: {detailed_plan}\n" if detailed_plan else ""
#     return (
#         "Generate one plausible target image that matches the user's intent "
#         "given the reference image and the modification text.\n"
#         f"Modification Text: {instruction}\n"
#         f"{detail_block}"
#         "Keep shared concepts from the reference image when they remain compatible with the intended target.\n"
#         "Modify, remove, or replace concepts that conflict with the intended target.\n"
#         "Do not preserve details only because they appear in the reference image if they are irrelevant.\n"
#         "The result should be visually coherent, commonsense, and faithful to the intended target image.\n"
#     )

def build_proxy_critique_prompt(edit_instruction: str, reference_caption: str = "") -> str:
    ref_caption_block = (
        f"Reference Caption: {reference_caption}\n" if reference_caption.strip() else ""
    )
    return (
        "You are evaluating a generated proxy image for composed image retrieval.\n"
        "The input image is a side-by-side comparison:\n"
        "- Left: reference image\n"
        "- Right: generated proxy image\n\n"
        f"{ref_caption_block}"
        f"Modification Text: {edit_instruction}\n\n"
        "Please judge whether the right image correctly applies the modification to the left image "
        "while preserving irrelevant visual content.\n"
        "Return ONLY valid JSON in the following format:\n"
        "{\n"
        '  "edit_fidelity": 0-5,\n'
        '  "preserve_fidelity": 0-5,\n'
        '  "main_errors": ["short error 1", "short error 2"],\n'
        '  "refinement_instruction": "one concise refinement instruction",\n'
        '  "brief_reason": "one short sentence"\n'
        "}"
    )

# def build_proxy_critique_prompt(edit_instruction: str, reference_caption: str = "") -> str:
#     ref_caption_block = f"Reference Caption: {reference_caption}\n" if reference_caption.strip() else ""
#     return (
#         "You are evaluating whether a generated target image is a plausible result for composed image retrieval.\n"
#         "The input image is a side-by-side comparison:\n"
#         "- Left: reference image\n"
#         "- Right: generated target hypothesis\n\n"
#         f"{ref_caption_block}"
#         f"Modification Text: {edit_instruction}\n\n"
#         "Judge the right image according to the following criteria:\n"
#         "1. intent_alignment: does it satisfy the user intent implied by the reference image and modification text?\n"
#         "2. shared_concept_consistency: does it keep the important shared concepts that should remain?\n"
#         "3. conflict_resolution: does it correctly revise/remove concepts that conflict with the target?\n"
#         "4. commonsense_plausibility: is the result visually coherent and reasonable?\n\n"
#         "Return ONLY valid JSON in the following format:\n"
#         "{\n"
#         '  "intent_alignment": 0-5,\n'
#         '  "shared_concept_consistency": 0-5,\n'
#         '  "conflict_resolution": 0-5,\n'
#         '  "commonsense_plausibility": 0-5,\n'
#         '  "main_errors": ["short error 1", "short error 2"],\n'
#         '  "refinement_instruction": "one concise refinement instruction",\n'
#         '  "brief_reason": "one short sentence"\n'
#         "}"
#     )


def build_refine_prompt(
    edit_instruction: str,
    critique: Dict[str, Any],
    detailed_plan: str = "",
) -> str:
    refine_inst = critique.get("refinement_instruction", "") or ""
    main_errors = critique.get("main_errors", []) or []
    error_text = "; ".join(main_errors) if main_errors else "No explicit errors extracted."
    detail_block = f"Detailed Edit Plan: {detailed_plan}\n" if detailed_plan.strip() else ""
    return (
        "Edit this image again to better satisfy the modification request.\n"
        f"Modification Text: {edit_instruction}\n"
        f"{detail_block}"
        f"Problems in the previous generated image: {error_text}\n"
        f"Refinement Instruction: {refine_inst}\n"
        "Preserve unchanged content and only correct the missing or incorrect edits."
    )

# def build_refine_prompt(
#     edit_instruction: str,
#     critique: Dict[str, Any],
#     detailed_plan: str = "",
# ) -> str:
#     refine_inst = critique.get("refinement_instruction", "") or ""
#     main_errors = critique.get("main_errors", []) or []
#     error_text = "; ".join(main_errors) if main_errors else "No explicit errors extracted."
#     detail_block = f"Detailed Target Hypothesis: {detailed_plan}\n" if detailed_plan.strip() else ""
#     return (
#         "Refine this generated image so that it better matches the user's intended target.\n"
#         f"Modification Text: {edit_instruction}\n"
#         f"{detail_block}"
#         f"Problems in the current generated image: {error_text}\n"
#         f"Refinement Guidance: {refine_inst}\n"
#         "Retain the shared concepts that are still compatible with the target.\n"
#         "Revise or discard concepts that conflict with the intended target.\n"
#         "The refined image should be plausible, visually coherent, and consistent with commonsense.\n"
#     )


# =========================
# BAGEL calls
# =========================
def run_text_only_prompt(
    editor,
    prompt_text: str,
    max_think_token_n: int,
    do_sample: bool,
) -> str:
    return editor.generate_caption(
        prompt_text,
        max_think_token_n=max_think_token_n,
        do_sample=do_sample,
    )


def run_image_conditioned_prompt(
    editor,
    image: Image.Image,
    prompt_text: str,
    max_think_token_n: int,
    do_sample: bool,
) -> str:
    output_dict = editor.inferencer(
        image=image,
        text=prompt_text,
        understanding_output=True,
        max_think_token_n=max_think_token_n,
        do_sample=do_sample,
    )
    return (output_dict.get("text") or "").strip()


def run_edit_image(editor, image_path: str, prompt: str, args: argparse.Namespace) -> Image.Image:
    image_result = editor.edit_image_no_think(
        image_path=image_path,
        prompt=prompt,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[args.cfg_interval_start, args.cfg_interval_end],
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )
    edited_image = image_result.get("image")
    if edited_image is None:
        raise RuntimeError("BAGEL edit_image_no_think returned no image.")
    return edited_image


# =========================
# Image helpers
# =========================
def make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    target_h = max(left.height, right.height)
    target_w = max(left.width, right.width)

    left_r = left.resize((target_w, target_h))
    right_r = right.resize((target_w, target_h))

    canvas = Image.new("RGB", (target_w * 2, target_h + 40), color=(255, 255, 255))
    canvas.paste(left_r, (0, 40))
    canvas.paste(right_r, (target_w, 40))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Left: Reference", fill=(0, 0, 0))
    draw.text((target_w + 10, 10), "Right: Generated Proxy", fill=(0, 0, 0))
    return canvas


def critique_proxy(
    editor,
    reference_image: Image.Image,
    candidate_image: Image.Image,
    edit_instruction: str,
    reference_caption: str,
    critique_max_think_token_n: int,
    do_sample: bool,
    save_compare_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    compare = make_side_by_side(reference_image, candidate_image)
    if save_compare_path is not None:
        compare.save(save_compare_path)

    critique_prompt = build_proxy_critique_prompt(
        edit_instruction=edit_instruction,
        reference_caption=reference_caption,
    )
    raw = run_image_conditioned_prompt(
        editor=editor,
        image=compare,
        prompt_text=critique_prompt,
        max_think_token_n=critique_max_think_token_n,
        do_sample=do_sample,
    )
    parsed = parse_critique_result(raw)
    parsed["score"] = critique_score(parsed)
    return raw, parsed


# =========================
# Planning helpers
# =========================
def select_multi_query(multi_result: Dict[str, Any], choice: str) -> Optional[Dict[str, Any]]:
    queries = multi_result.get("queries") or {}
    for key in MULTI_KEY_ALIASES[choice]:
        if key in queries and isinstance(queries[key], dict):
            desc = queries[key].get("description", "")
            rationale = queries[key].get("rationale", "")
            return {"key": key, "description": desc, "rationale": rationale}
    return None


def get_mode_chosen_target(
    mode: str,
    raw_output: str,
    multi_choice: str,
) -> Tuple[Dict[str, Any], str]:
    if mode == "mllm_cot_multi":
        parsed_multi = parse_multi_queries(raw_output)
        selected = select_multi_query(parsed_multi, multi_choice)
        chosen_target = selected["description"] if selected else ""
        mode_result = {
            "mode": mode,
            "parsed": parsed_multi,
            "selected_query": selected,
            "chosen_target_description": chosen_target,
        }
        return mode_result, chosen_target
    else:
        parsed = parse_single_description(raw_output)
        chosen_target = parsed.get("parsed_description", "")
        mode_result = {
            "mode": mode,
            "parsed": parsed,
            "chosen_target_description": chosen_target,
        }
        return mode_result, chosen_target


def extract_planning_description(summary_results: Dict[str, Any], planning_mode: str) -> str:
    if planning_mode not in summary_results:
        return ""

    item = summary_results[planning_mode]
    chosen = item.get("chosen_target_description", "") or ""

    # for multi query, chosen_target_description should already be set
    return chosen.strip()


# =========================
# Main image-branch refine demo
# =========================
def run_image_branch_refine_demo(
    args: argparse.Namespace,
    editor,
    ref_image: Image.Image,
    image_path: Path,
    summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    image_branch_dir = output_dir / "image_branch_refine"
    image_branch_dir.mkdir(parents=True, exist_ok=True)

    planning_desc = extract_planning_description(summary["results"], args.planning_mode)
    planning_info = {
        "planning_mode": args.planning_mode,
        "planning_multi_choice": args.planning_multi_choice,
        "planning_desc": planning_desc,
    }

    # -------------------------
    # 1) Generate two initial drafts
    # -------------------------
    direct_prompt = build_image_edit_prompt(
        mode="instruction_only",
        instruction=args.edit_instruction,
        target_text="",
    )
    planned_prompt = build_instruction_plus_plan_prompt(
        instruction=args.edit_instruction,
        detailed_plan=planning_desc,
    )

    save_text(image_branch_dir / "draft_direct_prompt.txt", direct_prompt)
    save_text(image_branch_dir / "draft_planned_prompt.txt", planned_prompt)

    print("[Image-Branch] Generating direct draft...")
    direct_img = run_edit_image(editor, str(image_path), direct_prompt, args)
    direct_path = image_branch_dir / "draft_direct.png"
    direct_img.save(direct_path)

    print("[Image-Branch] Generating planned draft...")
    planned_img = run_edit_image(editor, str(image_path), planned_prompt, args)
    planned_path = image_branch_dir / "draft_planned.png"
    planned_img.save(planned_path)

    # -------------------------
    # 2) Critique the two initial drafts
    # -------------------------
    print("[Image-Branch] Critiquing direct draft...")
    direct_raw, direct_crit = critique_proxy(
        editor=editor,
        reference_image=ref_image,
        candidate_image=direct_img,
        edit_instruction=args.edit_instruction,
        reference_caption=args.image_caption,
        critique_max_think_token_n=args.critique_max_think_token_n,
        do_sample=False,
        save_compare_path=(
            image_branch_dir / "compare_direct.png" if args.save_compare_canvas else None
        ),
    )
    save_text(image_branch_dir / "critique_direct_raw.txt", direct_raw)
    save_json(image_branch_dir / "critique_direct.json", direct_crit)

    print("[Image-Branch] Critiquing planned draft...")
    planned_raw, planned_crit = critique_proxy(
        editor=editor,
        reference_image=ref_image,
        candidate_image=planned_img,
        edit_instruction=args.edit_instruction,
        reference_caption=args.image_caption,
        critique_max_think_token_n=args.critique_max_think_token_n,
        do_sample=False,
        save_compare_path=(
            image_branch_dir / "compare_planned.png" if args.save_compare_canvas else None
        ),
    )
    save_text(image_branch_dir / "critique_planned_raw.txt", planned_raw)
    save_json(image_branch_dir / "critique_planned.json", planned_crit)

    if direct_crit["score"] >= planned_crit["score"]:
        selected_initial_name = "direct"
        selected_initial_path = direct_path
        selected_initial_img = direct_img
        selected_initial_crit = direct_crit
    else:
        selected_initial_name = "planned"
        selected_initial_path = planned_path
        selected_initial_img = planned_img
        selected_initial_crit = planned_crit

    # -------------------------
    # 3) Refinement rounds
    # -------------------------
    refine_records: List[Dict[str, Any]] = []
    current_best_name = selected_initial_name
    current_best_path = selected_initial_path
    current_best_img = selected_initial_img
    current_best_crit = selected_initial_crit

    for r in range(args.refine_rounds):
        round_id = r + 1
        print(f"[Image-Branch] Refinement round {round_id}...")

        refine_prompt = build_refine_prompt(
            edit_instruction=args.edit_instruction,
            critique=current_best_crit,
            detailed_plan=planning_desc,
        )
        save_text(image_branch_dir / f"refine_round_{round_id}_prompt.txt", refine_prompt)

        refine_source_path = (
            str(image_path)
            if args.refine_from == "reference"
            else str(current_best_path)
        )

        refined_img = run_edit_image(editor, refine_source_path, refine_prompt, args)
        refined_path = image_branch_dir / f"draft_refined_round_{round_id}.png"
        refined_img.save(refined_path)

        refined_raw, refined_crit = critique_proxy(
            editor=editor,
            reference_image=ref_image,
            candidate_image=refined_img,
            edit_instruction=args.edit_instruction,
            reference_caption=args.image_caption,
            critique_max_think_token_n=args.critique_max_think_token_n,
            do_sample=False,
            save_compare_path=(
                image_branch_dir / f"compare_refined_round_{round_id}.png"
                if args.save_compare_canvas
                else None
            ),
        )
        save_text(image_branch_dir / f"critique_refined_round_{round_id}_raw.txt", refined_raw)
        save_json(image_branch_dir / f"critique_refined_round_{round_id}.json", refined_crit)

        record = {
            "round": round_id,
            "refine_from": args.refine_from,
            "source_name": current_best_name,
            "source_path": str(current_best_path),
            "refined_path": str(refined_path),
            "refine_prompt": refine_prompt,
            "critique_raw": refined_raw,
            "critique": refined_crit,
        }
        refine_records.append(record)

        # Greedy update: keep the better one
        if refined_crit["score"] >= current_best_crit["score"]:
            current_best_name = f"refined_round_{round_id}"
            current_best_path = refined_path
            current_best_img = refined_img
            current_best_crit = refined_crit

    # -------------------------
    # 4) Final summary
    # -------------------------
    image_branch_summary: Dict[str, Any] = {
        "planning": planning_info,
        "initial_drafts": {
            "direct": {
                "image_path": str(direct_path),
                "edit_prompt": direct_prompt,
                "critique": direct_crit,
            },
            "planned": {
                "image_path": str(planned_path),
                "edit_prompt": planned_prompt,
                "critique": planned_crit,
            },
            "selected_initial": selected_initial_name,
        },
        "refinement": {
            "refine_rounds": args.refine_rounds,
            "refine_from": args.refine_from,
            "records": refine_records,
        },
        "final_selection": {
            "name": current_best_name,
            "image_path": str(current_best_path),
            "critique": current_best_crit,
        },
    }

    save_json(image_branch_dir / "image_branch_summary.json", image_branch_summary)
    return image_branch_summary


# =========================
# Main
# =========================
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

    # -------------------------
    # Original stage1 prompt analysis
    # -------------------------
    for mode in args.prompt_modes:
        prompt_name = PROMPT_MODE_TO_NAME[mode]
        base_prompt = get_prompt_text(prompts_module, prompt_name)

        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        if mode == "structural_modifier":
            if not args.image_caption.strip():
                raise ValueError("--image_caption is required for structural_modifier mode.")
            final_prompt = build_caption_prompt(
                base_prompt, args.image_caption, args.edit_instruction
            )
            raw_output = run_text_only_prompt(
                editor, final_prompt, args.max_think_token_n, args.do_sample
            )
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
            final_prompt = build_image_understanding_prompt(
                base_prompt, args.edit_instruction
            )
            raw_output = run_image_conditioned_prompt(
                editor, ref_image, final_prompt, args.max_think_token_n, args.do_sample
            )
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
            final_prompt = build_image_understanding_prompt(
                base_prompt, args.edit_instruction
            )
            raw_output = run_image_conditioned_prompt(
                editor, ref_image, final_prompt, args.max_think_token_n, args.do_sample
            )
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

        # original optional image generation for each prompt mode
        if args.generate_images and chosen_target:
            image_edit_prompt = build_image_edit_prompt(
                args.image_edit_prompt_mode,
                args.edit_instruction,
                chosen_target,
            )
            edited_image = run_edit_image(editor, str(image_path), image_edit_prompt, args)
            edited_path = mode_dir / f"edited_{args.reference_id}.png"
            edited_image.save(edited_path)
            mode_result["edited_image_path"] = str(edited_path)
            save_text(mode_dir / "image_edit_prompt.txt", image_edit_prompt)

        result_json_path = mode_dir / "result.json"
        save_json(result_json_path, mode_result)
        summary["results"][mode] = mode_result

    # -------------------------
    # New image branch refinement demo
    # -------------------------
    if args.run_image_branch_refine:
        image_branch_summary = run_image_branch_refine_demo(
            args=args,
            editor=editor,
            ref_image=ref_image,
            image_path=image_path,
            summary=summary,
            output_dir=output_dir,
        )
        summary["image_branch_refine"] = image_branch_summary

    # -------------------------
    # Save global summary
    # -------------------------
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(ensure_serializable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
