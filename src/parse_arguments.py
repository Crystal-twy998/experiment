import argparse
import prompts


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="Path to the json config file.")

    # === Base Arguments ===
    parser.add_argument("--exp_name", type=str, help="Experiment to evaluate")
    parser.add_argument("--device", type=int, default=0, help="GPU ID to use.")
    parser.add_argument(
        "--preload",
        nargs='+',
        type=str,
        default=['captions', 'mods'],
        help="List of properties to preload (computed once before)."
    )

    # === Base Model Choices ===
    parser.add_argument(
        "--clip",
        type=str,
        default='ViT-B-32',
        choices=[
            'ViT-Base/32', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
            'ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14'
        ],
        help="Which CLIP text-to-image retrieval model to use."
    )

    # === Dataset Arguments ===
    parser.add_argument(
        "--dataset",
        default="fashioniq_dress",
        type=str,
        required=False,
        choices=[
            'cirr', 'circo', 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
            'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute',
            'genecis_focus_object'
        ],
        help="Dataset to use"
    )
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'], help='Dataset split to evaluate on.')
    parser.add_argument("--dataset_path", default="../datasets/FASHIONIQ", type=str, required=False, help="Path to the dataset")
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'], help="Preprocess pipeline to use")

    # === LLM & BLIP Prompt Arguments ===
    available_prompts = [f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument(
        "--llm_prompt",
        default='prompts.structural_modifier_prompt_fashion',
        type=str,
        choices=available_prompts,
        help='Base prompt to use to probe the LLM. Must be available in prompts.py'
    )
    parser.add_argument("--openai_key", default="<your_openai_key_here>", type=str, help='Account key for OpenAI LLM usage.')

    # === Caption Checking Arguments ===
    parser.add_argument("--max_check_num", default=1, type=int, help='Maximum number of times the modified captions need to be checked.')

    # === LLM Model Arguments ===
    parser.add_argument(
        "--LLM_model_name",
        default="gpt-4o-mini",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "qwen-turbo", "bagel"],
        help='LLM model name to use.'
    )
    parser.add_argument(
        "--Check_LLM_model_name",
        default="gpt_4o_mini",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "qwen-turbo", "qwen2_5_vl_7b_instruct"],
        help='LLM model name to check modified captions.'
    )
    parser.add_argument(
        "--VQA_LLM_model_name",
        default="qwen2_5_vl_7b_instruct",
        type=str,
        choices=[
            "qwen2_5_vl_7b_instruct",
            "qwen2_5_vl_3b_instruct",
            "qwen2_vl_7b_instruct",
            "qwen2_5_vl_32b_instruct",
            "qwen2_5_vl_72b_instruct",
            "qwen3_vl_8b_instruct",
            "qwen-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        help='VQA verifier model name.'
    )

    # === Local VQA model paths ===
    parser.add_argument("--qwen2_5_vl_7b_path", type=str, default=None)
    parser.add_argument("--qwen2_vl_7b_path", type=str, default=None)
    parser.add_argument("--qwen2_5_vl_3b_path", type=str, default=None)
    parser.add_argument("--qwen2_5_vl_32b_path", type=str, default=None)
    parser.add_argument("--qwen2_5_vl_72b_path", type=str, default=None)
    parser.add_argument("--qwen3_vl_8b_path", type=str, default=None)

    parser.add_argument("--distributed_vqa", action="store_true", help="Shard Qwen VQA candidate scoring across ranks.")
    parser.add_argument("--vqa_cleanup_every", type=int, default=16, help="Clear CUDA cache every N local samples during VQA.")
    parser.add_argument("--vqa_attn_implementation", type=str, default="sdpa")
    parser.add_argument("--vqa_image_max_size", type=int, default=1024)
    parser.add_argument("--vqa_min_pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--vqa_max_pixels", type=int, default=2048 * 28 * 28)

    # === Distributed execution ===
    parser.add_argument("--distributed_generate", action="store_true", help="Shard BAGEL caption/image generation across ranks.")
    parser.add_argument(
        "--image_generation_mode",
        type=str,
        default="instruction_plus_target",
        choices=["instruction_only", "target_only", "instruction_plus_target"],
        help="Stage-1 image query generation mode."
    )
    parser.add_argument("--bagel_use_multi_gpu", action="store_true", help="Use multi-GPU model-parallel loading for BAGEL in a single process.")

    # === Text-to-Image Retrieval Arguments ===
    parser.add_argument("--retrieval", type=str, default='default', choices=['default'], help='Type of T2I Retrieval method.')

    return parser.parse_args()
