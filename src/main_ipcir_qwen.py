import os
import sys

import pandas as pd

# Always run relative imports from this src directory.
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from parse_arguments import parse_arguments
from experiments_ipcir_qwen import Experiment

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONFIG_FILE = "../config/start_config_circo_val_ipcir_qwen_pool50.json"

_DEBUG_KEYS = {
    "stage_mode",
    "task",
    "dataset",
    "split",
    "clip",
    "enable_multi_text_queries",
    "multi_text_queries_path",
    "ip_merge_text_feature_source",
    "ip_merge_use_similarity_override",
    "ip_merge_st_source",
    "image_generation_mode",
    "preload_modified_captions_file",
    "preload_edited_images_file",
    "dataset_path",
    "distributed_generate",
    "distributed_vqa",
    "device",
}

_CLI_BOOL_FLAGS = {
    "distributed_generate": ["--distributed_generate", "--distributed-generate"],
    "distributed_vqa": ["--distributed_vqa", "--distributed-vqa"],
    "bagel_use_multi_gpu": ["--bagel_use_multi_gpu", "--bagel-use-multi-gpu"],
}


def _coerce_bool_like(value):
    """Convert string bools from hand-edited JSON/configs while leaving other values intact."""
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    return value


def _cli_flag_present(names):
    argv = set(sys.argv[1:])
    return any(name in argv for name in names)


def _apply_cli_runtime_overrides(args):
    """Re-apply runtime flags after JSON loading.

    The project loads JSON after argparse. Therefore a config value such as
    "distributed_generate": false overwrites a command-line
    `--distributed_generate` flag unless we explicitly restore it here.
    This matters for torchrun jobs, where the shell script often appends
    --distributed_generate / --distributed_vqa to a shared config.
    """
    for attr, flag_names in _CLI_BOOL_FLAGS.items():
        if _cli_flag_present(flag_names):
            setattr(args, attr, True)
            print(f"[MAIN] CLI override: {attr}=True because one of {flag_names} was provided")


def json_config_start(args):
    config_file = getattr(args, "config", None) or DEFAULT_CONFIG_FILE
    print(f"[MAIN] __file__ = {__file__}")
    print(f"[MAIN] SRC_DIR = {SRC_DIR}")
    print(f"[MAIN] cwd = {os.getcwd()}")
    print(f"[MAIN] config_file = {config_file}")
    print(f"[MAIN] abs_config_file = {os.path.abspath(config_file)}")
    print(f"[MAIN] argv = {sys.argv}")
    print(f"[MAIN] env RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"[MAIN] config file not found: {config_file}")

    config_data = pd.read_json(config_file)
    if len(config_data) != 1:
        print(f"[MAIN][WARN] config has {len(config_data)} rows; using row 0.")
    config_data = config_data.iloc[0].to_dict()
    config_data = {k: _coerce_bool_like(v) for k, v in config_data.items()}

    print("[MAIN] loaded key experiment fields from JSON:")
    for key in sorted(_DEBUG_KEYS):
        if key in config_data:
            print(f"  {key}: {config_data[key]!r}")

    for key, value in config_data.items():
        setattr(args, key, value)

    # Important: the JSON loader above overwrites argparse booleans. Restore
    # runtime CLI flags explicitly.
    _apply_cli_runtime_overrides(args)

    print(f"[MAIN] args.stage_mode = {getattr(args, 'stage_mode', None)!r}")
    print(f"[MAIN] args.task = {getattr(args, 'task', None)!r}")
    print(f"[MAIN] args.enable_multi_text_queries = {getattr(args, 'enable_multi_text_queries', None)!r}")
    print(f"[MAIN] args.multi_text_queries_path = {getattr(args, 'multi_text_queries_path', None)!r}")
    print(f"[MAIN] args.ip_merge_text_feature_source = {getattr(args, 'ip_merge_text_feature_source', None)!r}")
    print(f"[MAIN] args.ip_merge_use_similarity_override = {getattr(args, 'ip_merge_use_similarity_override', None)!r}")
    print(f"[MAIN] args.ip_merge_st_source = {getattr(args, 'ip_merge_st_source', None)!r}")
    print(f"[MAIN] args.distributed_generate = {getattr(args, 'distributed_generate', None)!r}")
    print(f"[MAIN] args.distributed_vqa = {getattr(args, 'distributed_vqa', None)!r}")

    Experiment(args=args).run()


if __name__ == "__main__":
    args = parse_arguments()
    json_config_start(args)
