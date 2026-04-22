import os
import sys

import pandas as pd

os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from parse_arguments import parse_arguments
from experiments_ipcir_qwen import Experiment

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_CONFIG_FILE = "../config/start_config_circo_val_ipcir_qwen_pool50.json"


def json_config_start(args):
    config_file = getattr(args, "config", None) or DEFAULT_CONFIG_FILE
    config_data = pd.read_json(config_file)
    config_data = config_data.iloc[0].to_dict()
    for key, value in config_data.items():
        setattr(args, key, value)
    Experiment(args=args).run()


if __name__ == "__main__":
    args = parse_arguments()
    json_config_start(args)
