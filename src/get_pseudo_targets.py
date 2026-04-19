import json
import requests
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from enum import Enum, auto
import math
import base64
import gc

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from qwen_vl_utils import process_vision_info

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


class CheckModel(Enum):
    gpt_4o = auto()
    gpt_4o_mini = auto()
    gpt_3_5_turbo = auto()
    qwen_turbo = auto()
    qwen2_5_vl_7b_instruct = auto()
    qwen2_vl_7b_instruct = auto()
    qwen2_5_vl_72b_instruct = auto()
    qwen2_5_vl_32b_instruct = auto()
    qwen2_5_vl_3b_instruct = auto()
    qwen3_vl_8b_instruct = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return CheckModel[s.replace("-", "_").replace(".", "_").lower()]
        except KeyError:
            raise ValueError(f"Unknown model type: {s}")


class VQAModelHandler:
    def __init__(
        self,
        model_type: str,
        device: torch.device,
        openai_key: str = None,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 2048 * 28 * 28,
        image_max_size: int = 1024,
        attn_implementation: str = "sdpa",
        model_path_overrides: Optional[Dict[str, str]] = None,
    ):
        self.model_type = CheckModel.from_string(model_type.replace("-", "_"))
        self.device = device
        self.openai_key = openai_key
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_max_size = image_max_size
        self.attn_implementation = attn_implementation
        self.model_path_overrides = model_path_overrides or {}
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.y_token_id = None
        self.n_token_id = None

        if self.model_type in [CheckModel.gpt_4o, CheckModel.gpt_4o_mini, CheckModel.qwen_turbo]:
            self.chat_function = self._chat_api
        else:
            self.model, self.processor = self._load_model_and_process(device)
            self.y_token_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            self.n_token_id = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]
            self.chat_function = self._chat_local

    def release(self):
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "processor") and self.processor is not None:
                del self.processor
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _resolve_model_name(self) -> str:
        defaults = {
            CheckModel.qwen2_5_vl_7b_instruct: "/data1/shared_models/Qwen2.5-VL-7B-Instruct",
            CheckModel.qwen2_vl_7b_instruct: "Qwen/Qwen2-VL-7B-Instruct",
            CheckModel.qwen2_5_vl_32b_instruct: "Qwen/Qwen2.5-VL-32B-Instruct",
            CheckModel.qwen2_5_vl_3b_instruct: "Qwen/Qwen2.5-VL-3B-Instruct",
            CheckModel.qwen2_5_vl_72b_instruct: "Qwen/Qwen2.5-VL-72B-Instruct",
            CheckModel.qwen3_vl_8b_instruct: "/nativemm/share/cpfs/tangwenyue/models/Qwen3-VL-8B-Instruct",
        }
        override_keys = {
            CheckModel.qwen2_5_vl_7b_instruct: ["qwen2_5_vl_7b_path", "qwen2_5_vl_7b_instruct_path"],
            CheckModel.qwen2_vl_7b_instruct: ["qwen2_vl_7b_path", "qwen2_vl_7b_instruct_path"],
            CheckModel.qwen2_5_vl_32b_instruct: ["qwen2_5_vl_32b_path", "qwen2_5_vl_32b_instruct_path"],
            CheckModel.qwen2_5_vl_3b_instruct: ["qwen2_5_vl_3b_path", "qwen2_5_vl_3b_instruct_path"],
            CheckModel.qwen2_5_vl_72b_instruct: ["qwen2_5_vl_72b_path", "qwen2_5_vl_72b_instruct_path"],
            CheckModel.qwen3_vl_8b_instruct: ["qwen3_vl_8b_path", "qwen3_vl_8b_instruct_path"],
        }

        candidate = None
        for key in override_keys.get(self.model_type, []):
            value = self.model_path_overrides.get(key)
            if value:
                candidate = value
                break
        if candidate is None:
            candidate = defaults.get(self.model_type)
        if candidate is None:
            raise ValueError(f"Unsupported local model: {self.model_type}")
        return candidate

    @staticmethod
    def _is_local_path(model_name: str) -> bool:
        return os.path.exists(model_name) or model_name.startswith("/") or model_name.startswith("./")

    def _build_qwen3_model(self, model_name: str, local_files_only: bool):
        load_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": {"": self.device},
            "attn_implementation": self.attn_implementation,
            "local_files_only": local_files_only,
        }

        if Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(**load_kwargs)
            model.eval()
            return model

        if AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
            model.eval()
            return model

        if AutoModelForVision2Seq is not None:
            model = AutoModelForVision2Seq.from_pretrained(**load_kwargs)
            model.eval()
            return model

        raise ImportError(
            "Qwen3-VL requires a newer transformers build. "
            "Please install a transformers version that provides "
            "Qwen3VLForConditionalGeneration or AutoModelForImageTextToText."
        )

    def _load_model_and_process(self, device: torch.device):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        model_name = self._resolve_model_name()
        local_files_only = self._is_local_path(model_name)

        if self.model_type == CheckModel.qwen2_5_vl_7b_instruct:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only, **processor_kwargs)
            processor.tokenizer.padding_side = 'left'
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": device},
                attn_implementation=self.attn_implementation,
                local_files_only=local_files_only,
            )
            model.eval()
            return model, processor

        elif self.model_type == CheckModel.qwen2_vl_7b_instruct:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only, **processor_kwargs)
            processor.tokenizer.padding_side = 'left'
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map={"": device},
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
            model.eval()
            return model, processor

        elif self.model_type in [CheckModel.qwen2_5_vl_32b_instruct, CheckModel.qwen2_5_vl_3b_instruct, CheckModel.qwen2_5_vl_72b_instruct]:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only, **processor_kwargs)
            processor.tokenizer.padding_side = 'left'
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": device},
                attn_implementation=self.attn_implementation,
                local_files_only=local_files_only,
            )
            model.eval()
            return model, processor

        elif self.model_type == CheckModel.qwen3_vl_8b_instruct:
            processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only, **processor_kwargs)
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.padding_side = 'left'
            model = self._build_qwen3_model(model_name, local_files_only)
            return model, processor

        raise ValueError(f"Unsupported local model: {self.model_type}")

    def _send_request(self, url, headers, payload, max_retries=5, backoff=0.25):
        for _ in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code != 200:
                    print(f"HTTP {resp.status_code}: {resp.text}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                data = resp.json()
                if 'error' in data:
                    print(f"API Error: {data['error']}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return data
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                time.sleep(backoff)
                backoff *= 2
        return None

    def load_image(self, image_path, max_size=None):
        max_size = self.image_max_size if max_size is None else max_size
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return Image.new('RGB', (224, 224), color='white')

    @torch.inference_mode()
    def _chat_local(self, ref_image_path, relative_caption, cand_image_path, device, max_length=800):
        ref_image = self.load_image(ref_image_path)
        cand_image = self.load_image(cand_image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref_image},
                    {"type": "image", "image": cand_image},
                    {
                        "type": "text",
                        "text": f"""You are a strict visual verifier. Output exactly one token: yes or no (lowercase). Do not add punctuation or explanations.
Reference image: Picture1
Candidate image: Picture2
Instruction: {relative_caption}
Decide if the candidate image matches the result of applying the instruction to the reference image.
Return yes if all required elements implied by the instruction are satisfied (like counts, categories, attributes, spatial relations). If any required element is missing or contradicted, answer no.
Answer:"""
                    }
                ]
            }
        ]

        if self.model_type == CheckModel.qwen3_vl_8b_instruct:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_vision_id=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt"
            )

        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        try:
            outputs = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
        except torch.cuda.OutOfMemoryError as oom:
            for tensor_name in list(inputs.keys()):
                if torch.is_tensor(inputs[tensor_name]):
                    del inputs[tensor_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise torch.cuda.OutOfMemoryError(
                f"VQA OOM with image_max_size={self.image_max_size}, "
                f"min_pixels={self.min_pixels}, max_pixels={self.max_pixels}, "
                f"attn_implementation={self.attn_implementation}"
            ) from oom

        logits = outputs.logits[:, -1, :]
        y_logits = logits[:, self.y_token_id]
        n_logits = logits[:, self.n_token_id]
        y_n_logits = torch.stack([n_logits, y_logits], dim=1)
        probs = F.softmax(y_n_logits, dim=1)
        confidence_y = float(probs[:, 1].detach().cpu().item())

        del outputs, logits, y_logits, n_logits, y_n_logits, probs
        for tensor_name in list(inputs.keys()):
            if torch.is_tensor(inputs[tensor_name]):
                del inputs[tensor_name]
        return confidence_y

    def _chat_api(self, ref_image_path, relative_caption, cand_image_path, max_length=800):
        model_map = {
            CheckModel.gpt_4o: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o"
            },
            CheckModel.gpt_4o_mini: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o-mini"
            },
            CheckModel.gpt_3_5_turbo: {
                "url": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-3.5-turbo"
            },
            CheckModel.qwen_turbo: {
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model_name": "qwen-plus"
            }
        }

        if self.model_type not in model_map:
            raise ValueError("Unsupported API model.")

        model_info = model_map[self.model_type]
        ref_b64 = self._encode_image_base64(ref_image_path)
        cand_b64 = self._encode_image_base64(cand_image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a strict visual verifier. Output exactly one character: Y or N."},
                    {"type": "text", "text": f"Instruction: {relative_caption}"},
                    {"type": "text", "text": "Reference image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                    {"type": "text", "text": "Candidate image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cand_b64}"}},
                    {"type": "text", "text": "Decide if the candidate image matches the result of applying the instruction to the reference image. Return Y if ALL required elements are satisfied (like counts, categories, attributes, spatial relations). Otherwise return N. Answer:"}
                ]
            }
        ]

        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': f'Bearer {self.openai_key}'
        }

        payload = {
            "model": model_info["model_name"],
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 5,
        }

        data = self._send_request(model_info["url"], headers, payload)
        choice = data["choices"][0]
        top = choice["logprobs"]["content"][0]["top_logprobs"]

        lps = {item["token"]: item["logprob"] for item in top}

        def get_lp(candidates):
            for c in candidates:
                if c in lps:
                    return lps[c]
            return None

        lp_y = get_lp(["Y", " Y"])
        lp_n = get_lp(["N", " N"])

        if lp_y is not None and lp_n is not None:
            py = math.exp(lp_y)
            pn = math.exp(lp_n)
            conf_y = py / (py + pn)
            conf_n = pn / (py + pn)
        else:
            sampled = choice["message"]["content"]
            sampled_lp = choice["logprobs"]["content"][0]["logprob"]
            if sampled.strip().startswith("Y"):
                conf_y, conf_n = math.exp(sampled_lp), 1 - math.exp(sampled_lp)
            else:
                conf_n, conf_y = math.exp(sampled_lp), 1 - math.exp(sampled_lp)

        confidence_y = conf_y
        confidence_n = conf_n

        return float(confidence_y)
