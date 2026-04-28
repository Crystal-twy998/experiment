import os
from typing import Any, Dict, Optional, Union, List

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


class BagelImageEditor:
    """
    BAGEL wrapper used by the ZS-CIR pipeline.

    Important behavior:
    - edit_image_no_think(...): image-conditioned editing / I2I proxy generation.
    - text_to_image_no_think(...): pure text-to-image generation. This should be used
      for image_generation_mode == "target_only" when you want to test a real
      target-caption-only generated image branch.
    - generate_caption(...): text-only understanding/caption generation.
    """

    def __init__(
        self,
        model_path: str,
        max_mem_per_gpu: str = "80GiB",
        offload_folder: str = "/tmp/offload",
        device: Optional[Union[str, int, torch.device]] = None,
        use_multi_gpu: Optional[bool] = None,
    ):
        self.model_path = model_path
        self.max_mem_per_gpu = max_mem_per_gpu
        self.offload_folder = offload_folder
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.inferencer = None
        self.new_token_ids = None

        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                self.target_device = torch.device(f"cuda:{local_rank}")
            else:
                self.target_device = torch.device("cpu")
        elif isinstance(device, torch.device):
            self.target_device = device
        elif isinstance(device, int):
            self.target_device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.target_device = torch.device(device)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if use_multi_gpu is None:
            # In torchrun / multi-process mode, each process should keep the full
            # model on its own GPU. Multi-GPU dispatch is only useful for a single
            # process that wants model parallelism.
            use_multi_gpu = torch.cuda.device_count() > 1 and world_size == 1
        self.use_multi_gpu = bool(use_multi_gpu)
        self._initialize_model()

    def _build_device_map(self, model) -> Dict[str, Union[str, int]]:
        if (not torch.cuda.is_available()) or (not self.use_multi_gpu) or torch.cuda.device_count() <= 1:
            single_map = {"": str(self.target_device)}
            print(f"[BAGEL] Using single-device map: {single_map}")
            return single_map

        device_map = infer_auto_device_map(
            model,
            max_memory={i: self.max_mem_per_gpu for i in range(torch.cuda.device_count())},
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
        first_device = device_map.get(same_device_modules[0], 0)
        for module_name in same_device_modules:
            if module_name in device_map:
                device_map[module_name] = first_device
        print(f"[BAGEL] Using inferred multi-GPU device_map: {device_map}")
        return device_map

    def _initialize_model(self):
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        self.vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))
        if hasattr(self.vae_model, "to"):
            self.vae_model = self.vae_model.to(self.target_device)
        if hasattr(self.vae_model, "eval"):
            self.vae_model = self.vae_model.eval()

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

        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        os.makedirs(self.offload_folder, exist_ok=True)
        device_map = self._build_device_map(model)
        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(self.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=self.use_multi_gpu,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder=self.offload_folder,
        )
        self.model = self.model.eval()
        print(f"[BAGEL] Model loaded on target_device={self.target_device}, use_multi_gpu={self.use_multi_gpu}")

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=self.new_token_ids,
        )

    def edit_image_no_think(
        self,
        image_path: str,
        prompt: str,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = [0.0, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
    ) -> Dict[str, Any]:
        """Image-conditioned editing branch. Use for instruction_only and instruction_plus_target."""
        image = Image.open(image_path).convert("RGB")
        inference_hyper = {
            "cfg_text_scale": cfg_text_scale,
            "cfg_img_scale": cfg_img_scale,
            "cfg_interval": cfg_interval,
            "timestep_shift": timestep_shift,
            "num_timesteps": num_timesteps,
            "cfg_renorm_min": cfg_renorm_min,
            "cfg_renorm_type": cfg_renorm_type,
        }
        output_dict = self.inferencer(image=image, text=prompt, **inference_hyper)
        if output_dict.get("image", None) is None:
            raise RuntimeError("BAGEL edit_image_no_think returned no image.")
        return {"image": output_dict["image"]}

    def text_to_image_no_think(
        self,
        prompt: str,
        image_size: int = 1024,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: List[float] = [0.0, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
    ) -> Dict[str, Any]:
        """
        Pure text-to-image branch.

        This intentionally passes image=None to InterleaveInferencer so the generated
        image is conditioned only on the text prompt. Use this for target_only.
        """
        inference_hyper = {
            "cfg_text_scale": cfg_text_scale,
            "cfg_img_scale": cfg_img_scale,
            "cfg_interval": cfg_interval,
            "timestep_shift": timestep_shift,
            "num_timesteps": num_timesteps,
            "cfg_renorm_min": cfg_renorm_min,
            "cfg_renorm_type": cfg_renorm_type,
            "image_shapes": (int(image_size), int(image_size)),
        }
        output_dict = self.inferencer(image=None, text=prompt, **inference_hyper)
        if output_dict.get("image", None) is None:
            raise RuntimeError("BAGEL text_to_image_no_think returned no image.")
        return {"image": output_dict["image"]}

    def generate_caption(
        self,
        prompt: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
    ) -> str:
        inference_hyper = {
            "max_think_token_n": max_think_token_n,
            "do_sample": do_sample,
        }
        output_dict = self.inferencer(text=prompt, understanding_output=True, **inference_hyper)
        return output_dict["text"]
