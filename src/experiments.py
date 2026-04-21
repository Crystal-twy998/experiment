from argparse import Namespace
import torch
import os
import pickle
import data_utils
import utils
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset
import compute_results
import clip
from transformers import CLIPProcessor, CLIPModel
import termcolor
from classes import load_clip_model_and_preprocess
from bagel_inference import BagelImageEditor

import file_utils

import gc
import torch.distributed as dist


class Experiment:
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(args, "device", 0))))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1

        if self.distributed and dist.is_available() and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')

        print(f"[Init] rank={self.rank} local_rank={self.local_rank} world_size={self.world_size} device={self.device}")

    def is_main_process(self) -> bool:
        return self.rank == 0

    def _release_bagel_editor(self, bagel_editor):
        if bagel_editor is None:
            print("[BAGEL] No BAGEL editor to release.")
            return

        print("[BAGEL] Releasing BAGEL editor in experiments.py ...")
        try:
            if hasattr(bagel_editor, "model"):
                del bagel_editor.model
                print("[BAGEL] Deleted bagel_editor.model")
        except Exception as e:
            print(f"[BAGEL] Failed deleting model: {e}")
        try:
            if hasattr(bagel_editor, "inferencer"):
                del bagel_editor.inferencer
                print("[BAGEL] Deleted bagel_editor.inferencer")
        except Exception as e:
            print(f"[BAGEL] Failed deleting inferencer: {e}")
        try:
            del bagel_editor
        except Exception as e:
            print(f"[BAGEL] Failed deleting editor object: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[BAGEL] BAGEL editor released.")

    def run(self):
        clip_model, clip_processor = self.load_Clip_model()
        target_datasets, query_datasets, compute_results_function, compute_results_fuse2paths_function, pairings = self.load_dataset(
            clip_processor
        )
        self.evaluate(
            query_datasets,
            target_datasets,
            pairings,
            compute_results_function,
            compute_results_fuse2paths_function,
            clip_model,
            clip_processor
        )
        if self.distributed and dist.is_initialized():
            dist.barrier()
        if self.is_main_process():
            print("Finish.")
        return

    def data_preprocessing(self):
        return

    def load_Clip_model(self):
        clip_device = self.device
        print(f"Loading CLIP {self.clip}...")
        clip_model, clip_processor = load_clip_model_and_preprocess(
            dataset_path=self.dataset_path,
            clip_type=self.clip,
            device=clip_device
        )
        print("Done.")
        return clip_model, clip_processor

    def load_Bagel_model(self):
        print(f"[BAGEL] Start loading BAGEL from: {self.bagel_model_path}")
        print(f"[BAGEL] Current main device = {self.device}")
        bagel_editor = BagelImageEditor(
            self.bagel_model_path,
            device=self.device,
            use_multi_gpu=getattr(self, "bagel_use_multi_gpu", False),
        )
        print("[BAGEL] BAGEL loaded successfully.")
        return bagel_editor

    def _edited_images_cache_ready(self, metadata_path) -> bool:
        if metadata_path is None or not os.path.exists(metadata_path):
            return False
        try:
            with open(metadata_path, "rb") as f:
                edited_data = pickle.load(f)
            all_edit_img_paths = edited_data.get("all_edit_img_paths", [])
            if not isinstance(all_edit_img_paths, list) or len(all_edit_img_paths) == 0:
                return False
            return all(isinstance(path, str) and path != "" and os.path.exists(path) for path in all_edit_img_paths)
        except Exception as e:
            print(f"[BAGEL] Failed to validate edited image cache {metadata_path}: {e}")
            return False

    def _should_load_bagel(self, preload_dict) -> bool:
        stage_mode = getattr(self, "stage_mode", "initial_only")
        if stage_mode not in {"initial_only", "qwen_fusion"}:
            return True

        mods_ready = preload_dict.get("mods") is not None and os.path.exists(preload_dict["mods"])
        edits_ready = self._edited_images_cache_ready(preload_dict.get("edit_images"))

        print(f"[BAGEL] modified captions ready: {mods_ready}")
        print(f"[BAGEL] edited images ready: {edits_ready}")
        return not (mods_ready and edits_ready)

    def load_dataset(self, clip_processor: CLIPProcessor):
        target_datasets, query_datasets, pairings = [], [], []
        if 'fashioniq' in self.dataset.lower():
            dress_type = self.dataset.split('_')[-1]
            target_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'classic', clip_processor))
            query_datasets.append(FashionIQDataset(self.dataset_path, self.split, [dress_type], 'relative', clip_processor))
            pairings.append(dress_type)
            compute_results_function = compute_results.fiq
            compute_results_fuse2paths_function = compute_results.fiq_fuse2paths

        elif self.dataset.lower() == 'cirr':
            split = 'test1' if self.split == 'test' else self.split
            target_datasets.append(CIRRDataset(self.dataset_path, split, 'classic', clip_processor))
            query_datasets.append(CIRRDataset(self.dataset_path, split, 'relative', clip_processor))
            compute_results_function = compute_results.cirr
            pairings.append('default')
            compute_results_fuse2paths_function = compute_results.cirr_fuse2paths

        elif self.dataset.lower() == 'circo':
            target_datasets.append(CIRCODataset(self.dataset_path, self.split, 'classic', clip_processor))
            query_datasets.append(CIRCODataset(self.dataset_path, self.split, 'relative', clip_processor))
            compute_results_function = compute_results.circo
            pairings.append('default')
            compute_results_fuse2paths_function = compute_results.circo_fuse2paths

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        return target_datasets, query_datasets, compute_results_function, compute_results_fuse2paths_function, pairings

    def evaluate(self, query_datasets, target_datasets, pairings,
                 compute_results_function, compute_results_fuse2paths_function,
                 clip_model, clip_processor):
        preload_dict = {
            key: None for key in [
                "img_features", "new_captions", "captions", "img_paths",
                "mods", "suggestions", "pseudo_targets", "pseudo_targets_loop2",
                "candidates", "edit_images"
            ]
        }

        file_utils.init_folder(self.dataset_path, self.task)
        image_generation_mode = getattr(self, "image_generation_mode", "instruction_only")
        image_mode_tag = utils._sanitize_tag(image_generation_mode)
        stage1_edit_img_dir = os.path.join(self.edit_img_dir, image_mode_tag)
        os.makedirs(stage1_edit_img_dir, exist_ok=True)
        os.makedirs(f"{self.dataset_path}/preload/edited_images", exist_ok=True)
        os.makedirs(f"{self.dataset_path}/preload/img_features", exist_ok=True)
        print(f"[Stage-1 Image] image_generation_mode = {image_generation_mode}")
        print(f"[Stage-1 Image] edit_img_dir = {stage1_edit_img_dir}")

        if "mods" in self.preload:
            preload_dict["mods"] = f"{self.dataset_path}/task/{self.task}/modified_captions/{self.preload_modified_captions_file}"
        if "captions" in self.preload:
            preload_dict["captions"] = f"{self.dataset_path}/preload/image_captions/{self.preload_image_captions_file}"
        if "img_paths" in self.preload:
            preload_dict["img_paths"] = f"{self.dataset_path}/preload/image_paths/{self.preload_image_paths_file}"
        if "pseudo_targets" in self.preload:
            preload_dict["pseudo_targets"] = f"{self.dataset_path}/task/{self.task}/pseudo_targets/{self.clip}_{self.preload_pseudo_targets}"
        if "candidates" in self.preload:
            preload_dict["candidates"] = f"{self.dataset_path}/task/{self.task}/pseudo_targets/{self.clip}_{self.preload_candidates}"
        if "suggestions" in self.preload:
            preload_dict["suggestions"] = f"{self.dataset_path}/task/{self.task}/suggestions/{self.preload_suggestions}"
        if "img_features" in self.preload:
            preload_dict["img_features"] = f"{self.dataset_path}/preload/img_features/{self.clip}_{self.dataset}_{self.split}.pkl"
        if "edited_images" in self.preload:
            edit_meta_file = self.preload_edited_images_file
            meta_root, meta_ext = os.path.splitext(edit_meta_file)
            if meta_ext == "":
                meta_ext = ".pkl"
            edit_meta_file = f"{meta_root}_{image_mode_tag}{meta_ext}"
            preload_dict["edit_images"] = f"{self.dataset_path}/preload/edited_images/{edit_meta_file}"
        if "new_captions" in self.preload:
            preload_dict["new_captions"] = f"{self.dataset_path}/task/{self.task}/new_captions/{self.preload_new_captions}"

        stage_mode = getattr(self, "stage_mode", "initial_only")
        print(f"[Stage] stage_mode = {stage_mode}")

        bagel_editor = self.load_Bagel_model() if self._should_load_bagel(preload_dict) else None
        if bagel_editor is None:
            print("[BAGEL] Skipped loading BAGEL because modified captions and edited images are already available.")

        for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
            termcolor.cprint(f"\n------ Evaluating Retrieval Setup: {pairing}", color="yellow", attrs=["bold"])

            input_kwargs = {
                "args": self.args,
                "bagel_editor": bagel_editor,
                "dataset_name": self.dataset,
                "llm_prompt_args": self.llm_prompt,
                "retrieval": self.retrieval,
                "query_dataset": query_dataset,
                "target_dataset": target_dataset,
                "clip_model": clip_model,
                "processor": clip_processor,
                "device": self.device,
                "split": self.split,
                "preload_dict": preload_dict,
                "max_check_num": self.max_check_num,
                "Check_LLM_model_name": self.Check_LLM_model_name,
                "dataset_path": self.dataset_path,
                "edit_img_dir": stage1_edit_img_dir,
                "compute_results_function": compute_results_function,
                "VQA_LLM_model_name": self.VQA_LLM_model_name,
                "openai_key": self.openai_key,
                "task": self.task,
                "clip": self.clip,
                "preprocess": clip_processor
            }

            print(f"Extracting target image features using CLIP: {self.clip}.")
            index_features, index_names, index_ranks, aux_data = utils.extract_image_features(
                self.device,
                self.args,
                target_dataset,
                clip_model,
                preload=preload_dict["img_features"]
            )
            index_features = torch.nn.functional.normalize(index_features.float(), dim=-1)

            input_kwargs.update({
                "index_features": index_features,
                "index_names": index_names,
                "index_ranks": index_ranks,
                "LLM_model_name": self.LLM_model_name
            })

            print(f"Generating conditional query predictions (CLIP: {self.clip}).")
            out_dict = utils.generate_editimg_caption_iteration(**input_kwargs)
            input_kwargs.update(out_dict)

            if stage_mode == "initial_only":
                print("\nInitial-only stage finished.")
                print("Saved query text, query image, text-only result, image-only result, and loop-0 rankings.")
                if input_kwargs.get("txt_output_metrics") is not None:
                    print("Text-only metrics:", input_kwargs["txt_output_metrics"])
                if input_kwargs.get("img_output_metrics") is not None:
                    print("Image-only metrics:", input_kwargs["img_output_metrics"])
                print("Initial ranking file:", input_kwargs.get("initial_rank_path", ""))
                continue

            elif stage_mode == "qwen_fusion":
                self._release_bagel_editor(bagel_editor)
                bagel_editor = None

                if self.distributed and dist.is_initialized():
                    dist.barrier()

                if not self.is_main_process():
                    print(f"[Distributed] rank={self.rank} finished VQA shard for {pairing}.")
                    continue

                print("Split:", input_kwargs["split"])
                result_metrics, labels = compute_results_fuse2paths_function(**input_kwargs)

                print("\n")
                if result_metrics is not None:
                    termcolor.cprint(f"Metrics for {self.dataset.upper()} ({self.split})- {pairing}", attrs=["bold"])
                    for k, v in result_metrics.items():
                        print(f"{pairing}_{k} = {v:.2f}")
                else:
                    termcolor.cprint(
                        f"No explicit metrics available for {self.dataset.upper()} ({self.split}) - {pairing}.",
                        attrs=["bold"]
                    )
            else:
                raise ValueError(f"Unknown stage_mode: {stage_mode}")

        self._release_bagel_editor(bagel_editor)
