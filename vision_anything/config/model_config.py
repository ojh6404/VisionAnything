import os
from typing import List
from dataclasses import dataclass

from vision_anything.config.config_base import InferenceConfigBase
from vision_anything import CHECKPOINT_ROOT, THIRD_PARTY_ROOT
from vision_anything.utils.download_utils import download_data


@dataclass
class GroundingDINOConfig(InferenceConfigBase):
    model_type: str = "swinb"
    model_configs = {
        "swint": os.path.join(CHECKPOINT_ROOT, "groundingdino/GroundingDINO_SwinT_OGC.py"),
        "swinb": os.path.join(CHECKPOINT_ROOT, "groundingdino/GroundingDINO_SwinB_cfg.py"),
    }
    model_checkpoints = {
        "swint": os.path.join(CHECKPOINT_ROOT, "groundingdino/groundingdino_swint_ogc.pth"),
        "swinb": os.path.join(CHECKPOINT_ROOT, "groundingdino/groundingdino_swinb_cogcoor.pth"),
    }
    model_url = {
        "swint": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "swinb": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
    }
    model_md5sum = {
        "swint": "075ebfa7242d913f38cb051fe1a128a2",
        "swinb": "611367df01ee834e3baa408f54d31f02",
    }
    model_config_url = {
        "swint": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "swinb": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    }
    model_config_md5sum = {
        "swint": "bdb07fc17b611d622633d133d2cf873a",
        "swinb": "30cd120e2e46feb3050a99c4e745d813",
    }

    def get_predictor(self):
        download_data(
            path=self.model_checkpoints[self.model_type],
            url=self.model_url[self.model_type],
            md5=self.model_md5sum[self.model_type],
        )
        download_data(
            path=self.model_configs[self.model_type],
            url=self.model_config_url[self.model_type],
            md5=self.model_config_md5sum[self.model_type],
        )

        try:
            from groundingdino.util.inference import Model as GroundingDINOModel
        except ImportError:
            from GroundingDINO.groundingdino.util.inference import (
                Model as GroundingDINOModel,
            )
        return GroundingDINOModel(
            model_config_path=self.model_configs[self.model_type],
            model_checkpoint_path=self.model_checkpoints[self.model_type],
            device=self.device,
        )

    @classmethod
    def from_args(cls, model_type: str = "swinb", device: str = "cuda:0"):
        return cls(model_name="GroundingDINO", model_type=model_type, device=device)


@dataclass
class YOLOConfig(InferenceConfigBase):
    model_type: str = "yolov8x-worldv2"
    model_checkpoints = {
        "yolov8x-worldv2": "yolov8x-worldv2.pt",
    }

    def get_predictor(self):
        from ultralytics import YOLOWorld

        return YOLOWorld(
            self.model_checkpoints[self.model_type],
        )

    @classmethod
    def from_args(cls, model_type: str = "yolov8x-worldv2", device: str = "cuda:0"):
        return cls(model_name="YOLO", model_type=model_type, device=device)


@dataclass
class SAMConfig(InferenceConfigBase):
    model_type: str = "vit_t"
    mode: str = "prompt"

    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "sam")
    model_checkpoints = {
        "vit_t": os.path.join(model_checkpoint_root, "mobile_sam.pth"),
        "vit_b": os.path.join(model_checkpoint_root, "sam_vit_b.pth"),
        "vit_l": os.path.join(model_checkpoint_root, "sam_vit_l.pth"),
        "vit_h": os.path.join(model_checkpoint_root, "sam_vit_h.pth"),
        "vit_b_hq": os.path.join(model_checkpoint_root, "sam_vit_b_hq.pth"),
        "vit_l_hq": os.path.join(model_checkpoint_root, "sam_vit_l_hq.pth"),
        "vit_h_hq": os.path.join(model_checkpoint_root, "sam_vit_h_hq.pth"),
    }
    model_url = {
        "vit_t": "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_b_hq": "https://drive.google.com/uc?id=11yExZLOve38kRZPfRx_MRxfIAKmfMY47",
        "vit_l_hq": "https://drive.google.com/uc?id=1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G",
        "vit_h_hq": "https://drive.google.com/uc?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8",
    }
    model_md5sum = {
        "vit_t": "f3c0d8cda613564d499310dab6c812cd",
        "vit_b": "01ec64d29a2fca3f0661936605ae66f8",
        "vit_l": "0b3195507c641ddb6910d2bb5adee89c",
        "vit_h": "4b8939a88964f0f4ff5f5b2642c598a6",
        "vit_b_hq": "c6b8953247bcfdc8bb8ef91e36a6cacc",
        "vit_l_hq": "08947267966e4264fb39523eccc33f86",
        "vit_h_hq": "3560f6b6a5a6edacd814a1325c39640a",
    }

    def get_predictor(self):
        assert self.model_type in SAMConfig.model_checkpoints
        assert self.mode in ["prompt", "automatic"]

        download_data(
            path=self.model_checkpoints[self.model_type],
            url=self.model_url[self.model_type],
            md5=self.model_md5sum[self.model_type],
        )

        if "hq" in self.model_type:  # segment anything hq
            from segment_anything_hq import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        elif self.model_type == "vit_t":  # mobile sam
            from mobile_sam import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        else:  # segment anything
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        model = sam_model_registry[self.model_type[:5]](checkpoint=self.model_checkpoints[self.model_type])
        model.to(device=self.device).eval()
        return SamPredictor(model) if self.mode == "prompt" else SamAutomaticMaskGenerator(model)

    @classmethod
    def from_args(cls, model_type: str = "vit_t", mode: str = "prompt", device: str = "cuda:0"):
        return cls(model_name="SAM", model_type=model_type, mode=mode, device=device)


@dataclass
class CutieConfig(InferenceConfigBase):
    model_checkpoint = os.path.join(CHECKPOINT_ROOT, "cutie/cutie-base-mega.pth")

    def get_predictor(self):
        download_data(
            path=self.model_checkpoint,
            url="https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
            md5="a6071de6136982e396851903ab4c083a",
        )
        import torch
        from omegaconf import open_dict
        import hydra
        from hydra import compose, initialize

        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with torch.inference_mode():
            initialize(
                version_base="1.3.2",
                config_path="../../third_party/Cutie/cutie/config",
                job_name="eval_config",
            )
            cfg = compose(config_name="eval_config")

            with open_dict(cfg):
                cfg["weights"] = self.model_checkpoint
            data_cfg = get_dataset_cfg(cfg)

            cutie = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(cfg.weights, map_location=self.device)
            cutie.load_weights(model_weights)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return InferenceCore(cutie, cfg=cfg)

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="Cutie", device=device)


@dataclass
class DEVAConfig(InferenceConfigBase):
    model_checkpoint = os.path.join(CHECKPOINT_ROOT, "deva/DEVA-propagation.pth")
    detection_every: int = 5
    enable_long_term: bool = True
    max_num_objects: int = 50
    max_missed_detection_count: int = 10
    amp: bool = True
    chunk_size: int = 4
    temporal_setting: str = "online"
    pluralize: bool = True

    def get_predictor(self):
        download_data(
            path=self.model_checkpoint,
            url="https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth",
            md5="a614cc9737a5b4c22ecbdc93e7842ecb",
        )

        from argparse import ArgumentParser
        import torch
        from deva.model.network import DEVA
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.eval_args import add_common_eval_args
        from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args

        # default parameters
        parser = ArgumentParser()
        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser)
        args = parser.parse_args([])

        # deva model
        args.model = self.model_checkpoint

        cfg = vars(args)
        cfg["enable_long_term"] = True

        # Load our checkpoint
        deva_model = DEVA(cfg).to(self.device).eval()
        if args.model is not None:
            model_weights = torch.load(args.model, map_location=self.device)
            deva_model.load_weights(model_weights)
        else:
            print("No model loaded.")

        # TODO clean it and make it configurable
        cfg["enable_long_term_count_usage"] = self.enable_long_term
        cfg["max_num_objects"] = self.max_num_objects
        cfg["amp"] = self.amp
        cfg["chunk_size"] = self.chunk_size
        cfg["detection_every"] = self.detection_every
        cfg["max_missed_detection_count"] = self.max_missed_detection_count
        cfg["temporal_setting"] = self.temporal_setting
        cfg["pluralize"] = self.pluralize

        deva = DEVAInferenceCore(deva_model, config=cfg)
        deva.next_voting_frame = cfg["num_voting_frames"] - 1
        deva.enabled_long_id()

        return deva

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="DEVA", device=device)


@dataclass
class VLPartConfig(InferenceConfigBase):
    model_type: str = "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed"
    model_root = os.path.join(THIRD_PARTY_ROOT, "VLPart")
    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "vlpart")
    model_checkpoints = {
        "r50_voc": os.path.join(model_checkpoint_root, "r50_voc.pth"),
        "r50_coco": os.path.join(model_checkpoint_root, "r50_coco.pth"),
        "r50_lvis": os.path.join(model_checkpoint_root, "r50_lvis.pth"),
        "r50_partimagenet": os.path.join(model_checkpoint_root, "r50_partimagenet.pth"),
        "r50_pascalpart": os.path.join(model_checkpoint_root, "r50_pascalpart.pth"),
        "r50_paco": os.path.join(model_checkpoint_root, "r50_paco.pth"),
        "r50_lvis_paco": os.path.join(model_checkpoint_root, "r50_lvis_paco.pth"),
        "r50_lvis_paco_pascalpart": os.path.join(model_checkpoint_root, "r50_lvis_paco_pascalpart.pth"),
        "r50_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet.pth"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet_in.pth"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_checkpoint_root, "r50_lvis_paco_pascalpart_partimagenet_inparsed.pth"
        ),
        "swinbase_cascade_voc": os.path.join(model_checkpoint_root, "swinbase_cascade_voc.pth"),
        "swinbase_cascade_coco": os.path.join(model_checkpoint_root, "swinbase_cascade_coco.pth"),
        "swinbase_cascade_lvis": os.path.join(model_checkpoint_root, "swinbase_cascade_lvis.pth"),
        "swinbase_cascade_partimagenet": os.path.join(
            model_checkpoint_root, "swinbase_cascade_partimagenet.pth"
        ),
        "swinbase_cascade_pascalpart": os.path.join(model_checkpoint_root, "swinbase_cascade_pascalpart.pth"),
        "swinbase_cascade_paco": os.path.join(model_checkpoint_root, "swinbase_cascade_paco.pth"),
        "swinbase_cascade_lvis_paco": os.path.join(model_checkpoint_root, "swinbase_cascade_lvis_paco.pth"),
        "swinbase_cascade_lvis_paco_pascalpart": os.path.join(
            model_checkpoint_root, "swinbase_cascade_lvis_paco_pascalpart.pth"
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_checkpoint_root,
            "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        ),
    }
    model_configs = {
        "r50_voc": os.path.join(model_root, "configs/voc/r50_voc.yaml"),
        "r50_coco": os.path.join(model_root, "configs/coco/r50_coco.yaml"),
        "r50_lvis": os.path.join(model_root, "configs/lvis/r50_lvis.yaml"),
        "r50_partimagenet": os.path.join(model_root, "configs/partimagenet/r50_partimagenet.yaml"),
        "r50_pascalpart": os.path.join(model_root, "configs/pascalpart/r50_pascalpart.yaml"),
        "r50_paco": os.path.join(model_root, "configs/joint/r50_lvis_paco.yaml"),
        "r50_lvis_paco": os.path.join(model_root, "configs/joint/r50_lvis_paco.yaml"),
        "r50_lvis_paco_pascalpart": os.path.join(model_root, "configs/joint/r50_lvis_paco_pascalpart.yaml"),
        "r50_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_root, "configs/joint/r50_lvis_paco_pascalpart_partimagenet.yaml"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_root, "configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_in.yaml"
        ),
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_root,
            "configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_inparsed.yaml",
        ),
        "swinbase_cascade_voc": os.path.join(model_root, "configs/voc/swinbase_cascade_voc.yaml"),
        "swinbase_cascade_coco": os.path.join(model_root, "configs/coco/swinbase_cascade_coco.yaml"),
        "swinbase_cascade_lvis": os.path.join(model_root, "configs/lvis/swinbase_cascade_lvis.yaml"),
        "swinbase_cascade_partimagenet": os.path.join(
            model_root, "configs/partimagenet/swinbase_cascade_partimagenet.yaml"
        ),
        "swinbase_cascade_pascalpart": os.path.join(
            model_root, "configs/pascalpart/swinbase_cascade_pascalpart.yaml"
        ),
        "swinbase_cascade_paco": os.path.join(model_root, "configs/joint/swinbase_cascade_lvis_paco.yaml"),
        "swinbase_cascade_lvis_paco": os.path.join(
            model_root, "configs/joint/swinbase_cascade_lvis_paco.yaml"
        ),
        "swinbase_cascade_lvis_paco_pascalpart": os.path.join(
            model_root, "configs/joint/swinbase_cascade_lvis_paco_pascalpart.yaml"
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": os.path.join(
            model_root,
            "configs/joint/swinbase_cascade_lvis_paco_pascalpart_partimagenet.yaml",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": os.path.join(
            model_root,
            "configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.yaml",
        ),
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": os.path.join(
            model_root,
            "configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml",
        ),
    }

    model_url = {
        "r50_voc": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_voc.pth",
        "r50_coco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_coco.pth",
        "r50_lvis": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis.pth",
        "r50_partimagenet": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_partimagenet.pth",
        "r50_pascalpart": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth",
        "r50_paco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_paco.pth",
        "r50_lvis_paco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco.pth",
        "r50_lvis_paco_pascalpart": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart.pth",
        "r50_lvis_paco_pascalpart_partimagenet": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet.pth",
        "r50_lvis_paco_pascalpart_partimagenet_in": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_in.pth",
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        "swinbase_cascade_voc": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_voc.pth",
        "swinbase_cascade_coco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_coco.pth",
        "swinbase_cascade_lvis": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis.pth",
        "swinbase_cascade_partimagenet": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_partimagenet.pth",
        "swinbase_cascade_pascalpart": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_pascalpart.pth",
        "swinbase_cascade_paco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_paco.pth",
        "swinbase_cascade_lvis_paco": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco.pth",
        "swinbase_cascade_lvis_paco_pascalpart": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart.pth",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": "https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth",
    }
    model_md5sum = {
        "r50_voc": "70ce256b6f05810b5334f41f2402eb09",
        "r50_coco": "1684ce2c837abfba9e884a30524df5bc",
        "r50_lvis": "42da3fccb40901041c0ee2fdc5683fd3",
        "r50_partimagenet": "4def166e3d1fd04a54d05992df32500c",
        "r50_pascalpart": "e5671ffdb282eb903173520a7e3c035a",
        "r50_paco": "f8d5032b8e8caef5423dc8ce69f6e04c",
        "r50_lvis_paco": "c2c80112b3d6ac6109053b6e5986b698",
        "r50_lvis_paco_pascalpart": "854769b7901728fc9d1ce84189bb2e7b",
        "r50_lvis_paco_pascalpart_partimagenet": "6783141824dd228f559063f52ec7aadd",
        "r50_lvis_paco_pascalpart_partimagenet_in": "0943cefccedad36598bc1b32ae52c0e2",
        "r50_lvis_paco_pascalpart_partimagenet_inparsed": "2e8999eb3a8fabe9b5191bdd0a094f75",
        "swinbase_cascade_voc": "5e0a17bdfd0517ea7de14de6bcbea121",
        "swinbase_cascade_coco": "f18c6e2aa099118afb07820ff970c5fe",
        "swinbase_cascade_lvis": "a909f64862104121817ba89784efd1c9",
        "swinbase_cascade_partimagenet": "ab757d04bd1c48ba8e4995f55d0fee54",
        "swinbase_cascade_pascalpart": "0119fa74970c32824c5806ea86658126",
        "swinbase_cascade_paco": "157dd2d35400a1aa16c5dde8d17355b0",
        "swinbase_cascade_lvis_paco": "7969a7dfd9a5be8a7e88c7c174d54db2",
        "swinbase_cascade_lvis_paco_pascalpart": "5f5d1dd844eee2a3ccacc74f176b2bc1",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet": "b0a1960530b4150ccb9c8d84174fa985",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": "013f21eb55b8ec0e5f3d5841e2186ab5",
        "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": "c2b4865ad03db1cf00d5da921a72c5de",
    }

    def get_predictor(
        self,
        vocabulary: str = "custom",
        custom_vocabulary: List[str] = [],
        confidence_threshold: float = 0.7,
    ):
        download_data(
            path=self.model_checkpoints[self.model_type],
            url=self.model_url[self.model_type],
            md5=self.model_md5sum[self.model_type],
        )
        from detectron2.config import get_cfg

        import sys
        import argparse

        sys.path.insert(0, self.model_root)
        from vlpart.config import add_vlpart_config
        from demo.predictor import VisualizationDemo

        def setup_cfg(args):
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_vlpart_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            # Set score_threshold for builtin models
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

            # replace the filename in the list to the full path
            for idx, filename in enumerate(cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP):
                if filename:
                    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP[idx] = os.path.join(
                        self.model_root, filename
                    )
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH = os.path.join(
                self.model_root, cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH
            )
            for idx, filename in enumerate(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP):
                if filename:
                    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP[idx] = os.path.join(self.model_root, filename)
            cfg.freeze()
            return cfg

        custom_vocabulary = ",".join(custom_vocabulary) if vocabulary == "custom" else ""  # type: ignore
        args = {
            "config_file": self.model_configs[self.model_type],
            "vocabulary": vocabulary,
            "custom_vocabulary": custom_vocabulary,
            "confidence_threshold": confidence_threshold,
            "opts": [
                "MODEL.WEIGHTS",
                self.model_checkpoints[self.model_type],
                "VIS.BOX",
                "False",
            ],
        }

        args = argparse.Namespace(**args)  # type: ignore
        cfg = setup_cfg(args)

        demo = VisualizationDemo(cfg, args)
        return demo

    @classmethod
    def from_args(cls, model_type:str = "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed", device: str = "cuda:0"):
        return cls(model_name="VLPart", model_type=model_type, device=device)


@dataclass
class MaskDINOConfig(InferenceConfigBase):
    model_type: str = "panoptic_swinl"
    confidence_threshold: float = 0.7
    model_root = os.path.join(THIRD_PARTY_ROOT, "MaskDINO")
    model_checkpoint_root = os.path.join(CHECKPOINT_ROOT, "MaskDINO")
    model_checkpoints = {
        "instance_r50_hid1024": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
        ),
        "instance_r50": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
        ),
        "instance_swinl_no_mask_enhanced": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth",
        ),
        "instance_swinl": os.path.join(
            model_checkpoint_root,
            "instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
        ),
        "semantic_r50_ade20k": os.path.join(
            model_checkpoint_root,
            "semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
        ),
        "semantic_r50_cityscapes": os.path.join(
            model_checkpoint_root,
            "semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        ),
        "panoptic_r50": os.path.join(
            model_checkpoint_root,
            "panoptic/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth",
        ),
        "panoptic_swinl": os.path.join(
            model_checkpoint_root,
            "panoptic/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth",
        ),
    }
    model_configs = {
        "instance_r50_hid1024": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        ),
        "instance_r50": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        ),
        "instance_swinl_no_mask_enhanced": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
        "instance_swinl": os.path.join(
            model_root,
            "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
        "semantic_r50_ade20k": os.path.join(
            model_root,
            "configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml",
        ),
        "semantic_r50_cityscapes": os.path.join(
            model_root,
            "configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml",
        ),
        "panoptic_r50": os.path.join(
            model_root,
            "configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
        ),
        "panoptic_swinl": os.path.join(
            model_root,
            "configs/coco/panoptic-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        ),
    }
    model_url = {
        "instance_r50_hid1024": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
        "instance_r50": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
        "instance_swinl_no_mask_enhanced": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth",
        "instance_swinl": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
        "semantic_r50_ade20k": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
        "semantic_r50_cityscapes": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        "panoptic_r50": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth",
        "panoptic_swinl": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth",
    }
    model_md5sum = {
        "instance_r50_hid1024": "0542c05ce5eef21d7073e421cec3bf16",
        "instance_r50": "443a397e48f2901f115d514695c27097",
        "instance_swinl_no_mask_enhanced": "af112f2b59607b95bbbc7b285efb561d",
        "instance_swinl": "05307c4356d20a258fdd96a8dade61b1",
        "semantic_r50_ade20k": "2b89bb1950174eaff27c8acfa6203efe",
        "semantic_r50_cityscapes": "8debe34409d4a9ca1add8ba72b311397",
        "panoptic_r50": "1cbe19c16f4aacd64431e01c2b076ce8",
        "panoptic_swinl": "7f5c513abf9d42aba6e04e428bc8baac",
    }

    def get_predictor(self):
        download_data(
            path=self.model_checkpoints[self.model_type],
            url=self.model_url[self.model_type],
            md5=self.model_md5sum[self.model_type],
        )

        import argparse
        import sys

        sys.path.insert(0, self.model_root)
        sys.path.insert(0, os.path.join(self.model_root, "demo"))
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from maskdino import add_maskdino_config
        from predictor import VisualizationDemo

        def setup_cfg(args):
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskdino_config(cfg)
            cfg.merge_from_file(args.config_file)
            cfg.merge_from_list(args.opts)
            cfg.freeze()
            return cfg

        args = {
            "config_file": self.model_configs[self.model_type],
            "confidence_threshold": self.confidence_threshold,
            "opts": ["MODEL.WEIGHTS", self.model_checkpoints[self.model_type]],
        }

        args = argparse.Namespace(**args)  # type: ignore
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        return demo

    @classmethod
    def from_args(
        cls, model_type: str = "panoptic_swinl", confidence_threshold: float = 0.7, device: str = "cuda:0"
    ):
        return cls(
            model_name="MaskDINO",
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            device=device,
        )


@dataclass
class MASAConfig(InferenceConfigBase):
    model_type: str = "masa_r50"
    model_root = os.path.join(THIRD_PARTY_ROOT, "masa")
    # TODO there are another model
    model_configs = {
        "masa_r50": os.path.join(model_root, "configs/masa-one/masa_r50_plug_and_play.py"),
        "masa_gdino": os.path.join(model_root, "configs/masa-gdino/masa_gdino_swinb_inference.py"),
    }
    model_checkpoints = {
        "masa_r50": os.path.join(CHECKPOINT_ROOT, "masa/masa_r50.pth"),
        "masa_gdino": os.path.join(CHECKPOINT_ROOT, "masa/gdino_masa.pth"),
    }
    model_url = {
        "masa_r50": "https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth",
        "masa_gdino": "https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth",
    }
    model_md5sum = {
        "masa_r50": "1b67de86a3cb8cab35c4797bcb8ea95f",
        "masa_gdino": "4b703764ce82b6a133b47fb5aef2e4e2",
    }

    def get_predictor(self, memo_tracklet_frames: int = 10000):
        download_data(
            path=self.model_checkpoints[self.model_type],
            url=self.model_url[self.model_type],
            md5=self.model_md5sum[self.model_type],
        )
        import os
        import sys

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        sys.path.insert(0, self.model_root)

        from mmengine.config import Config

        self.model_config = Config.fromfile(self.model_configs[self.model_type])
        self.model_config["model"]["tracker"]["memo_tracklet_frames"] = memo_tracklet_frames

        from masa.apis import init_masa

        masa_model = init_masa(self.model_config, self.model_checkpoints[self.model_type], device=self.device)
        return masa_model

    @classmethod
    def from_args(cls, model_type: str = "masa_r50", device: str = "cuda:0"):
        return cls(model_name="MASA", model_type=model_type, device=device)
