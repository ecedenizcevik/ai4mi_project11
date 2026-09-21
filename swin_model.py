from argparse import Namespace
from pathlib import Path
import sys

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
SWIN_ROOT = PROJECT_ROOT / "Swin-Unet"

# if str(SWIN_ROOT) not in sys.path:
#     sys.path.insert(0, str(SWIN_ROOT))

_original_sys_path = sys.path.copy()
sys.path.insert(0, str(SWIN_ROOT))

try:
    from config import get_config
    from networks.vision_transformer import SwinUnet
finally:
    sys.path[:] = _original_sys_path


def build_swin_unet(
    num_classes: int,
    checkpoint: str | Path,
    img_size: int = 224, # Image size must be compatible with the patch size and window size for divisibility
) -> nn.Module:
    config_path = (
        SWIN_ROOT
        / "configs"
        / "swin_tiny_patch4_window7_224_lite.yaml"
    )

    config_args = Namespace(
        cfg=str(config_path),
        opts=[
            "MODEL.PRETRAIN_CKPT",
            str(Path(checkpoint).resolve()),
            # "DATA.IMG_SIZE",
            # img_size,
        ],
        batch_size=None,
        zip=False,
        cache_mode=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="",
        tag=None,
        eval=False,
        throughput=False,
    )

    config = get_config(config_args)

    model = SwinUnet(
        config,
        img_size=img_size,
        num_classes=num_classes,
    )
    model.load_from(config)

    return model