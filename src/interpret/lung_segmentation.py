import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

_LEFT_LUNG_IDX = 4
_RIGHT_LUNG_IDX = 5

_crop = xrv.datasets.XRayCenterCrop()
_seg_model = None


def _get_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = xrv.baseline_models.chestx_det.PSPNet()
        _seg_model.eval()
    return _seg_model


@torch.no_grad()
def lung_mask(image_path: str, threshold: float = 0.5) -> np.ndarray:
    """Binary lung field mask from a pretrained chest-xray segmentation
    model (PSPNet, torchxrayvision), independent of our own classifier's
    preprocessing. Used as ground truth for the shortcut-feature metric:
    does the classifier's Grad-CAM attention actually fall on the lungs?
    """
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
    img = xrv.datasets.normalize(img, 255)  # -> [-1024, 1024]
    img = img.mean(2)[None, ...]  # single channel, (1, H, W)
    img = _crop(img)
    img_t = torch.from_numpy(img).float().unsqueeze(0)  # (1, 1, H, W)

    logits = _get_model()(img_t)
    lung_prob = torch.sigmoid(logits[0, _LEFT_LUNG_IDX] + logits[0, _RIGHT_LUNG_IDX])
    return (lung_prob >= threshold).numpy()
