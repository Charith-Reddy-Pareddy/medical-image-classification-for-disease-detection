import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import DenseNet121Transfer, ResNet50Transfer


def target_layers_for(model) -> list:
    """The last conv feature map before pooling/classification -- the
    standard Grad-CAM target for each architecture.
    """
    if isinstance(model, BaselineCNN):
        return [model.features[-1][0]]
    if isinstance(model, ResNet50Transfer):
        return [model.backbone.layer4[-1]]
    if isinstance(model, DenseNet121Transfer):
        return [model.backbone.features]
    raise ValueError(f"no known Grad-CAM target layer for {type(model)}")


def compute_gradcam(model, image_tensor: torch.Tensor, category: int, plus_plus: bool = False):
    """Grad-CAM (or Grad-CAM++) heatmap for a single preprocessed image
    tensor (C, H, W), values in [0, 1] at the input's spatial resolution.
    category=1 targets the "pneumonia" logit, 0 targets "normal".
    """
    cam_cls = GradCAMPlusPlus if plus_plus else GradCAM
    cam = cam_cls(model=model, target_layers=target_layers_for(model))
    targets = [BinaryClassifierOutputTarget(category)]
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)
    return grayscale_cam[0]
