from src.models.baseline_cnn import BaselineCNN
from src.models.transfer import DenseNet121Transfer, ResNet50Transfer

MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
    "resnet50": ResNet50Transfer,
    "densenet121": DenseNet121Transfer,
}
