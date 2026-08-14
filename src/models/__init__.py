from src.models.baseline_cnn import BaselineCNN

# extended in Day 4 with resnet50 / densenet121 transfer-learning builders
MODEL_REGISTRY = {
    "baseline_cnn": BaselineCNN,
}
