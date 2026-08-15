import torch

from src.eval.metrics import compute_metrics


@torch.no_grad()
def evaluate_across_datasets(model, loaders: dict, device) -> dict:
    """Runs a single trained model, unmodified, across several DataLoaders
    (one per dataset/institution) and reports predictions + metrics for
    each. This is the domain-shift comparison: no retraining between sites.
    """
    model.eval()
    results = {}
    for name, loader in loaders.items():
        y_true, y_prob = [], []
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.tolist())
        results[name] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": compute_metrics(y_true, y_prob),
        }
    return results
