import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import IMAGE_SIZE, KAGGLE_DIR, MODEL_DIR, SEED, get_device
from src.data.dataset import ChestXrayDataset, get_transforms
from src.data.split import (
    assert_no_patient_leakage,
    build_manifest,
    patient_level_split,
)
from src.eval.metrics import compute_metrics
from src.models import MODEL_REGISTRY


def get_dataloaders(batch_size: int):
    manifest = build_manifest(KAGGLE_DIR)
    train_df, val_df, _test_df = patient_level_split(manifest, seed=SEED)
    assert_no_patient_leakage(train_df, val_df, _test_df)

    train_ds = ChestXrayDataset(train_df, transform=get_transforms(IMAGE_SIZE, train=True))
    val_ds = ChestXrayDataset(val_df, transform=get_transforms(IMAGE_SIZE, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=6, persistent_workers=True
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())
    return compute_metrics(y_true, y_prob)


def train(model_name: str, epochs: int, batch_size: int, lr: float):
    device = get_device()
    torch.manual_seed(SEED)

    train_loader, val_loader = get_dataloaders(batch_size)

    model = MODEL_REGISTRY[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val={val_metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_cnn", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.model, args.epochs, args.batch_size, args.lr)
