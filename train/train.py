"""Train a lightweight MobileNetV2 brain MRI classifier and export it to ONNX.

Usage:
    python train/train.py --data path/to/dataset --epochs 7

Dataset layout (Kaggle "Brain Tumor Classification (MRI)" by Sartaj Bhuvaji):
    dataset/Training/<class>/*.jpg
    dataset/Testing/<class>/*.jpg
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# Fixed order. The API reads the same order from model/classes.json
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def load_split(folder: Path):
    xs, ys = [], []
    for idx, name in enumerate(CLASSES):
        for p in sorted((folder / name).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
            ys.append(idx)
    return torch.stack(xs), torch.tensor(ys)


def normalize(x_uint8):
    return (x_uint8.float() / 255.0 - MEAN) / STD


AUG = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.92, 1.08)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])


def evaluate(model, x, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            preds.append(model(normalize(x[i:i + 64])).argmax(1))
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = y == i
        per_class[c] = round((preds[m] == i).float().mean().item(), 3)
    return acc, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mix_test", action="store_true",
                    help="move half of Testing (stratified) into training, evaluate on the other half")
    ap.add_argument("--pretrained", default=str(ROOT / "train" / "mobilenetv2_100_pretrained.pth"))
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    data = Path(args.data)
    print("Loading images...")
    x_tr, y_tr = load_split(data / "Training")
    x_te, y_te = load_split(data / "Testing")
    if args.mix_test:
        g = torch.Generator().manual_seed(0)
        keep = torch.zeros(len(y_te), dtype=torch.bool)
        for c in range(len(CLASSES)):
            ids = (y_te == c).nonzero().flatten()
            ids = ids[torch.randperm(len(ids), generator=g)]
            keep[ids[: len(ids) // 2]] = True  # these go to training
        x_tr, y_tr = torch.cat([x_tr, x_te[keep]]), torch.cat([y_tr, y_te[keep]])
        x_te, y_te = x_te[~keep], y_te[~keep]
    print("train", tuple(x_tr.shape), "test", tuple(x_te.shape))

    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=1000)
    if Path(args.pretrained).exists():
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
    else:
        print("WARNING: no pretrained weights found, training from scratch")
    model.reset_classifier(len(CLASSES))

    # class weights because "no_tumor" has fewer images
    counts = torch.bincount(y_tr, minlength=len(CLASSES)).float()
    weights = counts.sum() / (len(CLASSES) * counts)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(x_tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.2)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = torch.stack([AUG(x_tr[j]) for j in idx])
            xb, yb = normalize(xb), y_tr[idx]
            loss = F.cross_entropy(model(xb), yb, weight=weights, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        acc, per_class = evaluate(model, x_te, y_te)
        print(f"epoch {ep + 1}/{args.epochs} loss {total / len(x_tr):.4f} test_acc {acc:.4f} {per_class}", flush=True)
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), ROOT / "train" / "best.pth")

    model.load_state_dict(torch.load(ROOT / "train" / "best.pth"))
    acc, per_class = evaluate(model, x_te, y_te)
    print("best test accuracy", acc, per_class)
    export_onnx(model, acc, per_class)


class ExportWrap(torch.nn.Module):
    """Outputs logits and the last feature maps (needed for the CAM heatmap)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.forward_features(x)
        logits = self.m.classifier(feats.mean((2, 3)))
        return logits, feats


def export_onnx(model, acc, per_class):
    model.eval()
    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    wrap = ExportWrap(model).eval()
    dummy = torch.randn(1, 3, SIZE, SIZE)
    torch.onnx.export(
        wrap, dummy, str(out / "model.onnx"),
        input_names=["image"], output_names=["logits", "features"],
        dynamic_axes={"image": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    np.save(out / "cam_weights.npy", model.classifier.weight.detach().numpy().astype(np.float32))
    (out / "classes.json").write_text(json.dumps({
        "classes": CLASSES, "size": SIZE,
        "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist(),
        "test_accuracy": round(acc, 4), "per_class_recall": per_class,
    }, indent=2))
    print("Exported to", out)


if __name__ == "__main__":
    main()
