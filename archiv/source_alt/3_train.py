"""
Trainingsschleife für das Kettfaden-Markierungs-Tracking Modell.

Unterstützt Multi-Folder-Training über alle Versuchsreihen gleichzeitig
(Versuch_1 bis Versuch_11, insg. ~47.000 Bilder + Annotationen).

Verwendung (einfach, ein Ordner):
    python 3_train.py
        --annotations Versuch_1/annotations.json
        --images      Versuch_1
        --calib       camera_calibration.json

Verwendung (multi-folder, alle Versuche):
    python 3_train.py
        --annotations Versuch_1/annotations.json Versuch_2/annotations.json ...
        --images      Versuch_1 Versuch_2 ...
        --calib       camera_calibration.json
        --epochs      150

Wichtige Argumente:
    --annotations   JSON-Annotationsdatei(en) (aus 2_annotate.py)
    --images        Bildordner (korrespondierend zu --annotations, einer pro Ordner)
    --calib         Pfad zu camera_calibration.json (Scherungskorrektur + Maßstab)
    --rectify       Alternativ: rectification_params.json (älterer Workflow)
    --epochs        Trainings-Epochen (Standard: 100)
    --batch         Batch-Größe (Standard: 16)
    --lr            Lernrate (Standard: 1e-3)
    --model         "standard" (~500k Parameter) oder "small" (~40k, für wenige Daten)
    --output        Ausgabeordner für Modelle (Standard: checkpoints/)
    --target_size   H W – Bilder auf feste Größe skalieren (Standard: None = nativ)

Ausgabe (checkpoints/):
    best_model.pt          : Modell mit niedrigstem Val-Loss (für Inferenz)
    checkpoint_ep0100.pt   : Checkpoint alle 10 Epochen (für Fortsetzung)
    training_history.json  : Loss + MAE pro Epoche
    learning_curves.png    : Lernkurven-Plot

Modell-Details:
    MarkerNet: [B,1,H,W] → [y_top/H, y_bot/H, confidence]
    AdaptiveAvgPool2d(1): Modell verarbeitet 128×1000 und 128×880 nativ (kein Resize nötig).
    target_size=None (Standard): Bilder behalten ihre native Größe, Batch-Bildung funktioniert
    da alle Bilder einer Versuchsreihe gleich groß sind.
    Für gemischte Größen in einem Batch: --target_size 1000 128 (oder kleinste gemeinsame Größe).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders, build_combined_dataloaders
from model import build_model


# ---------------------------------------------------------------------------
# Verlustfunktion
# ---------------------------------------------------------------------------

class MarkerLoss(nn.Module):
    """
    Kombinierter Verlust für 4-Ausgaben-Modell [y_top, y_bot, conf_top, conf_bot]:

    Modell-Ausgabe pred:   [B, 4] = [y_top_pred, y_bot_pred, conf_top_pred, conf_bot_pred]
    Label          target: [B, 4] = [y_top_norm, y_bot_norm, conf_top,      conf_bot     ]

    3 Trainingszustände:
      conf_top=0, conf_bot=0  →  kein Marker:   trainiert beide conf gegen 0
      conf_top=0, conf_bot=1  →  nur unten:     trainiert conf_bot gegen 1, y_bot L1
      conf_top=1, conf_bot=1  →  beide Marker:  trainiert alles inkl. Ordnungsbedingung
    """

    def __init__(self, order_weight: float = 1.0, conf_weight: float = 1.0):
        super().__init__()
        self.order_weight = order_weight
        self.conf_weight  = conf_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        conf_top_label = target[:, 2]
        conf_bot_label = target[:, 3]

        # Konfidenz-Loss (BCE für beide Marker separat, dann gemittelt)
        conf_loss = (
            F.binary_cross_entropy(pred[:, 2], conf_top_label, reduction="mean") +
            F.binary_cross_entropy(pred[:, 3], conf_bot_label, reduction="mean")
        ) / 2.0

        # y_bot-Loss (L1, nur wenn unterer Marker sichtbar)
        mask_bot = (conf_bot_label > 0.5).float()
        n_bot = mask_bot.sum().clamp(min=1.0)
        bot_loss = (torch.abs(pred[:, 1] - target[:, 1]) * mask_bot).sum() / n_bot

        # y_top-Loss (L1, nur wenn oberer Marker sichtbar)
        mask_top = (conf_top_label > 0.5).float()
        n_top = mask_top.sum().clamp(min=1.0)
        top_loss = (torch.abs(pred[:, 0] - target[:, 0]) * mask_top).sum() / n_top

        # Ordnungsbedingung: y_top < y_bot
        mask_both = (conf_top_label * conf_bot_label > 0.5).float()
        n_both = mask_both.sum().clamp(min=1.0)
        order_loss = (torch.relu(pred[:, 0] - pred[:, 1]) * mask_both).sum() / n_both

        return (bot_loss + top_loss
                + self.conf_weight * conf_loss
                + self.order_weight * order_loss)


# ---------------------------------------------------------------------------
# Trainings- und Validierungsschritt
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0

    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
        h  = imgs.size(2)
        total_loss += loss.item() * bs
        mask_bot = (labels[:, 3] > 0.5).float()
        mask_top = (labels[:, 2] > 0.5).float()
        n_bot = mask_bot.sum().clamp(min=1.0)
        n_top = mask_top.sum().clamp(min=1.0)
        mae_bot = (torch.abs(pred[:, 1] - labels[:, 1]) * h * mask_bot).sum().item() / n_bot.item()
        mae_top = (torch.abs(pred[:, 0] - labels[:, 0]) * h * mask_top).sum().item() / n_top.item()
        total_mae += ((mae_bot + mae_top) / 2.0) * bs
        n += bs

    return {"loss": total_loss / n, "mae_px": total_mae / n}


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n = 0

    for batch in loader:
        imgs   = batch["image"].to(device)
        labels = batch["labels"].to(device)
        pred   = model(imgs)
        loss   = criterion(pred, labels)

        bs = imgs.size(0)
        h  = imgs.size(2)
        total_loss += loss.item() * bs
        mask_bot = (labels[:, 3] > 0.5).float()
        mask_top = (labels[:, 2] > 0.5).float()
        n_bot = mask_bot.sum().clamp(min=1.0)
        n_top = mask_top.sum().clamp(min=1.0)
        mae_bot = (torch.abs(pred[:, 1] - labels[:, 1]) * h * mask_bot).sum().item() / n_bot.item()
        mae_top = (torch.abs(pred[:, 0] - labels[:, 0]) * h * mask_top).sum().item() / n_top.item()
        total_mae += ((mae_bot + mae_top) / 2.0) * bs
        n += bs

    return {"loss": total_loss / n, "mae_px": total_mae / n}


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    ) if args.device == "auto" else torch.device(args.device)
    print(f"[Train] Gerät: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Kamerakalibrierung laden
    calib_params   = None
    rectify_params = None
    if args.calib:
        from image_utils import load_camera_calibration
        calib_params = load_camera_calibration(args.calib)
        print(f"[Train] Kamerakalibrierung geladen: {args.calib}")
    elif args.rectify:
        from rectify import load_rectification_params
        rectify_params = load_rectification_params(args.rectify)
        print(f"[Train] Rektifizierung aktiv: {rectify_params}")

    # Bildgröße
    if args.target_size is not None:
        target_size = tuple(args.target_size)
        print(f"[Train] Zielgröße: {target_size[1]}×{target_size[0]} px")
    else:
        target_size = None
        print(f"[Train] Zielgröße: nativ (AdaptivePool verarbeitet jede Größe)")

    # Prüfen ob multi-folder
    annotation_files = args.annotations
    image_folders    = args.images
    if len(annotation_files) != len(image_folders):
        print(f"Fehler: --annotations ({len(annotation_files)}) und "
              f"--images ({len(image_folders)}) müssen gleich viele Einträge haben.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[Train] {len(annotation_files)} Ordner werden kombiniert.")

    # DataLoader bauen
    if len(annotation_files) == 1:
        train_loader, val_loader = build_dataloaders(
            annotation_file=annotation_files[0],
            image_folder=image_folders[0],
            val_fraction=args.val_fraction,
            batch_size=args.batch,
            num_workers=0,
            target_size=target_size,
            seed=args.seed,
            calib_params=calib_params,
            rectify_params=rectify_params,
        )
    else:
        train_loader, val_loader = build_combined_dataloaders(
            annotation_files=annotation_files,
            image_folders=image_folders,
            val_fraction=args.val_fraction,
            batch_size=args.batch,
            num_workers=0,
            target_size=target_size,
            seed=args.seed,
            calib_params=calib_params,
            rectify_params=rectify_params,
        )

    # Modell, Optimierer, Loss
    model_kwargs = {"dropout": args.dropout}
    if args.model == "standard":
        model_kwargs["base_ch"] = args.base_ch
    model     = build_model(args.model, **model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = MarkerLoss(order_weight=1.0, conf_weight=1.0)

    # Training
    best_val_loss = float("inf")
    history = []

    print(f"\n{'Epoche':>6}  {'Train Loss':>10}  {'Train MAE':>10}  "
          f"{'Val Loss':>10}  {'Val MAE':>10}  {'LR':>10}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss":   train_m["loss"],
            "train_mae_px": train_m["mae_px"],
            "val_loss":     val_m["loss"],
            "val_mae_px":   val_m["mae_px"],
            "lr": lr_now,
        })

        print(f"{epoch:>6}  {train_m['loss']:>10.5f}  {train_m['mae_px']:>9.2f}px  "
              f"{val_m['loss']:>10.5f}  {val_m['mae_px']:>9.2f}px  {lr_now:>10.2e}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val_loss,
                "val_mae_px":  val_m["mae_px"],
                "args":        vars(args),
                "target_size": target_size,
            }, out_dir / "best_model.pt")
            print(f"         → Bestes Modell gespeichert (Val Loss={best_val_loss:.5f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args":            vars(args),
                "target_size":     target_size,
            }, out_dir / f"checkpoint_ep{epoch:04d}.pt")

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining abgeschlossen. Val Loss={best_val_loss:.5f}")
    print(f"Bestes Modell: {out_dir / 'best_model.pt'}")

    _plot_history(history, out_dir)


def _plot_history(history: list, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, [h["train_loss"] for h in history], label="Train")
        ax1.plot(epochs, [h["val_loss"]   for h in history], label="Val")
        ax1.set_xlabel("Epoche"); ax1.set_ylabel("Loss"); ax1.legend()
        ax1.set_title("Verlustfunktion (MarkerLoss)")

        ax2.plot(epochs, [h["train_mae_px"] for h in history], label="Train")
        ax2.plot(epochs, [h["val_mae_px"]   for h in history], label="Val")
        ax2.set_xlabel("Epoche"); ax2.set_ylabel("MAE (px)"); ax2.legend()
        ax2.set_title("Mittlerer absoluter Fehler")

        plt.tight_layout()
        fig.savefig(out_dir / "learning_curves.png", dpi=150)
        plt.close(fig)
        print(f"Lernkurven: {out_dir / 'learning_curves.png'}")
    except Exception as e:
        print(f"Lernkurven konnten nicht gespeichert werden: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training: Kettfaden-Markierungs-Tracking (Multi-Folder)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", nargs="+", required=True,
                        help="JSON-Annotationsdatei(en) (eine pro Versuchsordner)")
    parser.add_argument("--images", nargs="+", required=True,
                        help="Bildordner (korrespondierend zu --annotations)")
    parser.add_argument("--calib", type=str, default=None,
                        help="Pfad zu camera_calibration.json (Scherungskorrektur)")
    parser.add_argument("--rectify", type=str, default=None,
                        help="Alternativ: rectification_params.json")
    parser.add_argument("--model", default="standard", choices=["standard", "small"])
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Basiskanäle im Encoder (Standard: 32=3.4M Param, 16=220k Param).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--target_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Zielgröße (H W). Standard: None (nativ, empfohlen).")
    args = parser.parse_args()
    main(args)
