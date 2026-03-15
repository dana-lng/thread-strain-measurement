"""
Dataset-Klasse und Datenpipeline für das Kettfaden-Markierungs-Tracking.

Unterstützt mehrere Annotationsdateien und Bildordner gleichzeitig
(Multi-Folder-Training über Versuch_1 bis Versuch_11).

Ausgabe des Datasets pro Sample:
    image  : Tensor [1, H, W]  float32, normiert auf [0, 1]
    labels : Tensor [4]        float32
             labels[0] = y_top / H   (Dummy 0.5 wenn conf_top=0)
             labels[1] = y_bot / H   (Dummy 0.5 wenn conf_bot=0)
             labels[2] = conf_top    (1.0 = oberer Marker sichtbar)
             labels[3] = conf_bot    (1.0 = unterer Marker sichtbar)

Annotationsformat (JSON) – erstellt von 2_annotate.py:
    Beide Marker:   {"y_top": float, "y_bot": float, ..., "conf_top": 1.0, "conf_bot": 1.0}
    Nur unten:      {"y_top": 0.0,   "y_bot": float, ..., "conf_top": 0.0, "conf_bot": 1.0}
    Kein Marker:    {"y_top": 0.0,   "y_bot": 0.0,   ..., "conf_top": 0.0, "conf_bot": 0.0}

Kamerakalibrierung:
    Wenn calib_params (aus camera_calibration.json) übergeben wird, wendet
    MarkerDataset die Y-Scherungskorrektur automatisch an (über load_and_crop_calibrated).
    Alternativ: rectify_params für den alten rectification_params.json Workflow.

Multi-Folder-Training:
    Für Versuch_1 bis Versuch_11 mit je eigener Annotationsdatei:
    → build_combined_dataloaders() erzeugt einen gemeinsamen Train-/Val-Split
      über alle Ordner gleichzeitig (ConcatDataset + Subset).
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

from image_utils import load_and_crop_calibrated


class MarkerDataset(Dataset):
    """
    Dataset für Kettfaden-Markierungs-Tracking.

    Parameters
    ----------
    annotation_file : str | Path
        Pfad zur JSON-Annotationsdatei (erstellt mit 2_annotate.py).
    image_folder : str | Path
        Ordner mit den .tif Bilddateien.
    augment : bool
        Wenn True, wird Datenaugmentierung angewendet (nur für Training).
    target_size : tuple[int, int] | None
        Wenn angegeben (H, W), werden Bilder auf diese Größe skaliert.
        None empfohlen: AdaptiveAvgPool2d im Modell verarbeitet jede Größe nativ
        (128×1000 und 128×880 ohne Resize).
    calib_params : dict | None
        Ausgabe von load_camera_calibration(). Enthält shear_model, scale_v_model, …
        Wenn angegeben, wird Y-Scherungskorrektur angewendet.
    rectify_params : RectificationParams | None
        Alternativ (älterer Workflow): RectificationParams-Objekt.
        Wird ignoriert wenn calib_params angegeben.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_folder: str | Path,
        augment: bool = False,
        target_size: Optional[tuple[int, int]] = None,
        calib_params: dict | None = None,
        rectify_params=None,
    ):
        self.image_folder = Path(image_folder)
        self.augment = augment
        self.target_size = target_size
        self.calib_params = calib_params
        self.rectify_params = rectify_params

        with open(annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = []
        missing = 0
        for fname, ann in raw.items():
            fpath = self.image_folder / fname
            if not fpath.exists():
                missing += 1
                continue
            # conf_top / conf_bot – mit Fallback auf altes confidence-Feld
            if "conf_top" in ann and "conf_bot" in ann:
                conf_top = float(ann["conf_top"])
                conf_bot = float(ann["conf_bot"])
            else:
                confidence = float(ann.get("confidence", 1.0))
                if confidence >= 0.75:
                    conf_top, conf_bot = 1.0, 1.0
                elif confidence >= 0.25:
                    conf_top, conf_bot = 0.0, 1.0
                else:
                    conf_top, conf_bot = 0.0, 0.0
            self.samples.append({
                "path": str(fpath),
                "y_top": ann.get("y_top", 0.0),
                "y_bot": ann.get("y_bot", 0.0),
                "h_orig": ann["image_height"],
                "w_orig": ann["image_width"],
                "conf_top": conf_top,
                "conf_bot": conf_bot,
            })

        if missing > 0:
            print(f"[Dataset] Warnung: {missing} Bilddateien nicht gefunden in {self.image_folder}")
        print(f"[Dataset] {len(self.samples)} Samples aus {Path(annotation_file).name} "
              f"({self.image_folder.name})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Bild laden und Kalibrierung anwenden
        img = load_and_crop_calibrated(s["path"], self.calib_params, self.rectify_params)

        h_orig, w_orig = img.shape[:2]
        y_top = s["y_top"]
        y_bot = s["y_bot"]
        conf_top = s["conf_top"]
        conf_bot = s["conf_bot"]

        # Optionale Skalierung auf target_size
        if self.target_size is not None:
            h_new, w_new = self.target_size
            scale_y = h_new / h_orig
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            if conf_top > 0.5:
                y_top = y_top * scale_y
            if conf_bot > 0.5:
                y_bot = y_bot * scale_y
            h_orig = h_new

        # Perzentil-Kontrastnormierung (belichtungsinvariant: 1%–99% → 0–255)
        lo, hi = np.percentile(img, [1, 99])
        if hi > lo:
            img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255,
                          0, 255).astype(np.uint8)

        # Augmentierung (kein Shift wenn kein unterer Marker sichtbar)
        if self.augment:
            img, y_top, y_bot = self._augment(img, y_top, y_bot,
                                               allow_shift=(conf_bot > 0.5))

        # Normieren auf float32 [0,1]
        img_f = img.astype(np.float32) / 255.0

        # Labels normieren auf [0,1]: [y_top/H, y_bot/H, conf_top, conf_bot]
        h = img_f.shape[0]
        y_top_norm = float(np.clip(y_top / h, 0.0, 1.0)) if conf_top > 0.5 else 0.5
        y_bot_norm = float(np.clip(y_bot / h, 0.0, 1.0)) if conf_bot > 0.5 else 0.5
        label = np.array([y_top_norm, y_bot_norm, conf_top, conf_bot], dtype=np.float32)

        img_t = torch.from_numpy(img_f).unsqueeze(0)   # [1, H, W]
        label_t = torch.from_numpy(label)               # [4]

        return {"image": img_t, "labels": label_t, "path": s["path"]}

    # ------------------------------------------------------------------
    # Augmentierung
    # ------------------------------------------------------------------

    def _augment(
        self,
        img: np.ndarray,
        y_top: float,
        y_bot: float,
        allow_shift: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]

        # 1) Helligkeit & Kontrast
        alpha = random.uniform(0.75, 1.25)
        beta  = random.randint(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 2) Gauss'sches Rauschen
        if random.random() < 0.5:
            sigma = random.uniform(0, 8)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 3) Vertikaler Shift – Labels werden mitbewegt
        if allow_shift:
            max_shift = int(h * 0.05)
            if max_shift > 0:
                dy = random.randint(-max_shift, max_shift)
                M = np.float32([[1, 0, 0], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                y_top = float(np.clip(y_top + dy, 0, h - 1))
                y_bot = float(np.clip(y_bot + dy, 0, h - 1))

        # 4) Horizontaler Flip (y-Labels unverändert)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        return img, y_top, y_bot


# ---------------------------------------------------------------------------
# Single-Folder DataLoader (rückwärtskompatibel)
# ---------------------------------------------------------------------------

def build_dataloaders(
    annotation_file: str | Path,
    image_folder: str | Path,
    val_fraction: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: Optional[tuple[int, int]] = None,
    seed: int = 42,
    calib_params: dict | None = None,
    rectify_params=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Erstellt Train- und Validierungs-DataLoader aus einer einzelnen Annotationsdatei.

    Parameters
    ----------
    annotation_file : Pfad zur JSON-Annotationsdatei
    image_folder    : Ordner mit .tif Bildern
    val_fraction    : Anteil der Validierungsdaten (0.0–1.0)
    batch_size      : Batch-Größe
    num_workers     : Datenlader-Prozesse (0 = Hauptprozess, empfohlen auf Windows)
    target_size     : (H, W) oder None (None: keine Skalierung, Modell nutzt AdaptivePool)
    seed            : Zufalls-Seed für reproduzierbare Aufteilung
    calib_params    : dict aus camera_calibration.json (Scherungskorrektur)
    rectify_params  : RectificationParams (älterer Workflow, Fallback)

    Returns
    -------
    train_loader, val_loader
    """
    full_dataset = MarkerDataset(
        annotation_file=annotation_file,
        image_folder=image_folder,
        augment=False,
        target_size=target_size,
        calib_params=calib_params,
        rectify_params=rectify_params,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Trainings-Dataset neu mit augment=True (gleiche Pfade, andere Transforms)
    aug_dataset = MarkerDataset(
        annotation_file=annotation_file,
        image_folder=image_folder,
        augment=True,
        target_size=target_size,
        calib_params=calib_params,
        rectify_params=rectify_params,
    )
    train_set = Subset(aug_dataset, train_idx.indices)
    val_set   = Subset(full_dataset, val_idx.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False)

    print(f"[DataLoader] Train: {len(train_set)}  Val: {len(val_set)}  Batch: {batch_size}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Multi-Folder DataLoader (Versuch_1 bis Versuch_11)
# ---------------------------------------------------------------------------

def build_combined_dataloaders(
    annotation_files: list[str | Path],
    image_folders: list[str | Path],
    val_fraction: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: Optional[tuple[int, int]] = None,
    seed: int = 42,
    calib_params: dict | None = None,
    rectify_params=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Kombiniert mehrere Annotationsdateien + Bildordner zu einem gemeinsamen Dataset.

    Alle Versuchsordner werden als ConcatDataset zusammengefügt. Der Train/Val-Split
    erfolgt einmalig über den gesamten kombinierten Datensatz (keine pro-Ordner-Splits).

    Parameters
    ----------
    annotation_files : Liste von JSON-Annotationspfaden (eine pro Versuch)
    image_folders    : Liste von Bildordnern (korrespondierend zu annotation_files)
    …                : Alle anderen Parameter wie bei build_dataloaders()

    Returns
    -------
    train_loader, val_loader
    """
    assert len(annotation_files) == len(image_folders), (
        f"Anzahl Annotationsdateien ({len(annotation_files)}) ≠ "
        f"Bildordner ({len(image_folders)})"
    )

    if len(annotation_files) == 1:
        # Fallback auf Single-Folder für Einfachheit
        return build_dataloaders(
            annotation_files[0], image_folders[0],
            val_fraction=val_fraction, batch_size=batch_size,
            num_workers=num_workers, target_size=target_size,
            seed=seed, calib_params=calib_params, rectify_params=rectify_params,
        )

    # Zwei ConcatDatasets: ohne Augmentierung (für Val + Index-Split)
    #                       und mit Augmentierung (für Training)
    plain_datasets = []
    aug_datasets   = []
    for ann_file, img_folder in zip(annotation_files, image_folders):
        plain_datasets.append(MarkerDataset(
            ann_file, img_folder, augment=False,
            target_size=target_size, calib_params=calib_params,
            rectify_params=rectify_params,
        ))
        aug_datasets.append(MarkerDataset(
            ann_file, img_folder, augment=True,
            target_size=target_size, calib_params=calib_params,
            rectify_params=rectify_params,
        ))

    combined_plain = ConcatDataset(plain_datasets)
    combined_aug   = ConcatDataset(aug_datasets)

    n_total = len(combined_plain)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(combined_plain, [n_train, n_val], generator=generator)

    train_set = Subset(combined_aug,   train_idx.indices)
    val_set   = Subset(combined_plain, val_idx.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), drop_last=False)

    print(f"[DataLoader] {len(annotation_files)} Ordner, gesamt {n_total} Samples  "
          f"→  Train: {len(train_set)}  Val: {len(val_set)}  Batch: {batch_size}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 3:
        print("Verwendung: python dataset.py annotations.json /pfad/zu/bildern "
              "[camera_calibration.json]")
        sys.exit(1)

    ann_file   = sys.argv[1]
    img_folder = sys.argv[2]
    calib = None
    if len(sys.argv) >= 4:
        from image_utils import load_camera_calibration
        calib = load_camera_calibration(sys.argv[3])

    ds = MarkerDataset(ann_file, img_folder, augment=True, calib_params=calib)
    sample = ds[0]
    img_np   = sample["image"].squeeze().numpy()
    y_top_px = sample["labels"][0].item() * img_np.shape[0]
    y_bot_px = sample["labels"][1].item() * img_np.shape[0]
    conf_top = sample["labels"][2].item()
    conf_bot = sample["labels"][3].item()
    print(f"Shape: {img_np.shape}  "
          f"y_top={y_top_px:.1f}px  y_bot={y_bot_px:.1f}px  "
          f"conf_top={conf_top:.2f}  conf_bot={conf_bot:.2f}")

    plt.figure(figsize=(4, 10))
    plt.imshow(img_np, cmap="gray", vmin=0, vmax=1, aspect="auto")
    if conf_top > 0.5:
        plt.axhline(y_top_px, color="lime", lw=1.5, label=f"y_top={y_top_px:.1f}")
    if conf_bot > 0.5:
        plt.axhline(y_bot_px, color="orange", lw=1.5, label=f"y_bot={y_bot_px:.1f}")
    plt.legend()
    plt.title(os.path.basename(sample["path"]))
    plt.tight_layout()
    plt.show()
