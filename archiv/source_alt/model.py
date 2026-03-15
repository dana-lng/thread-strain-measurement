"""
CNN-Modell für die Y-Koordinaten-Regression der Kettfaden-Markierungen.

Architektur: MarkerNet (Y-Achsen-Soft-Argmax)
==========================================
Eingabe : [B, 1, H, W]  Graustufenbild, normiert [0,1]
Ausgabe : [B, 4]         [y_top/H, y_bot/H, conf_top, conf_bot], alle in [0,1]

  y_top/H    : Normierte Y-Koordinate der oberen Markierung  (Soft-Argmax)
  y_bot/H    : Normierte Y-Koordinate der unteren Markierung (Soft-Argmax)
  conf_top   : Konfidenz oberer Marker sichtbar  (1=sicher, 0=nicht sichtbar)
  conf_bot   : Konfidenz unterer Marker sichtbar (1=sicher, 0=nicht sichtbar)

Kernidee: Y-Achsen-Soft-Argmax statt Global Average Pooling
------------------------------------------------------------
Global Average Pooling (GAP) verliert alle Rauminformation – das Modell sieht
danach nur globale Bildstatistiken ohne Ortskenntnis. Für Lokalisierungsaufgaben
(wo ist der Marker?) ist das fundamental ungeeignet.

Stattdessen:
  1. Encoder → [B, 8c, H/16, W/16]  (wie bisher)
  2. X-Achse (Breite) mitteln → Y-Profil [B, 8c, H/16]
     ⇒ erhält die vollständige Y-Ortsinformation
  3. 1D-Conv → zwei separate Heatmaps [B, 1, H/16] (eine pro Marker)
  4. Soft-Argmax:  y = Σ(i/H' · softmax(heatmap[i]))
     ⇒ differenzierbarer, sub-pixel-genauer Positions-Output
  5. Konfidenz-Kopf: AdaptiveAvgPool1d(1) + Linear → [B, 2]

Vorteile:
- Y-Ortsinformation bleibt vollständig erhalten
- Soft-Argmax ist per Definition positionstreu (keine Kanten-Bias)
- Zwei getrennte Heatmap-Köpfe → unabhängige Lokalisierung beider Marker
- Größenunabhängig: 128×1000 (2000 fps) und 128×880 (6000 fps) ohne Resize

Design-Entscheidungen:
- Batch Normalization für stabiles Training
- Residual-Verbindungen (enc3, enc4) für besseren Gradientenfluss
- Dropout nur im Konfidenz-Kopf (nicht in Heatmap-Köpfen)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baustein: Konvolutionsblock
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2D → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Residual-Block (einfach)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Einfacher Residual-Block: 2×(Conv→BN→ReLU) + Skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + x)


# ---------------------------------------------------------------------------
# Hauptmodell: MarkerNet mit Y-Achsen-Soft-Argmax
# ---------------------------------------------------------------------------

class MarkerNet(nn.Module):
    """
    CNN mit Y-Achsen-Soft-Argmax für Kettfaden-Markierungs-Tracking.

    Parameters
    ----------
    base_ch : int
        Basisanzahl an Filtern. Standard: 32.
    dropout : float
        Dropout-Rate im Konfidenz-Kopf (0 = kein Dropout).
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.3):
        super().__init__()

        c = base_ch

        # ------------------------------------------------------------------
        # 2D Encoder (identisch zur Vorgängerversion)
        # ------------------------------------------------------------------
        self.stem = ConvBnRelu(1, c, kernel=7, stride=1)           # [B, c,   H,    W   ]

        self.enc1 = nn.Sequential(
            ConvBnRelu(c, c * 2),
            nn.MaxPool2d(2),                                        # [B, 2c,  H/2,  W/2 ]
        )
        self.enc2 = nn.Sequential(
            ConvBnRelu(c * 2, c * 4),
            nn.MaxPool2d(2),                                        # [B, 4c,  H/4,  W/4 ]
        )
        self.enc3 = nn.Sequential(
            ConvBnRelu(c * 4, c * 8),
            ResBlock(c * 8),
            nn.MaxPool2d(2),                                        # [B, 8c,  H/8,  W/8 ]
        )
        self.enc4 = nn.Sequential(
            ConvBnRelu(c * 8, c * 8),
            ResBlock(c * 8),
            nn.MaxPool2d(2),                                        # [B, 8c,  H/16, W/16]
        )
        # Nach enc4: [B, 8c, H', W'] mit H'=H/16  (62 bei H=1000, 55 bei H=880)

        # ------------------------------------------------------------------
        # 1D Y-Profil Verarbeitung (nach X-Achsen-Mittelung)
        # ------------------------------------------------------------------
        self.profile_conv = nn.Sequential(
            nn.Conv1d(c * 8, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(c * 4),
            nn.ReLU(inplace=True),
        )
        # Nach profile_conv: [B, 4c, H']

        # ------------------------------------------------------------------
        # Zwei separate Heatmap-Köpfe (je einer pro Marker)
        # ------------------------------------------------------------------
        # Rohe Logits → Soft-Argmax → Y-Position in [0,1]
        self.heatmap_top = nn.Conv1d(c * 4, 1, kernel_size=3, padding=1)  # oberer Marker
        self.heatmap_bot = nn.Conv1d(c * 4, 1, kernel_size=3, padding=1)  # unterer Marker

        # ------------------------------------------------------------------
        # Konfidenz-Kopf (aus globalem Y-Profil-Feature)
        # ------------------------------------------------------------------
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # [B, 4c, 1]
            nn.Flatten(),              # [B, 4c]
            nn.Dropout(p=dropout),
            nn.Linear(c * 4, 2),       # [B, 2] → (conf_top, conf_bot)
            nn.Sigmoid(),
        )

    @staticmethod
    def _soft_argmax(logits: torch.Tensor) -> torch.Tensor:
        """
        Differenzierbarer Argmax über Y-Achse.

        Gibt normierte Position in [0,1] zurück, unabhängig von H'.

        - Flache Aktivierung  → 0.5 (neutraler Wert = Bildmitte)
        - Scharfer Peak bei k → ≈ k / (H' - 1)

        Parameters
        ----------
        logits : [B, H'] – rohe Aktivierungen der 1D-Heatmap

        Returns
        -------
        [B] – normierte Y-Position in [0,1]
        """
        weights = torch.softmax(logits, dim=1)                           # [B, H']
        pos     = torch.linspace(0.0, 1.0, logits.shape[1],
                                 device=logits.device, dtype=logits.dtype)
        return (weights * pos.unsqueeze(0)).sum(dim=1)                   # [B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D Encoder
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)                        # [B, 8c, H', W']

        # X-Achse (Bildbreite) mitteln → Y-Profil
        profile = x.mean(dim=3)                 # [B, 8c, H']

        # 1D Feature-Extraktion
        feat = self.profile_conv(profile)        # [B, 4c, H']

        # Konfidenz aus globalem Profil
        conf = self.conf_head(feat)              # [B, 2]

        # Heatmaps → Sub-Pixel-Positionen per Soft-Argmax
        hm_top = self.heatmap_top(feat).squeeze(1)   # [B, H']
        hm_bot = self.heatmap_bot(feat).squeeze(1)   # [B, H']
        y_top  = self._soft_argmax(hm_top)            # [B]
        y_bot  = self._soft_argmax(hm_bot)            # [B]

        # [y_top/H, y_bot/H, conf_top, conf_bot]
        return torch.stack([y_top, y_bot, conf[:, 0], conf[:, 1]], dim=1)  # [B, 4]


# ---------------------------------------------------------------------------
# Kleinere Variante für sehr wenige Trainingsdaten
# ---------------------------------------------------------------------------

class MarkerNetSmall(nn.Module):
    """
    Kompakteres Modell mit Y-Achsen-Soft-Argmax für <50 Trainingssample.
    ~60k Parameter statt ~500k bei MarkerNet(base_ch=32).

    Ausgabe: [B, 4] = [y_top/H, y_bot/H, conf_top, conf_bot], alle in [0,1]
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # 2D Encoder (kleiner)
        self.encoder = nn.Sequential(
            ConvBnRelu(1, 16, kernel=5),
            nn.MaxPool2d(2),             # /2
            ConvBnRelu(16, 32),
            nn.MaxPool2d(2),             # /4
            ConvBnRelu(32, 64),
            nn.MaxPool2d(2),             # /8
            ConvBnRelu(64, 64),
            nn.MaxPool2d(2),             # /16  ← ersetzt AdaptiveAvgPool2d(1)
        )
        # [B, 64, H/16, W/16]

        # 1D Y-Profil Verarbeitung
        self.profile_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Heatmap-Köpfe
        self.heatmap_top = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        self.heatmap_bot = nn.Conv1d(32, 1, kernel_size=3, padding=1)

        # Konfidenz
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    @staticmethod
    def _soft_argmax(logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits, dim=1)
        pos     = torch.linspace(0.0, 1.0, logits.shape[1],
                                 device=logits.device, dtype=logits.dtype)
        return (weights * pos.unsqueeze(0)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.encoder(x)                    # [B, 64, H/16, W/16]
        profile = x.mean(dim=3)                      # [B, 64, H/16]
        feat    = self.profile_conv(profile)          # [B, 32, H/16]
        conf    = self.conf_head(feat)                # [B, 2]
        hm_top  = self.heatmap_top(feat).squeeze(1)  # [B, H/16]
        hm_bot  = self.heatmap_bot(feat).squeeze(1)  # [B, H/16]
        y_top   = self._soft_argmax(hm_top)
        y_bot   = self._soft_argmax(hm_bot)
        return torch.stack([y_top, y_bot, conf[:, 0], conf[:, 1]], dim=1)  # [B, 4]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Erstellt ein Modell.

    Parameters
    ----------
    model_type : "standard" | "small"
    kwargs     : Werden an den Modell-Konstruktor weitergereicht.
    """
    if model_type == "small":
        model = MarkerNetSmall(**kwargs)
    else:
        model = MarkerNet(**kwargs)
    print(f"[Model] {model.__class__.__name__}  –  {count_parameters(model):,} trainierbare Parameter")
    return model


# ---------------------------------------------------------------------------
# Schnelltest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for mtype in ("standard", "small"):
        m = build_model(mtype)
        # Versuch 1-6: 128×1000 px (2000 fps)
        x1 = torch.zeros(4, 1, 1000, 128)
        y1 = m(x1)
        print(f"  [{mtype}] Input 1000×128: Output {tuple(y1.shape)}  "
              f"Werte: {y1[0].detach().tolist()}")
        # Versuch 8-11: 128×880 px (6000 fps)
        x2 = torch.zeros(4, 1, 880, 128)
        y2 = m(x2)
        print(f"  [{mtype}] Input  880×128: Output {tuple(y2.shape)}")
        print(f"  Output-Bedeutung: [y_top/H, y_bot/H, conf_top, conf_bot] – alle in [0,1]")
        print()
