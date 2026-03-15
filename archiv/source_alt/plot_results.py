#!/usr/bin/env python3
"""
plot_results.py – Publikationsreife Plots aus results.json

Erzeugt zwei Graphen:
  1. Y-Position beider Marker über Zeit [ms]
  2. Dehnung [%] über Zeit [ms]

Jeder Graph wird gespeichert als:
  <prefix>_y_position.pdf  +  <prefix>_y_position.py
  <prefix>_strain.pdf      +  <prefix>_strain.py

Die .py-Dateien sind eigenständige Python-Skripte mit eingebetteten Daten –
öffnen, nach Belieben anpassen und nochmals ausführen.

Verwendung:
    python plot_results.py
        --results  ../Versuch_1/results.json
        [--fps     2000]
        [--prefix  ../Versuch_1/Versuch_1]
        [--smooth  100]
        [--title_y "Marker-Y-Position – Versuch 1"]
        [--title_s "Dehnung – Versuch 1"]
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Optik-Konstanten ───────────────────────────────────────────────────────────
LABEL_FS   = 13
TITLE_FS   = 14
TICK_FS    = 11
LEGEND_FS  = 10
LINE_LW    = 0.8
SMOOTH_LW  = 2.5
AXIS_LW    = 1.3
RAW_ALPHA  = 0.25
GRID_A     = 0.30

FIG_SIZE_Y = (12, 5.0)
FIG_SIZE_S = (12, 4.5)

COLOR_TOP    = "#008800"
COLOR_BOT    = "#FF6600"
COLOR_STRAIN = "#CC2200"
COLOR_ZERO   = "#888888"


# ── Rolling-Mean (zentriert, für statische Plots) ─────────────────────────────

def centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    Zentrierter Rolling-Mean mit NaN-Behandlung.

    Für statische Plots besser als kausaler Mean: glättet symmetrisch
    (kein Phasenversatz gegenüber den Rohdaten).

    Parameters
    ----------
    values : 1D float-Array (NaN = kein Messwert)
    window : Fenstergröße in Frames (wird auf halbiert: ±window//2)

    Returns
    -------
    1D float-Array, NaN wo kein Wert vorhanden
    """
    half = window // 2
    out  = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        if np.isnan(values[i]):
            continue
        i0 = max(0, i - half)
        i1 = min(len(values), i + half + 1)
        neighbors = values[i0:i1]
        valid = neighbors[~np.isnan(neighbors)]
        if len(valid) > 0:
            out[i] = float(np.mean(valid))
    return out


# ── Hilfsfunktion: Achsen stylen ──────────────────────────────────────────────

def _style_ax(ax):
    ax.tick_params(labelsize=TICK_FS, width=AXIS_LW, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_LW)
    ax.grid(True, alpha=GRID_A)


# ── Python-Export: Array als String ───────────────────────────────────────────

def _arr_to_str(arr: np.ndarray, vals_per_line: int = 8) -> str:
    """Gibt numpy-Array als lesbaren Multi-Zeilen-String zurück."""
    items = []
    for v in arr.tolist():
        if isinstance(v, float) and math.isnan(v):
            items.append("np.nan")
        else:
            items.append(repr(round(float(v), 4)))
    lines = []
    for i in range(0, len(items), vals_per_line):
        lines.append("    " + ", ".join(items[i : i + vals_per_line]) + ",")
    return "np.array([\n" + "\n".join(lines) + "\n])"


# ── Y-Position-Plot ───────────────────────────────────────────────────────────

def plot_y_position(time_ms, y_top, y_bot, sm_ytop, sm_ybot,
                    title: str, prefix: str):
    """Erstellt Y-Position-Plot, speichert als PDF + Python."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_Y)

    mask_b  = ~np.isnan(y_bot);  mask_t  = ~np.isnan(y_top)
    smask_b = ~np.isnan(sm_ybot); smask_t = ~np.isnan(sm_ytop)

    # Rohdaten (dünn, transparent)
    if mask_b.any():
        ax.plot(time_ms[mask_b], y_bot[mask_b],
                color=COLOR_BOT, lw=LINE_LW, alpha=RAW_ALPHA,
                label="y_bot (roh)")
    if mask_t.any():
        ax.plot(time_ms[mask_t], y_top[mask_t],
                color=COLOR_TOP, lw=LINE_LW, alpha=RAW_ALPHA,
                label="y_top (roh)")

    # Geglättet (dick, voll opak)
    if smask_b.any():
        ax.plot(time_ms[smask_b], sm_ybot[smask_b],
                color=COLOR_BOT, lw=SMOOTH_LW,
                label="y_bot (geglättet)")
    if smask_t.any():
        ax.plot(time_ms[smask_t], sm_ytop[smask_t],
                color=COLOR_TOP, lw=SMOOTH_LW,
                label="y_top (geglättet)")

    ax.invert_yaxis()   # oben = kleinere px-Koordinate (Bildkonvention)
    ax.set_title(title, fontsize=TITLE_FS, pad=6)
    ax.set_xlabel("Zeit (ms)", fontsize=LABEL_FS)
    ax.set_ylabel("Y-Koordinate (px)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS)
    _style_ax(ax)
    fig.tight_layout()

    pdf_path = f"{prefix}_y_position.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Gespeichert: {pdf_path}")
    plt.close(fig)

    _export_py_y_position(prefix, time_ms, y_top, y_bot,
                          sm_ytop, sm_ybot, title)


def _export_py_y_position(prefix, time_ms, y_top, y_bot,
                           sm_ytop, sm_ybot, title):
    """Erzeugt eigenständiges Python-Skript für den Y-Position-Plot."""
    py_path = f"{prefix}_y_position.py"

    code = textwrap.dedent("""\
    #!/usr/bin/env python3
    # Automatisch erstellt von plot_results.py
    # Bearbeiten und nochmals ausfuehren um den Plot anzupassen.
    # Kann mit jedem Python ausgefuehrt werden – laedt Pakete automatisch aus dem Venv.

    import sys
    import pathlib

    # Projektumgebung: Pakete aus dem Venv laden (numpy, matplotlib)
    # Skript liegt in: Kettenwirken/2026-02-26/Versuch_X/
    # Venv liegt in:   Kettenwirken/nn_marker_tracking/venv/
    _sp = (pathlib.Path(__file__).parents[2]
           / "nn_marker_tracking" / "venv" / "Lib" / "site-packages")
    if _sp.exists() and str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── Daten ──────────────────────────────────────────────────────────────────
    """)

    code += f"time_ms  = {_arr_to_str(time_ms)}\n"
    code += f"y_top    = {_arr_to_str(y_top)}   # px; NaN = kein Marker\n"
    code += f"y_bot    = {_arr_to_str(y_bot)}\n"
    code += f"sm_ytop  = {_arr_to_str(sm_ytop)}  # geglaettet\n"
    code += f"sm_ybot  = {_arr_to_str(sm_ybot)}\n"

    code += textwrap.dedent(f"""
    # ── Optik ──────────────────────────────────────────────────────────────────
    COLOR_TOP  = {repr(COLOR_TOP)}
    COLOR_BOT  = {repr(COLOR_BOT)}
    LINE_LW    = {LINE_LW}
    SMOOTH_LW  = {SMOOTH_LW}
    RAW_ALPHA  = {RAW_ALPHA}
    AXIS_LW    = {AXIS_LW}
    LABEL_FS   = {LABEL_FS}
    TITLE_FS   = {TITLE_FS}
    TICK_FS    = {TICK_FS}
    LEGEND_FS  = {LEGEND_FS}

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    mask_b  = ~np.isnan(y_bot);  mask_t  = ~np.isnan(y_top)
    smask_b = ~np.isnan(sm_ybot); smask_t = ~np.isnan(sm_ytop)

    if mask_b.any():
        ax.plot(time_ms[mask_b], y_bot[mask_b],
                color=COLOR_BOT, lw=LINE_LW, alpha=RAW_ALPHA, label="y_bot (roh)")
    if mask_t.any():
        ax.plot(time_ms[mask_t], y_top[mask_t],
                color=COLOR_TOP, lw=LINE_LW, alpha=RAW_ALPHA, label="y_top (roh)")
    if smask_b.any():
        ax.plot(time_ms[smask_b], sm_ybot[smask_b],
                color=COLOR_BOT, lw=SMOOTH_LW, label="y_bot (geglaettet)")
    if smask_t.any():
        ax.plot(time_ms[smask_t], sm_ytop[smask_t],
                color=COLOR_TOP, lw=SMOOTH_LW, label="y_top (geglaettet)")

    ax.invert_yaxis()
    ax.set_title({repr(title)}, fontsize=TITLE_FS, pad=6)
    ax.set_xlabel("Zeit (ms)", fontsize=LABEL_FS)
    ax.set_ylabel("Y-Koordinate (px)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(labelsize=TICK_FS, width=AXIS_LW, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_LW)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()

    out = pathlib.Path(__file__).with_suffix(".pdf")
    try:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Gespeichert:", out)
    except PermissionError:
        print("FEHLER: Kann", out.name, "nicht speichern.")
        print("  Tipp: Schliessen Sie die PDF im Viewer, dann nochmals ausfuehren.")
    plt.show()
    """)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[Plot] Gespeichert: {py_path}")


# ── Dehnungs-Plot ─────────────────────────────────────────────────────────────

def plot_strain(time_ms, strain, sm_strain, title: str, prefix: str):
    """Erstellt Dehnungs-Plot, speichert als PDF + Python."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_S)

    mask   = ~np.isnan(strain)
    smask  = ~np.isnan(sm_strain)

    if mask.any():
        ax.plot(time_ms[mask], strain[mask],
                color=COLOR_STRAIN, lw=LINE_LW, alpha=RAW_ALPHA,
                label="Dehnung (roh)")
    if smask.any():
        ax.plot(time_ms[smask], sm_strain[smask],
                color=COLOR_STRAIN, lw=SMOOTH_LW,
                label="Dehnung (geglättet)")

    ax.axhline(0, color=COLOR_ZERO, lw=1.2, linestyle="--")
    ax.set_title(title, fontsize=TITLE_FS, pad=6)
    ax.set_xlabel("Zeit (ms)", fontsize=LABEL_FS)
    ax.set_ylabel("Dehnung (%)", fontsize=LABEL_FS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=LEGEND_FS)
    _style_ax(ax)
    fig.tight_layout()

    pdf_path = f"{prefix}_strain.pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"[Plot] Gespeichert: {pdf_path}")
    plt.close(fig)

    _export_py_strain(prefix, time_ms, strain, sm_strain, title)


def _export_py_strain(prefix, time_ms, strain, sm_strain, title):
    """Erzeugt eigenständiges Python-Skript für den Dehnungs-Plot."""
    py_path = f"{prefix}_strain.py"

    code = textwrap.dedent("""\
    #!/usr/bin/env python3
    # Automatisch erstellt von plot_results.py
    # Bearbeiten und nochmals ausfuehren um den Plot anzupassen.
    # Kann mit jedem Python ausgefuehrt werden – laedt Pakete automatisch aus dem Venv.

    import sys
    import pathlib

    # Projektumgebung: Pakete aus dem Venv laden (numpy, matplotlib)
    # Skript liegt in: Kettenwirken/2026-02-26/Versuch_X/
    # Venv liegt in:   Kettenwirken/nn_marker_tracking/venv/
    _sp = (pathlib.Path(__file__).parents[2]
           / "nn_marker_tracking" / "venv" / "Lib" / "site-packages")
    if _sp.exists() and str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── Daten ──────────────────────────────────────────────────────────────────
    """)

    code += f"time_ms   = {_arr_to_str(time_ms)}\n"
    code += f"strain    = {_arr_to_str(strain)}    # %; NaN = kein Marker\n"
    code += f"sm_strain = {_arr_to_str(sm_strain)}  # geglaettet\n"

    code += textwrap.dedent(f"""
    # ── Optik ──────────────────────────────────────────────────────────────────
    COLOR_STRAIN = {repr(COLOR_STRAIN)}
    COLOR_ZERO   = {repr(COLOR_ZERO)}
    LINE_LW      = {LINE_LW}
    SMOOTH_LW    = {SMOOTH_LW}
    RAW_ALPHA    = {RAW_ALPHA}
    AXIS_LW      = {AXIS_LW}
    LABEL_FS     = {LABEL_FS}
    TITLE_FS     = {TITLE_FS}
    TICK_FS      = {TICK_FS}
    LEGEND_FS    = {LEGEND_FS}

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))

    mask  = ~np.isnan(strain)
    smask = ~np.isnan(sm_strain)

    if mask.any():
        ax.plot(time_ms[mask], strain[mask],
                color=COLOR_STRAIN, lw=LINE_LW, alpha=RAW_ALPHA,
                label="Dehnung (roh)")
    if smask.any():
        ax.plot(time_ms[smask], sm_strain[smask],
                color=COLOR_STRAIN, lw=SMOOTH_LW,
                label="Dehnung (geglaettet)")

    ax.axhline(0, color=COLOR_ZERO, lw=1.2, linestyle="--")
    ax.set_title({repr(title)}, fontsize=TITLE_FS, pad=6)
    ax.set_xlabel("Zeit (ms)", fontsize=LABEL_FS)
    ax.set_ylabel("Dehnung (%)", fontsize=LABEL_FS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(labelsize=TICK_FS, width=AXIS_LW, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_LW)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()

    out = pathlib.Path(__file__).with_suffix(".pdf")
    try:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("Gespeichert:", out)
    except PermissionError:
        print("FEHLER: Kann", out.name, "nicht speichern.")
        print("  Tipp: Schliessen Sie die PDF im Viewer, dann nochmals ausfuehren.")
    plt.show()
    """)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[Plot] Gespeichert: {py_path}")


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Publikationsreife Plots aus results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",  required=True,
                   help="Pfad zu results.json (Ausgabe von 4_infer.py)")
    p.add_argument("--fps",      type=float, default=None,
                   help="Kamera-Framerate für Zeitachse (fps). "
                        "Standard: aus time_s-Feld in JSON (empfohlen).")
    p.add_argument("--prefix",   default=None,
                   help="Ausgabe-Prefix für Dateinamen. "
                        "Standard: <Versuch-Ordner>/<Versuch-Ordner>")
    p.add_argument("--smooth",   type=int, default=100,
                   help="Zentrierter Rolling-Mean Fenstergröße (Frames)")
    p.add_argument("--title_y",  default=None,
                   help="Titel für Y-Position-Plot")
    p.add_argument("--title_s",  default=None,
                   help="Titel für Dehnungs-Plot")
    args = p.parse_args()

    # ── Daten laden ──────────────────────────────────────────────────────────
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Fehler: {results_path} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Plot] {len(data)} Frames aus {results_path.name}")

    n = len(data)

    # ── Zeit-Achse ────────────────────────────────────────────────────────────
    # Bevorzuge time_s aus JSON; Fallback: frame / fps
    if "time_s" in data[0]:
        time_ms = np.array([r["time_s"] * 1000.0 for r in data])
    elif args.fps is not None:
        time_ms = np.array([r["frame"] / args.fps * 1000.0 for r in data])
    else:
        print("Warnung: Kein time_s in JSON und kein --fps angegeben. "
              "Verwende Frame-Nummer als X-Achse.", file=sys.stderr)
        time_ms = np.array([float(r["frame"]) for r in data])

    # ── Messwert-Arrays (NaN = kein Marker) ──────────────────────────────────
    def _get(r, key):
        v = r.get(key)
        return float(v) if v is not None else np.nan

    y_top  = np.array([_get(r, "y_top_px")     for r in data])
    y_bot  = np.array([_get(r, "y_bot_px")      for r in data])
    strain = np.array([_get(r, "strain_percent") for r in data])

    # ── Geglättete Serien (zentriert) ────────────────────────────────────────
    print(f"[Plot] Zentrierter Rolling-Mean: Fenster = {args.smooth} Frames")
    sm_ytop  = centered_rolling_mean(y_top,  args.smooth)
    sm_ybot  = centered_rolling_mean(y_bot,  args.smooth)
    sm_strain = centered_rolling_mean(strain, args.smooth)

    # ── Ausgabe-Prefix bestimmen ──────────────────────────────────────────────
    if args.prefix is not None:
        prefix = args.prefix
    else:
        parent = results_path.parent
        prefix = str(parent / parent.name)

    # ── Plot-Titel ────────────────────────────────────────────────────────────
    versuch = results_path.parent.name
    title_y = args.title_y or f"Marker-Y-Position – {versuch}"
    title_s = args.title_s or f"Dehnung – {versuch}"

    # ── Plots erzeugen ────────────────────────────────────────────────────────
    plot_y_position(time_ms, y_top, y_bot, sm_ytop, sm_ybot, title_y, prefix)
    plot_strain(time_ms, strain, sm_strain, title_s, prefix)

    # ── Statistik ─────────────────────────────────────────────────────────────
    n_valid = int(np.sum(~np.isnan(strain)))
    print(f"[Plot] Valide Frames: {n_valid}/{n} ({100*n_valid/max(n,1):.1f}%)")
    s_valid = strain[~np.isnan(strain)]
    if len(s_valid) > 0:
        print(f"[Plot] Dehnung: min={s_valid.min():+.2f}%  "
              f"max={s_valid.max():+.2f}%  ∅={s_valid.mean():+.2f}%")


if __name__ == "__main__":
    main()
