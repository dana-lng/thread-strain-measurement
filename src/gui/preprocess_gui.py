from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = ROOT / 'src' / 'preprocessesing'
SOBEL_DIR = ROOT / 'src' / 'sobelAlg'

import sys
for folder in (PREPROCESS_DIR, SOBEL_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rectify_experiment import ProcessingCancelled, process_experiment  # noqa: E402
from detect_markers import INPUT_ROOT, find_two_markers, load_image, save_sobel_diagram  # noqa: E402


class PreprocessGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('Versuch Preprocessing und Sobel')
        self.root.geometry('1380x920')
        self.root.minsize(1180, 840)

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.active_mode: str | None = None
        self.cancel_event = threading.Event()

        self.experiment_var = tk.StringVar(value='11')
        self.status_var = tk.StringVar(value='Bereit. Bitte Versuchsnummer eingeben.')
        self.progress_var = tk.StringVar(value='Noch nicht gestartet.')

        self._build_ui()
        self.root.after(150, self._poll_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill='both', expand=True)
        top = ttk.Frame(outer)
        top.pack(fill='x')
        left = ttk.Frame(top)
        left.pack(side='left', fill='y')

        ttk.Label(left, text='Versuch Pipeline', font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        ttk.Label(left, text='Nur die Versuchsnummer eingeben. Beispiel: 1, 10 oder 11.').pack(anchor='w', pady=(6, 18))

        input_row = ttk.Frame(left)
        input_row.pack(fill='x')
        ttk.Label(input_row, text='Versuchsnummer:').pack(side='left')
        self.entry = ttk.Entry(input_row, textvariable=self.experiment_var, width=12)
        self.entry.pack(side='left', padx=(10, 12))
        self.entry.focus_set()
        self.entry.bind('<Return>', lambda _event: self.start_preprocess())

        button_row = ttk.Frame(left)
        button_row.pack(fill='x', pady=(14, 0))
        self.preprocess_button = ttk.Button(button_row, text='Preprocess starten', command=self.start_preprocess)
        self.preprocess_button.pack(side='left')
        self.sobel_button = ttk.Button(button_row, text='Sobel starten', command=self.start_sobel)
        self.sobel_button.pack(side='left', padx=(10, 0))
        self.cancel_button = ttk.Button(button_row, text='Abbrechen', command=self.cancel_current, state='disabled')
        self.cancel_button.pack(side='left', padx=(10, 0))

        self.progress = ttk.Progressbar(left, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(18, 10))
        ttk.Label(left, textvariable=self.status_var, wraplength=420).pack(anchor='w', pady=(4, 8))
        ttk.Label(left, textvariable=self.progress_var, wraplength=420).pack(anchor='w')

        self.log_box = tk.Text(left, height=10, width=58, state='disabled')
        self.log_box.pack(fill='x', pady=(18, 0))

        preview = ttk.LabelFrame(top, text='Live Ansicht', padding=8)
        preview.pack(side='left', fill='both', expand=True, padx=(18, 0))
        self.figure = Figure(figsize=(10.5, 6.3), dpi=100)
        self.axes = self.figure.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1.35]})
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        titles = ['Preprocessed Bild', 'Bild mit Markierungen', 'Diagramm']
        for axis, title in zip(self.axes, titles):
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, 'Noch keine Daten', ha='center', va='center', transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def log(self, text: str) -> None:
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _set_busy(self, busy: bool) -> None:
        state = 'disabled' if busy else 'normal'
        self.preprocess_button.configure(state=state)
        self.sobel_button.configure(state=state)
        self.entry.configure(state=state)
        self.cancel_button.configure(state='normal' if busy else 'disabled')

    def _normalize_experiment(self) -> str | None:
        experiment_number = self.experiment_var.get().strip()
        if not experiment_number:
            messagebox.showerror('Fehlende Eingabe', 'Bitte eine Versuchsnummer eingeben.')
            return None
        return experiment_number if experiment_number.lower().startswith('versuch_') else f'Versuch_{experiment_number}'

    def cancel_current(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        self.cancel_event.set()
        self.status_var.set('Abbruch angefordert ...')
        self.progress_var.set('Der laufende Vorgang stoppt nach dem aktuellen Bild.')
        self.log('Abbruch angefordert.')
        self.cancel_button.configure(state='disabled')

    def start_preprocess(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        experiment = self._normalize_experiment()
        if experiment is None:
            return

        self.cancel_event.clear()
        self.active_mode = 'preprocess'
        self.progress['value'] = 0
        self.status_var.set(f'Starte Preprocessing fuer {experiment} ...')
        self.progress_var.set('Vorbereitung laeuft.')
        self._set_busy(True)
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')
        self.worker = threading.Thread(target=self._run_preprocess, args=(experiment,), daemon=True)
        self.worker.start()

    def start_sobel(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        experiment = self._normalize_experiment()
        if experiment is None:
            return

        experiment_dir = INPUT_ROOT / experiment
        if not experiment_dir.exists():
            messagebox.showerror('Fehlender Versuch', f'Ordner nicht gefunden: {experiment_dir}')
            return

        self.cancel_event.clear()
        self.active_mode = 'sobel'
        self.progress['value'] = 0
        self.status_var.set(f'Starte Sobel-Auswertung fuer {experiment} ...')
        self.progress_var.set('Diagramme werden live aktualisiert.')
        self._set_busy(True)
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')
        self._draw_placeholder()
        self.worker = threading.Thread(target=self._run_sobel, args=(experiment,), daemon=True)
        self.worker.start()

    def _run_preprocess(self, experiment: str) -> None:
        def progress_callback(index: int, total: int, output_path: Path) -> None:
            self.queue.put(('preprocess_progress', (experiment, index, total, output_path)))

        try:
            total = process_experiment(
                experiment,
                progress_callback=progress_callback,
                should_cancel=self.cancel_event.is_set,
            )
        except ProcessingCancelled as exc:
            self.queue.put(('cancelled', (experiment, str(exc))))
            return
        except Exception as exc:
            self.queue.put(('error', (experiment, exc)))
            return

        self.queue.put(('done', (experiment, total, 'preprocess')))

    def _run_sobel(self, experiment: str) -> None:
        files = sorted((INPUT_ROOT / experiment).glob('*.png'))
        if not files:
            self.queue.put(('error', (experiment, FileNotFoundError(f'Keine PNG-Bilder gefunden in {INPUT_ROOT / experiment}'))))
            return

        total = len(files)
        previous_markers: np.ndarray | None = None
        try:
            for index, input_path in enumerate(files, start=1):
                if self.cancel_event.is_set():
                    raise ProcessingCancelled(f'Sobel-Auswertung von {experiment} wurde abgebrochen.')
                image = load_image(input_path)
                markers, intensity_profile, sobel_profile = find_two_markers(
                    image,
                    previous_markers=previous_markers,
                )
                previous_markers = markers
                output_path = save_sobel_diagram(image, markers, intensity_profile, sobel_profile, input_path)
                payload = {
                    'experiment': experiment,
                    'index': index,
                    'total': total,
                    'input_path': input_path,
                    'output_path': output_path,
                    'image': image,
                    'markers': markers,
                    'intensity_profile': intensity_profile,
                    'sobel_profile': sobel_profile,
                }
                self.queue.put(('sobel_progress', payload))
        except ProcessingCancelled as exc:
            self.queue.put(('cancelled', (experiment, str(exc))))
            return
        except Exception as exc:
            self.queue.put(('error', (experiment, exc)))
            return

        self.queue.put(('done', (experiment, total, 'sobel')))

    def _update_live_plot(self, payload: dict[str, object]) -> None:
        image = payload['image']
        markers = payload['markers']
        intensity_profile = payload['intensity_profile']
        sobel_profile = payload['sobel_profile']
        input_path = payload['input_path']

        for axis in self.axes:
            axis.clear()
        self.axes[0].imshow(image, cmap='gray', aspect='auto')
        self.axes[0].set_title('Preprocessed Bild')
        self.axes[0].set_xlabel('X [px]')
        self.axes[0].set_ylabel('Y [px]')
        self.axes[1].imshow(image, cmap='gray', aspect='auto')
        for marker in markers:
            self.axes[1].axhline(int(marker), color='red', linewidth=2)
        self.axes[1].set_title('Bild mit Markierungen')
        self.axes[1].set_xlabel('X [px]')
        self.axes[1].set_ylabel('Y [px]')
        y = np.arange(image.shape[0])
        self.axes[2].plot(intensity_profile, y, label='Intensitaetsprofil', color='black')
        self.axes[2].plot(sobel_profile, y, label='|Sobel Y| Profil', color='royalblue')
        for index, marker in enumerate(markers, start=1):
            self.axes[2].axhline(int(marker), color='red', linestyle='--', linewidth=1.5, label=f'Markierung {index}')
        self.axes[2].invert_yaxis()
        self.axes[2].set_title(Path(input_path).stem)
        self.axes[2].set_xlabel('Profilwert')
        self.axes[2].set_ylabel('Y [px]')
        handles, labels = self.axes[2].get_legend_handles_labels()
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        self.axes[2].legend(unique.values(), unique.keys(), loc='best')
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            while True:
                event, payload = self.queue.get_nowait()
                if event == 'preprocess_progress':
                    experiment, index, total, output_path = payload
                    self.progress['value'] = index / total * 100
                    self.status_var.set(f'{experiment}: {index} von {total} Bildern preprocesset')
                    self.progress_var.set(f'Letzte Datei: {output_path.name}')
                    if index == 1 or index % 250 == 0 or index == total:
                        self.log(f'{experiment}: {index}/{total} -> {output_path.name}')
                elif event == 'sobel_progress':
                    experiment = payload['experiment']
                    index = payload['index']
                    total = payload['total']
                    output_path = payload['output_path']
                    input_path = payload['input_path']
                    self.progress['value'] = index / total * 100
                    self.status_var.set(f'{experiment}: {index} von {total} Bildern mit Sobel ausgewertet')
                    self.progress_var.set(f'Letztes Diagramm: {Path(output_path).name}')
                    self._update_live_plot(payload)
                    if index == 1 or index % 100 == 0 or index == total:
                        self.log(f'{experiment}: {index}/{total} -> {Path(input_path).name}')
                elif event == 'done':
                    experiment, total, mode = payload
                    self.progress['value'] = 100
                    if mode == 'preprocess':
                        self.status_var.set(f'Fertig: {experiment} wurde komplett preprocesset.')
                        self.progress_var.set(f'{total} Bilder gespeichert in data/preprocessed/{experiment}')
                    else:
                        self.status_var.set(f'Fertig: Sobel fuer {experiment} abgeschlossen.')
                        self.progress_var.set(f'{total} Diagramme gespeichert in figures/sobel/{experiment}')
                    self.log(f'Fertig: {experiment} ({total} Bilder)')
                    self._set_busy(False)
                    self.active_mode = None
                    self.cancel_event.clear()
                elif event == 'cancelled':
                    experiment, message = payload
                    self.status_var.set(f'Abgebrochen: {experiment}')
                    self.progress_var.set(message)
                    self.log(message)
                    self._set_busy(False)
                    self.active_mode = None
                    self.cancel_event.clear()
                elif event == 'error':
                    experiment, exc = payload
                    self.status_var.set(f'Fehler bei {experiment}.')
                    self.progress_var.set(str(exc))
                    self.log(f'Fehler: {exc}')
                    self._set_busy(False)
                    self.active_mode = None
                    self.cancel_event.clear()
                    messagebox.showerror('Vorgang fehlgeschlagen', str(exc))
        except queue.Empty:
            pass

        self.root.after(150, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if 'vista' in style.theme_names():
        style.theme_use('vista')
    PreprocessGui(root)
    root.mainloop()


if __name__ == '__main__':
    main()
