from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_DIR = ROOT / 'src' / 'preprocessesing'
SOBEL_DIR = ROOT / 'src' / 'sobelAlg'
STRAIN_DIR = ROOT / 'src' / 'strain'

import sys
for folder in (PREPROCESS_DIR, SOBEL_DIR, STRAIN_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from rectify_experiment import ProcessingCancelled, process_experiment  # noqa: E402
from detect_markers import INPUT_ROOT, OUTPUT_ROOT, find_two_markers, load_image, save_sobel_diagram  # noqa: E402
from strain_analysis import compute_strain, load_calibration, open_strain_csv  # noqa: E402


DEFAULT_UPDATE_EVERY = 10


class PreprocessGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title('Versuch Preprocessing und Sobel')
        self.root.geometry('1480x960')
        self.root.minsize(1260, 880)

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.active_mode: str | None = None
        self.cancel_event = threading.Event()
        self.calibration = load_calibration()
        self.live_frames: list[int] = []
        self.live_time_ms: list[float] = []
        self.live_strain_values: list[float] = []
        self.live_distance_values: list[float] = []
        self.live_y_top_raw: list[float] = []
        self.live_y_bot_raw: list[float] = []

        self.experiment_var = tk.StringVar(value=' ')
        self.update_every_var = tk.StringVar(value=str(DEFAULT_UPDATE_EVERY))
        self.status_var = tk.StringVar(value='Bereit. Bitte Ordnernamen eingeben.')
        self.progress_var = tk.StringVar(value='Noch nicht gestartet.')

        self._build_ui()
        self.root.after(150, self._poll_queue)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill='both', expand=True)
        top = ttk.Frame(outer)
        top.pack(fill='both', expand=True)
        left = ttk.Frame(top)
        left.pack(side='left', fill='y')

        ttk.Label(left, text='Versuch Pipeline', font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        ttk.Label(left, text='Den kompletten Ordnernamen eingeben. Beispiele: Versuch_1 oder Abfahrrampe bis 0_II.').pack(anchor='w', pady=(6, 18))

        input_row = ttk.Frame(left)
        input_row.pack(fill='x')
        ttk.Label(input_row, text='Ordnername:').pack(side='left')
        self.entry = ttk.Entry(input_row, textvariable=self.experiment_var, width=24)
        self.entry.pack(side='left', padx=(10, 12))
        self.entry.focus_set()
        self.entry.bind('<Return>', lambda _event: self.start_preprocess())

        interval_row = ttk.Frame(left)
        interval_row.pack(fill='x', pady=(10, 0))
        ttk.Label(interval_row, text='Jedes').pack(side='left')
        self.interval_entry = ttk.Entry(interval_row, textvariable=self.update_every_var, width=8)
        self.interval_entry.pack(side='left', padx=(10, 12))
        ttk.Label(interval_row, text='Bild soll gespeichert und live geplottet werden. Je höher die Zahl, desto schneller wird geplottet.').pack(side='left')

        button_row = ttk.Frame(left)
        button_row.pack(fill='x', pady=(14, 0))
        self.preprocess_button = ttk.Button(button_row, text='Preprocess starten', command=self.start_preprocess)
        self.preprocess_button.pack(side='left')
        self.sobel_button = ttk.Button(button_row, text='Sobel auf preprocessed', command=self.start_sobel)
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
        self.figure = Figure(figsize=(11.8, 7.4), dpi=100)
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0])
        self.marker_axis = self.figure.add_subplot(grid[0, 0])
        self.profile_axis = self.figure.add_subplot(grid[0, 1])
        self.y_position_axis = self.figure.add_subplot(grid[1, 0])
        self.strain_axis = self.figure.add_subplot(grid[1, 1])
        self.distance_axis = None
        self.figure.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.08, wspace=0.28, hspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.live_frames.clear()
        self.live_time_ms.clear()
        self.live_strain_values.clear()
        self.live_distance_values.clear()
        self.live_y_top_raw.clear()
        self.live_y_bot_raw.clear()
        placeholders = [
            (self.marker_axis, 'Bild mit Markierungen'),
            (self.profile_axis, 'Sobel-Profil'),
            (self.y_position_axis, 'Marker-Y-Position ueber Zeit'),
            (self.strain_axis, 'Dehnung ueber Zeit'),
        ]
        for axis, title in placeholders:
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, 'Noch keine Daten', ha='center', va='center', transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        self.canvas.draw_idle()

    def _infer_fps(self, experiment: str) -> float:
        normalized = experiment.strip().lower()
        if normalized.startswith('versuch_'):
            try:
                number = int(experiment.split('_')[-1])
            except (TypeError, ValueError):
                return 1.0
            if 1 <= number <= 6:
                return 2000.0
            if 8 <= number <= 11:
                return 6000.0
            return 1.0
        if normalized.startswith('abfahrrampe bis'):
            return 2000.0
        return 1.0

    def _smooth_series(self, values: list[float], window: int = 15) -> np.ndarray:
        if not values:
            return np.array([], dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < 3:
            return arr
        window = max(3, min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1))
        if window % 2 == 0:
            window -= 1
        kernel = np.ones(window, dtype=np.float64) / float(window)
        padded = np.pad(arr, (window // 2, window // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

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
        self.interval_entry.configure(state=state)
        self.cancel_button.configure(state='normal' if busy else 'disabled')

    def _normalize_experiment(self) -> str | None:
        experiment_name = self.experiment_var.get().strip()
        if not experiment_name:
            messagebox.showerror('Fehlende Eingabe', 'Bitte einen Ordnernamen eingeben.')
            return None
        return experiment_name

    def _normalize_update_every(self) -> int | None:
        raw_value = self.update_every_var.get().strip()
        if not raw_value:
            messagebox.showerror('Fehlende Eingabe', 'Bitte ein Intervall n eingeben.')
            return None
        try:
            value = int(raw_value)
        except ValueError:
            messagebox.showerror('Ungueltiges Intervall', 'Das Intervall n muss eine ganze Zahl sein.')
            return None
        if value <= 0:
            messagebox.showerror('Ungueltiges Intervall', 'Das Intervall n muss groesser als 0 sein.')
            return None
        return value

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
        self._start_sobel_mode('sobel_preprocessed', INPUT_ROOT, OUTPUT_ROOT, 'Sobel-Auswertung fuer')

    def _start_sobel_mode(self, mode: str, input_root: Path, output_root: Path, title_prefix: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        experiment = self._normalize_experiment()
        if experiment is None:
            return
        update_every = self._normalize_update_every()
        if update_every is None:
            return

        experiment_dir = input_root / experiment
        if not experiment_dir.exists():
            messagebox.showerror('Fehlender Versuch', f'Ordner nicht gefunden: {experiment_dir}')
            return

        self.cancel_event.clear()
        self.active_mode = mode
        self.progress['value'] = 0
        self.status_var.set(f'Starte {title_prefix} {experiment} ...')
        self.progress_var.set(f'Marker, Profile und Dehnung werden alle {update_every} Bilder aktualisiert.')
        self._set_busy(True)
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')
        self._draw_placeholder()
        self.worker = threading.Thread(target=self._run_sobel, args=(experiment, input_root, output_root, mode, update_every), daemon=True)
        self.worker.start()

    def _run_preprocess(self, experiment: str) -> None:
        def progress_callback(index: int, total: int, output_path: Path) -> None:
            self.queue.put(('preprocess_progress', (experiment, index, total, output_path)))

        try:
            total = process_experiment(experiment, progress_callback=progress_callback, should_cancel=self.cancel_event.is_set)
        except ProcessingCancelled as exc:
            self.queue.put(('cancelled', (experiment, str(exc))))
            return
        except Exception as exc:
            self.queue.put(('error', (experiment, exc)))
            return

        self.queue.put(('done', (experiment, total, 'preprocess', 'data/preprocessed', None)))

    def _run_sobel(self, experiment: str, input_root: Path, output_root: Path, mode: str, update_every: int) -> None:
        files = sorted((input_root / experiment).glob('*.png' if input_root == INPUT_ROOT else '*.tif'))
        if not files:
            self.queue.put(('error', (experiment, FileNotFoundError(f'Keine Bilddateien gefunden in {input_root / experiment}'))))
            return

        total = len(files)
        fps = self._infer_fps(experiment)
        previous_markers: np.ndarray | None = None
        csv_handle = None
        csv_path = None
        try:
            csv_handle, csv_writer, csv_path = open_strain_csv(experiment)
            for index, input_path in enumerate(files, start=1):
                if self.cancel_event.is_set():
                    raise ProcessingCancelled(f'Sobel-Auswertung von {experiment} wurde abgebrochen.')
                image = load_image(input_path)
                markers, intensity_profile, sobel_profile = find_two_markers(image, previous_markers=previous_markers)
                previous_markers = markers
                strain_result = compute_strain(markers, image.shape[0], self.calibration)
                output_path = None
                if index == 1 or index % update_every == 0 or index == total:
                    output_path = save_sobel_diagram(
                        image,
                        markers,
                        intensity_profile,
                        sobel_profile,
                        input_path,
                        input_root=input_root,
                        output_root=output_root,
                    )
                time_ms = (index - 1) / fps * 1000.0 if fps > 0 else float(index - 1)
                csv_writer.writerow([
                    index,
                    input_path.name,
                    f'{strain_result.y_top_px:.3f}',
                    f'{strain_result.y_bot_px:.3f}',
                    f'{strain_result.y_center_px:.3f}',
                    f'{strain_result.distance_px:.3f}',
                    f'{strain_result.mm_per_px:.6f}',
                    f'{strain_result.distance_mm:.6f}',
                    f'{strain_result.strain_percent:.6f}',
                ])
                progress_payload = {
                    'mode': mode,
                    'experiment': experiment,
                    'index': index,
                    'total': total,
                    'input_path': input_path,
                    'output_path': output_path,
                    'strain_result': strain_result,
                    'time_ms': time_ms,
                }
                self.queue.put(('sobel_progress', progress_payload))

                if index == 1 or index % update_every == 0 or index == total:
                    plot_payload = {
                        'index': index,
                        'input_path': input_path,
                        'image': image,
                        'markers': markers,
                        'intensity_profile': intensity_profile,
                        'sobel_profile': sobel_profile,
                        'strain_result': strain_result,
                        'time_ms': time_ms,
                        'total': total,
                    }
                    self.queue.put(('sobel_plot', plot_payload))
        except ProcessingCancelled as exc:
            self.queue.put(('cancelled', (experiment, str(exc))))
            return
        except Exception as exc:
            self.queue.put(('error', (experiment, exc)))
            return
        finally:
            if csv_handle is not None:
                csv_handle.close()

        target_dir = 'figures/sobel'
        self.queue.put(('done', (experiment, total, mode, target_dir, str(csv_path) if csv_path is not None else None)))

    def _update_live_plot(self, payload: dict[str, object]) -> None:
        image = payload['image']
        markers = payload['markers']
        intensity_profile = payload['intensity_profile']
        sobel_profile = payload['sobel_profile']
        input_path = payload['input_path']
        strain_result = payload['strain_result']
        frame_index = int(payload['index'])
        time_ms = float(payload['time_ms'])

        self.live_frames.append(frame_index)
        self.live_time_ms.append(time_ms)
        self.live_distance_values.append(float(strain_result.distance_mm))
        self.live_strain_values.append(float(strain_result.strain_percent))
        self.live_y_top_raw.append(float(strain_result.y_top_px))
        self.live_y_bot_raw.append(float(strain_result.y_bot_px))

        y_top_smoothed = self._smooth_series(self.live_y_top_raw)
        y_bot_smoothed = self._smooth_series(self.live_y_bot_raw)

        self.marker_axis.clear()
        self.profile_axis.clear()
        self.y_position_axis.clear()
        self.strain_axis.clear()
        if self.distance_axis is not None:
            self.distance_axis.remove()
            self.distance_axis = None

        self.marker_axis.imshow(image, cmap='gray', aspect='auto')
        for marker in markers:
            self.marker_axis.axhline(int(marker), color='red', linewidth=2)
        self.marker_axis.set_title('Bild mit Markierungen')
        self.marker_axis.set_xlabel('X [px]')
        self.marker_axis.set_ylabel('Y [px]')

        y = np.arange(image.shape[0])
        self.profile_axis.plot(intensity_profile, y, label='Intensitaetsprofil', color='black')
        self.profile_axis.plot(sobel_profile, y, label='|Sobel Y| Profil', color='royalblue')
        for marker_index, marker in enumerate(markers, start=1):
            self.profile_axis.axhline(int(marker), color='red', linestyle='--', linewidth=1.5, label=f'Markierung {marker_index}')
        self.profile_axis.invert_yaxis()
        self.profile_axis.set_title(Path(input_path).stem)
        self.profile_axis.set_xlabel('Profilwert')
        self.profile_axis.set_ylabel('Y [px]')
        handles, labels = self.profile_axis.get_legend_handles_labels()
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        self.profile_axis.legend(unique.values(), unique.keys(), loc='best')

        self.y_position_axis.plot(self.live_time_ms, self.live_y_bot_raw, color='#f6b26b', alpha=0.55, linewidth=1.0, label='y_bot (roh)')
        self.y_position_axis.plot(self.live_time_ms, self.live_y_top_raw, color='#93c47d', alpha=0.55, linewidth=1.0, label='y_top (roh)')
        self.y_position_axis.plot(self.live_time_ms, y_bot_smoothed, color='#ff6f00', linewidth=2.0, label='y_bot (geglaettet)')
        self.y_position_axis.plot(self.live_time_ms, y_top_smoothed, color='green', linewidth=2.0, label='y_top (geglaettet)')
        self.y_position_axis.invert_yaxis()
        self.y_position_axis.set_title(f'Marker-Y-Position - {self._normalize_experiment() or "Versuch"}')
        self.y_position_axis.set_xlabel('Zeit (ms)')
        self.y_position_axis.set_ylabel('Y-Koordinate (px)')
        self.y_position_axis.grid(True, alpha=0.3)
        self.y_position_axis.legend(loc='best')

        total = int(payload.get('total', frame_index))

        strain_line = self.strain_axis.plot(self.live_time_ms, self.live_strain_values, color='darkorange', label='Dehnung [%]')[0]
        self.strain_axis.scatter(self.live_time_ms[-1], self.live_strain_values[-1], color='darkorange', s=28)
        self.strain_axis.set_title(f'Frame {frame_index}: {strain_result.distance_mm:.2f} mm, {strain_result.strain_percent:.2f} %')
        self.strain_axis.set_xlabel('Zeit (ms)')
        self.strain_axis.set_ylabel('Dehnung [%]', color='darkorange')
        self.strain_axis.set_ylim(-5.0, 5.0)
        self.strain_axis.tick_params(axis='y', colors='darkorange')
        self.strain_axis.grid(True, alpha=0.3)

        self.distance_axis = self.strain_axis.twinx()
        distance_line = self.distance_axis.plot(self.live_time_ms, self.live_distance_values, color='darkgreen', label='Abstand [mm]')[0]
        self.distance_axis.set_ylabel('Abstand [mm]', color='darkgreen')
        self.distance_axis.set_ylim(15.0, 21.0)
        self.distance_axis.tick_params(axis='y', colors='darkgreen')

        legend_handles = [strain_line, distance_line]
        legend_labels = ['Dehnung [%]', 'Abstand [mm]']
        if frame_index == total and len(self.live_time_ms) >= 2:
            time_arr = np.asarray(self.live_time_ms, dtype=np.float64)
            strain_arr = np.asarray(self.live_strain_values, dtype=np.float64)
            distance_arr = np.asarray(self.live_distance_values, dtype=np.float64)

            strain_coeffs = np.polyfit(time_arr, strain_arr, 1)
            strain_fit = np.polyval(strain_coeffs, time_arr)
            strain_fit_line = self.strain_axis.plot(time_arr, strain_fit, color='firebrick', linestyle='--', linewidth=1.8, label='Regression Dehnung')[0]

            distance_coeffs = np.polyfit(time_arr, distance_arr, 1)
            distance_fit = np.polyval(distance_coeffs, time_arr)
            distance_fit_line = self.distance_axis.plot(time_arr, distance_fit, color='navy', linestyle='--', linewidth=1.8, label='Regression Abstand')[0]

            legend_handles.extend([strain_fit_line, distance_fit_line])
            legend_labels.extend(['Regression Dehnung', 'Regression Abstand'])

        self.strain_axis.legend(legend_handles, legend_labels, loc='best')

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
                    mode = payload['mode']
                    experiment = payload['experiment']
                    index = payload['index']
                    total = payload['total']
                    output_path = payload['output_path']
                    input_path = payload['input_path']
                    strain_result = payload['strain_result']
                    self.progress['value'] = index / total * 100
                    label = 'preprocessed'
                    self.status_var.set(f'{experiment}: {index} von {total} Bildern mit Sobel auf {label} ausgewertet')
                    if output_path is not None:
                        self.progress_var.set(
                            f'Letztes Diagramm: {Path(output_path).name} | Abstand {strain_result.distance_mm:.2f} mm | Dehnung {strain_result.strain_percent:.2f} %'
                        )
                    else:
                        self.progress_var.set(
                            f'Kein Diagramm fuer dieses Bild gespeichert | Abstand {strain_result.distance_mm:.2f} mm | Dehnung {strain_result.strain_percent:.2f} %'
                        )
                    if index == 1 or index % 100 == 0 or index == total:
                        self.log(f'{experiment}: {index}/{total} -> {Path(input_path).name}')
                elif event == 'sobel_plot':
                    self._update_live_plot(payload)
                elif event == 'done':
                    experiment, total, mode, target_dir, csv_path = payload
                    self.progress['value'] = 100
                    if mode == 'preprocess':
                        self.status_var.set(f'Fertig: {experiment} wurde komplett preprocesset.')
                        self.progress_var.set(f'{total} Bilder gespeichert in {target_dir}/{experiment}')
                    else:
                        self.status_var.set(f'Fertig: Sobel fuer {experiment} abgeschlossen.')
                        if csv_path:
                            self.progress_var.set(f'{total} Diagramme gespeichert in {target_dir}/{experiment} | CSV: {csv_path}')
                        else:
                            self.progress_var.set(f'{total} Diagramme gespeichert in {target_dir}/{experiment}')
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
