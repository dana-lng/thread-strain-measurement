"""
Hilfsskript: Zeigt Schachbrettbild vor und nach Scherungskorrektur
als PNG-Vergleich (wie Annotationstool-Anzeige).
Aufruf: python show_correction.py
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, '.')
from image_utils import load_and_crop_calibrated, load_camera_calibration

# Kalibrierung laden
calib = load_camera_calibration('camera_calibration.json')

# Original (nur crop, keine Korrektur)
img_orig = load_and_crop_calibrated(
    '../Massstab/Checkerboard_Pattern_10mm.tif',
    calib_params=None)

# Korrigiert (crop + Scherungskorrektur)
img_corr = load_and_crop_calibrated(
    '../Massstab/Checkerboard_Pattern_10mm.tif',
    calib_params=calib)

print(f'Original:   {img_orig.shape[1]}x{img_orig.shape[0]} px')
print(f'Korrigiert: {img_corr.shape[1]}x{img_corr.shape[0]} px')

def contrast_stretch(img):
    """Kontraststreckung wie preprocess_for_display im Annotationstool."""
    lo, hi = np.percentile(img, [1, 99])
    return np.clip(
        (img.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255
    ).astype('uint8')

img_orig_disp = contrast_stretch(img_orig)
img_corr_disp = contrast_stretch(img_corr)

# Nebeneinander-Vergleich (Original links | Korrigiert rechts)
h1, w1 = img_orig_disp.shape
h2, w2 = img_corr_disp.shape
h_max = max(h1, h2)

left  = cv2.copyMakeBorder(img_orig_disp, 0, h_max - h1, 0, 0,
                            cv2.BORDER_CONSTANT, value=128)
right = cv2.copyMakeBorder(img_corr_disp, 0, h_max - h2, 0, 0,
                            cv2.BORDER_CONSTANT, value=128)

sep = np.full((h_max, 4), 200, dtype='uint8')
combined_gray = np.hstack([left, sep, right])
combined = cv2.cvtColor(combined_gray, cv2.COLOR_GRAY2BGR)

# Beschriftungen
cv2.putText(combined, 'ORIGINAL', (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
cv2.putText(combined, '(vor Scherungskorrektur)', (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
cv2.putText(combined, 'KORRIGIERT', (w1 + 8, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
cv2.putText(combined, '(Scherung entfernt)', (w1 + 8, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

out = 'checkerboard_korrektur_vergleich.png'
cv2.imwrite(out, combined)
print(f'\nGespeichert: {out}')
print(f'Groesse:     {combined.shape[1]}x{combined.shape[0]} px')
print('\nOeffne die Datei um zu pruefen ob die Schachbrettzeilen')
print('im rechten Bild (KORRIGIERT) waagerecht verlaufen.')
