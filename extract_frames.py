"""Достаёт кадры из видео для разметки.
python extract_frames.py --videos videos/cam1.mp4 ... videos/cam5.mp4 --every 20 --out frames
"""
import argparse, os, cv2
ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--out", default="frames")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
n = 0
for k, v in enumerate(a.videos, 1):
    cap = cv2.VideoCapture(v); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % a.every == 0:
            cv2.imwrite(os.path.join(a.out, f"cam{k}_{i:04d}.jpg"), f); n += 1
        i += 1
print("сохранено кадров:", n)
