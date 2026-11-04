
#!/usr/bin/env python3
"""
Автоматическая разметка кадров для обучения YOLO по эталонной траектории + параметрам камер.
3D-коробка грузовика проецируется в каждую камеру -> рамка в формате YOLO.

python autolabel.py --videos videos/cam1.mp4 ... videos/cam5.mp4 \
   --cams data/object_transform_CineCameraActor1.csv ... --truth data/object_transform_SM_AA_track.csv \
   --every 10 --out dataset
Потом: yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=60 imgsz=960
"""
import argparse, os, random, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track3d import load_camera_csv, load_truth, projection_matrix, project

def corners(pt, heading, size, along, zdown):
    L, W, H = size
    c, s = np.cos(heading), np.sin(heading)
    ex, ey = np.array([c, s, 0]), np.array([-s, c, 0])
    base = pt + along * ex + np.array([0, 0, -zdown])
    return np.array([base + a * L * ex + b * W * ey + np.array([0, 0, h * H])
                     for a in (-.5, .5) for b in (-.5, .5) for h in (0, 1)])

def bbox(P, cs, W, H):
    pr = [project(P, c) for c in cs]
    if min(z for _, z in pr) <= 0: return None
    uv = np.array([p for p, _ in pr])
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    area = (x2 - x1) * (y2 - y1)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if cx2 <= cx1 or cy2 <= cy1 or area <= 0: return None
    if (cx2 - cx1) * (cy2 - cy1) / area < 0.5: return None      # больше половины за кадром
    if (cx2 - cx1) < 12 or (cy2 - cy1) < 8: return None          # слишком мелко
    return cx1, cy1, cx2, cy2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True); ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--truth", required=True); ap.add_argument("--out", default="dataset")
    ap.add_argument("--every", type=int, default=10); ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--size", nargs=3, type=float, default=[2178, 548, 789], help="длина ширина высота, см")
    ap.add_argument("--along", type=float, default=-1213, help="сдвиг центра коробки вдоль курса от точки трека, см")
    ap.add_argument("--zdown", type=float, default=317, help="на сколько низ грузовика ниже точки трека, см")
    ap.add_argument("--skip", nargs="*", default=[], help="кадры без грузовика (горa): cam:start-end, напр. 1:160-1001 3:0-380")
    a = ap.parse_args()
    skips = {}
    for s in a.skip:
        c, r = s.split(":"); lo, hi = r.split("-"); skips.setdefault(int(c), []).append((int(lo), int(hi)))
    cams = [load_camera_csv(p) for p in a.cams]; tr = load_truth(a.truth)
    for sp in ("train", "val"):
        os.makedirs(f"{a.out}/images/{sp}", exist_ok=True); os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)
    os.makedirs(f"{a.out}/preview", exist_ok=True)
    random.seed(0); n_tot = 0
    for ci, vp in enumerate(a.videos, 1):
        cap = cv2.VideoCapture(vp); W = int(cap.get(3)); H = int(cap.get(4)); d, f, sw = cams[ci - 1]
        i = 0; prev = []; last_h = 0.0; n = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % a.every == 0 and i in d.index and i in tr.index and not any(lo <= i <= hi for lo, hi in skips.get(ci, [])):
                j0, j1 = max(i - 5, tr.index.min()), min(i + 5, tr.index.max())
                v = tr.loc[j1].values - tr.loc[j0].values
                if np.linalg.norm(v[:2]) > 1: last_h = np.arctan2(v[1], v[0])
                P = projection_matrix(d.loc[i], f, sw, W, H)
                b = bbox(P, corners(tr.loc[i].values, last_h, a.size, a.along, a.zdown), W, H)
                if b is not None:
                    sp = "val" if random.random() < a.val else "train"; name = f"cam{ci}_{i:04d}"
                    cv2.imwrite(f"{a.out}/images/{sp}/{name}.jpg", fr)
                    x1, y1, x2, y2 = b
                    open(f"{a.out}/labels/{sp}/{name}.txt", "w").write(
                        f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
                    n += 1
                    if len(prev) < 12 and n % 3 == 1:
                        p = fr.copy(); cv2.rectangle(p, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        cv2.putText(p, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                        prev.append(cv2.resize(p, (480, int(480 * H / W))))
            i += 1
        n_tot += n; print(f"cam{ci}: {n} кадров")
        if prev:
            rows = [np.hstack(prev[k:k + 4] + [np.zeros_like(prev[0])] * (4 - len(prev[k:k + 4]))) for k in range(0, len(prev), 4)]
            cv2.imwrite(f"{a.out}/preview/cam{ci}.jpg", np.vstack(rows))
    open(f"{a.out}/data.yaml", "w").write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: truck\n")
    print("всего кадров:", n_tot, "| проверьте", f"{a.out}/preview/*.jpg")

if __name__ == "__main__":
    main()
