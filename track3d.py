#!/usr/bin/env python3
"""
Мультикамерный 3D-трекинг грузовика (Unreal Engine, сантиметры).

Пример:
  python track3d.py --videos cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 cam5.mp4 \
      --cams object_transform_CineCameraActor1.csv ... object_transform_CineCameraActor5.csv \
      --overlay-cam 5 --truth object_transform_SM_AA_track.csv --out out

Проверка без видео и детектора:  python track3d.py --selftest --cams ... --truth ...
Зависимости: numpy pandas opencv-python scipy matplotlib ultralytics
"""
import argparse, os, re
import numpy as np, pandas as pd, cv2
from scipy.optimize import least_squares

# ---------------------------------------------------------------- чтение CSV
def _num(s):
    return float(str(s).replace(",", "."))

def load_camera_csv(path):
    """Возвращает DataFrame [X,Y,Z,Roll,Pitch,Yaw] по кадрам, focal_mm, sensor_w_mm."""
    raw = pd.read_csv(path, sep=";", encoding="cp1251", header=None, dtype=str)
    focal = _num(raw.iloc[0, 10])
    sensor_w = 23.76
    if raw.shape[1] > 11 and isinstance(raw.iloc[1, 11], str):
        m = re.match(r"\s*([\d,\.]+)\s*[*xх]\s*([\d,\.]+)", raw.iloc[1, 11])
        if m:
            sensor_w = _num(m.group(1))
    d = raw.iloc[1:, :7].map(_num)
    d.columns = ["Frame", "X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    return d.set_index("Frame"), focal, sensor_w

def load_truth(path):
    t = pd.read_csv(path, sep=";", encoding="cp1251", dtype=str)
    for c in t.columns:  # битые ячейки (Excel превратил число в дату) -> NaN
        t[c] = pd.to_numeric(t[c].str.replace(",", "."), errors="coerce")
    return t.set_index("Frame").interpolate(limit_direction="both")[["Loc_X", "Loc_Y", "Loc_Z"]]

# ---------------------------------------------------------------- геометрия
def unreal_axes(roll, pitch, yaw):
    """Unreal: левая система, X вперёд, Y вправо, Z вверх; +pitch = взгляд вверх."""
    r, p, y = np.radians([roll, pitch, yaw])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    R = Rz @ Ry @ Rx
    return R[:, 0], R[:, 1], R[:, 2]  # вперёд, вправо, вверх

def projection_matrix(row, focal, sensor_w, W, H):
    fwd, right, up = unreal_axes(row.Roll, row.Pitch, row.Yaw)
    Rcw = np.vstack([right, -up, fwd])  # камера: x вправо, y вниз, z вперёд
    fx = focal * W / sensor_w
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    c = np.array([row.X, row.Y, row.Z], float)
    return K @ np.hstack([Rcw, (-Rcw @ c).reshape(3, 1)])

def project(P, X):
    q = P @ np.append(X, 1.0)
    return q[:2] / q[2], q[2]

def dlt(Ps, pts, w=None):
    A = []
    for k, (P, (u, v)) in enumerate(zip(Ps, pts)):
        s = 1.0 if w is None else w[k]
        A += [s * (u * P[2] - P[0]), s * (v * P[2] - P[1])]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else None

def reproj_err(P, X, uv):
    p, z = project(P, X)
    return np.inf if z <= 0 else float(np.linalg.norm(p - uv))

def refine(X0, Ps, pts):
    def res(X):
        out = []
        for P, uv in zip(Ps, pts):
            p, _ = project(P, X)
            out += list(p - uv)
        return out
    return least_squares(res, X0).x

def triangulate_robust(Ps, pts, thr=8.0):
    """Перебор пар камер -> лучшая гипотеза по числу inliers -> DLT + уточнение по inliers."""
    n = len(Ps)
    if n < 2:
        return None, [], np.nan
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            X = dlt([Ps[i], Ps[j]], [pts[i], pts[j]])
            if X is None:
                continue
            errs = [reproj_err(Ps[k], X, pts[k]) for k in range(n)]
            inl = [k for k in range(n) if errs[k] < thr]
            score = (len(inl), -sum(errs[k] for k in inl))
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or len(best[1]) < 2:
        return None, [], np.nan
    inl = best[1]
    X = dlt([Ps[k] for k in inl], [pts[k] for k in inl])
    if X is None:
        return None, [], np.nan
    X = refine(X, [Ps[k] for k in inl], [pts[k] for k in inl])
    err = np.mean([reproj_err(Ps[k], X, pts[k]) for k in inl])
    return X, inl, err

# ---------------------------------------------------------------- фильтр Калмана + RTS
def kalman_rts(meas, q=50.0, r=100.0, gate=9.0):
    """meas: (N,3) с NaN на пропусках, в тех же единицах (см). Модель постоянной скорости.
    Возвращает сглаженные (N,3) и статус каждого кадра."""
    N = len(meas)
    F = np.eye(6); F[:3, 3:] = np.eye(3)
    Q = np.diag([q * 0.25] * 3 + [q] * 3)
    H = np.hstack([np.eye(3), np.zeros((3, 3))]); Rm = np.eye(3) * r ** 2
    xs, Ps, xp, Pp, status = [], [], [], [], []
    x = None; P = None
    for t in range(N):
        z = meas[t]
        if x is None:
            if np.isnan(z).any():
                xs.append(np.full(6, np.nan)); Ps.append(np.eye(6)); xp.append(np.full(6, np.nan)); Pp.append(np.eye(6)); status.append("lost"); continue
            x = np.r_[z, np.zeros(3)]; P = np.eye(6) * 1e4
            xp.append(x.copy()); Pp.append(P.copy()); xs.append(x.copy()); Ps.append(P.copy()); status.append("init"); continue
        x_pred = F @ x; P_pred = F @ P @ F.T + Q
        xp.append(x_pred.copy()); Pp.append(P_pred.copy())
        st = "predicted"
        if not np.isnan(z).any():
            S = H @ P_pred @ H.T + Rm
            inn = z - H @ x_pred
            if inn @ np.linalg.solve(S, inn) < gate ** 2 * 3:
                K = P_pred @ H.T @ np.linalg.inv(S)
                x = x_pred + K @ inn; P = (np.eye(6) - K @ H) @ P_pred; st = "tracked"
            else:
                x, P = x_pred, P_pred; st = "outlier"
        else:
            x, P = x_pred, P_pred
        xs.append(x.copy()); Ps.append(P.copy()); status.append(st)
    xs = np.array(xs); sm = xs.copy()
    for t in range(N - 2, -1, -1):  # RTS
        if np.isnan(xs[t]).any() or np.isnan(xp[t + 1]).any():
            continue
        C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        sm[t] = xs[t] + C @ (sm[t + 1] - xp[t + 1])
    return sm[:, :3], sm[:, 3:], status

# ---------------------------------------------------------------- детекторы
class YoloDetector:
    def __init__(self, model="yolov8n.pt", conf=0.25, classes=(7, 2, 5)):  # COCO: 7 truck, 2 car, 5 bus
        from ultralytics import YOLO
        self.m = YOLO(model); self.conf = conf; self.classes = set(classes)

    def __call__(self, frame):
        r = self.m(frame, verbose=False, conf=self.conf)[0]
        best = None
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(k) in self.classes and (best is None or c > best[1]):  # truck/car/bus
                best = (b, float(c))
        return best  # ((x1,y1,x2,y2), conf) | None

class ColorDetector:
    """Для --selftest: ищет пурпурную коробку."""
    def __call__(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (140, 150, 150), (160, 255, 255))
        ys, xs = np.nonzero(m)
        if len(xs) < 15:
            return None
        return (np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), 1.0)

# ---------------------------------------------------------------- 3D-коробка грузовика
def truck_box_corners(center, heading, size, centered=False):
    L, Wd, Ht = size
    c, s = np.cos(heading), np.sin(heading)
    ex, ey = np.array([c, s, 0]), np.array([-s, c, 0])
    pts = []
    for a in (-.5, .5):
        for b in (-.5, .5):
            for h in (0, 1):
                pts.append(center + a * L * ex + b * Wd * ey + np.array([0, 0, h * Ht - (Ht / 2 if centered else 0)]))
    return np.array(pts)

def box2d_from_3d(P, corners, W, H):
    uv, zs = zip(*[project(P, c) for c in corners])
    if min(zs) <= 0:
        return None
    uv = np.array(uv)
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    if x2 < 0 or y2 < 0 or x1 > W or y1 > H:
        return None
    return x1, y1, x2, y2

# ---------------------------------------------------------------- основной конвейер
def run(args):
    cams = [load_camera_csv(p) for p in args.cams]
    n_cam = len(cams)
    caps, W, H, nfr = None, None, None, None
    if args.selftest:
        return selftest(args, cams)
    caps = [cv2.VideoCapture(v) for v in args.videos]
    W = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfr = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    nfr = min(nfr, len(cams[0][0]))
    det = YoloDetector(args.model, args.conf, args.cls)
    os.makedirs(args.out, exist_ok=True)
    meas, ninl, rerr = np.full((nfr, 3), np.nan), np.zeros(nfr, int), np.full(nfr, np.nan)
    dets = [[None] * n_cam for _ in range(nfr)]
    for t in range(nfr):
        Ps, pts, ids = [], [], []
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if not ok:
                continue
            d = det(frame)
            dets[t][i] = d
            if d is None:
                continue
            (x1, y1, x2, y2), conf = d
            d_, f_, s_ = cams[i]
            Ps.append(projection_matrix(d_.iloc[t], f_, s_, W, H)); pts.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2])); ids.append(i)
        X, inl, e = triangulate_robust(Ps, pts, args.thr)
        if X is not None:
            meas[t] = X; ninl[t] = len(inl); rerr[t] = e
        if t % 50 == 0:
            print(f"кадр {t}/{nfr}")
    finish(args, cams, meas, ninl, rerr, W, H, nfr)

def finish(args, cams, meas, ninl, rerr, W, H, nfr, dets=None):
    pos, vel, status = kalman_rts(meas)
    t = np.arange(nfr)
    df = pd.DataFrame({"frame": t, "x_cm": pos[:, 0], "y_cm": pos[:, 1], "z_cm": pos[:, 2],
                       "raw_x": meas[:, 0], "raw_y": meas[:, 1], "raw_z": meas[:, 2],
                       "n_cams": ninl, "reproj_px": rerr, "status": status})
    df.to_csv(os.path.join(args.out, "trajectory.csv"), index=False)
    # графики X(t), Y(t), Z(t)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k, nm in enumerate("XYZ"):
        ax[k].plot(t, meas[:, k] / 100, ".", ms=2, alpha=.4, label="сырые")
        ax[k].plot(t, pos[:, k] / 100, "-", lw=1.5, label="сглаженные")
        if args.truth:
            tr = load_truth(args.truth).reindex(t).values
            ax[k].plot(t, tr[:, k] / 100, "k--", lw=.8, label="эталон")
        ax[k].set_ylabel(f"{nm}, м"); ax[k].grid(alpha=.3)
    ax[0].legend(); ax[2].set_xlabel("кадр"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "plots.png"), dpi=130)
    # метрики
    if args.truth:
        tr = load_truth(args.truth).reindex(t).values
        e = np.linalg.norm(pos - tr, axis=1) / 100
        er = np.linalg.norm(meas - tr, axis=1) / 100
        print(f"Ошибка 3D (сглаженная): среднее {np.nanmean(e):.2f} м, медиана {np.nanmedian(e):.2f} м, p95 {np.nanpercentile(e,95):.2f} м")
        print(f"Ошибка 3D (сырая):      среднее {np.nanmean(er):.2f} м, медиана {np.nanmedian(er):.2f} м")
    print(f"Кадров с >=2 камерами: {(ninl>=2).mean()*100:.0f}%; средняя reprojection-ошибка {np.nanmean(rerr):.2f} px")
    # контрольное видео
    if args.overlay_cam and args.videos:
        write_overlay(args, cams, pos, vel, status, W, H, nfr)

def write_overlay(args, cams, pos, vel, status, W, H, nfr):
    ci = args.overlay_cam - 1
    cap = cv2.VideoCapture(args.videos[ci])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw = cv2.VideoWriter(os.path.join(args.out, f"overlay_cam{args.overlay_cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    d_, f_, s_ = cams[ci]
    last_h = 0.0
    for t in range(nfr):
        ok, frame = cap.read()
        if not ok:
            break
        if not np.isnan(pos[t]).any():
            P = projection_matrix(d_.iloc[t], f_, s_, W, H)
            sp = np.linalg.norm(vel[t][:2])
            if sp > 5: last_h = np.arctan2(vel[t][1], vel[t][0])
            b = box2d_from_3d(P, truck_box_corners(pos[t], last_h, args.truck_size, centered=True), W, H)
            if b is not None:
                x1, y1, x2, y2 = map(int, b)
                col = (0, 255, 0) if status[t] == "tracked" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                cv2.putText(frame, f"{status[t]}  XYZ(m) {pos[t,0]/100:.1f} {pos[t,1]/100:.1f} {pos[t,2]/100:.1f}", (max(x1, 5), max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, col, 2)
        vw.write(frame)
    vw.release()

# ---------------------------------------------------------------- selftest
def selftest(args, cams):
    """Генерирует видео из эталонной траектории (пурпурная коробка) и гоняет весь конвейер."""
    W, H = 960, 540
    tr = load_truth(args.truth); nfr = min(args.max_frames, len(tr))
    os.makedirs(args.out, exist_ok=True)
    size = np.array(args.truck_size, float)
    rng = np.random.default_rng(1)
    det = ColorDetector()
    vids = []
    for i, (d_, f_, s_) in enumerate(cams):
        path = os.path.join(args.out, f"synthetic_cam{i+1}.mp4"); vids.append(path)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for t in range(nfr):
            fr = np.full((H, W, 3), 90, np.uint8)
            X = tr.iloc[t].values; Vv = tr.iloc[min(t + 1, nfr - 1)].values - X
            h = np.arctan2(Vv[1], Vv[0]) if np.linalg.norm(Vv[:2]) > 1e-6 else 0
            b = box2d_from_3d(projection_matrix(d_.iloc[t], f_, s_, W, H), truck_box_corners(X, h, size), W, H)
            if b is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in b]
                cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 0, 255), -1)
            vw.write(fr)
        vw.release()
    args.videos = vids
    caps = [cv2.VideoCapture(v) for v in vids]
    meas, ninl, rerr = np.full((nfr, 3), np.nan), np.zeros(nfr, int), np.full(nfr, np.nan)
    for t in range(nfr):
        Ps, pts = [], []
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            dd = det(frame) if ok else None
            if dd is None: continue
            (x1, y1, x2, y2), _ = dd
            d_, f_, s_ = cams[i]
            Ps.append(projection_matrix(d_.iloc[t], f_, s_, W, H))
            pts.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2]) + rng.normal(0, 1.5, 2))
        X, inl, e = triangulate_robust(Ps, pts, args.thr)
        if X is not None: meas[t] = X; ninl[t] = len(inl); rerr[t] = e
    finish(args, cams, meas, ninl, rerr, W, H, nfr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=[])
    ap.add_argument("--cams", nargs="+", required=True, help="CSV камер в том же порядке, что и видео")
    ap.add_argument("--truth", help="CSV эталонной траектории (необязательно, для оценки ошибки)")
    ap.add_argument("--overlay-cam", type=int, default=5, help="на какой камере рисовать bounding box (1..N), 0 = не рисовать")
    ap.add_argument("--truck-size", nargs=3, type=float, default=[2080, 520, 750], help="длина ширина высота грузовика, см (оценка: модель 668x168x242 см, в сцене масштаб ~3.1)")
    ap.add_argument("--model", default="yolov8n.pt"); ap.add_argument("--cls", nargs="+", type=int, default=[7, 2, 5], help="id классов; для своей модели с одним классом: --cls 0"); ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thr", type=float, default=8.0, help="порог reprojection error, px")
    ap.add_argument("--out", default="out"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300)
    run(ap.parse_args())
