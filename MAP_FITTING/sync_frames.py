# sync_frames.py
# synchronizacja zdarzeń z logów do osi czasu klatek
import json
import numpy as np
import pandas as pd
from pathlib import Path


in_dir_main = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-2")
#in_dir_main = Path(r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-19")
OUT_PATH = in_dir_main / "out.txt"
CAMERA_PATH = in_dir_main / "camera.txt"
OUT_CSV = in_dir_main / "synced_raw.csv"

# "zoh" albo "linear"
MODE = "zoh"


def load_camera(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", header=None, names=["frame_id", "time"])
    df["frame_id"] = df["frame_id"].astype(int)
    df["time"] = df["time"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def load_out_events(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    motor_rows = []
    servo_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if "time" not in rec:
                continue
            try:
                t = float(rec["time"])
            except Exception:
                continue

            typ = str(rec.get("type", "")).upper()
            d = rec.get("data", {}) or {}

            if typ == "MOTOR":
                mid = d.get("id", None)
                if mid is None:
                    continue
                try:
                    mid = int(mid)
                except Exception:
                    continue

                motor_rows.append(
                    {
                        "time": t,
                        "id": mid,
                        "spd": float(d["spd"]) if d.get("spd") is not None else np.nan,
                    }
                )

            elif typ == "SERVO":
                servo_rows.append(
                    {
                        "time": t,
                        "servo_1": float(d["servo_1"])
                        if d.get("servo_1") is not None
                        else np.nan,
                        "servo_2": float(d["servo_2"])
                        if d.get("servo_2") is not None
                        else np.nan,
                    }
                )

    motors = (
        pd.DataFrame(motor_rows)
        .dropna(subset=["time", "id"])
        .sort_values(["id", "time"])
        .reset_index(drop=True)
    )
    servos = (
        pd.DataFrame(servo_rows)
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return motors, servos


def zoh_sample(tq: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t, tq, side="right") - 1
    out = np.full_like(tq, np.nan, dtype=float)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


def linear_sample(tq: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.interp(tq, t, v, left=np.nan, right=np.nan)


def sample_to_frames(t_frame: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return np.full_like(t_frame, np.nan, dtype=float)
    if MODE == "linear":
        if len(t) < 2:
            return np.full_like(t_frame, np.nan, dtype=float)
        return linear_sample(t_frame, t, v)
    return zoh_sample(t_frame, t, v)


def main():
    cam = load_camera(CAMERA_PATH)
    motors, servos = load_out_events(OUT_PATH)

    t_frame = cam["time"].to_numpy(dtype=float)

    # SERVO (bez offsetu, tylko średnia w stopniach)
    ts = servos["time"].to_numpy(dtype=float)
    s1 = servos["servo_1"].to_numpy(dtype=float)
    s2 = servos["servo_2"].to_numpy(dtype=float)
    servo_1_sync = sample_to_frames(t_frame, ts, s1)
    servo_2_sync = sample_to_frames(t_frame, ts, s2)
    servo_mean_deg_raw = (-servo_1_sync + servo_2_sync) / 2.0

    # MOTORY (średnia jak w Matlabie, ale nadal "raw spd units")
    spd_sync = {}
    for mid in [1, 2, 3, 4]:
        m = motors[(motors["id"] == mid)].dropna(subset=["spd"])
        tm = m["time"].to_numpy(dtype=float)
        vm = m["spd"].to_numpy(dtype=float)
        spd_sync[mid] = sample_to_frames(t_frame, tm, vm)

    spd_mean_raw = (-spd_sync[1] + spd_sync[2] + spd_sync[3] - spd_sync[4]) / 4.0

    out = pd.DataFrame(
        {
            "frame_id": cam["frame_id"].to_numpy(dtype=int),
            "time": t_frame,
            "spd_mean_raw": spd_mean_raw,
            "servo_mean_deg_raw": servo_mean_deg_raw,
        }
    )

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Zapisano: {OUT_CSV} (wiersze={len(out)})")


if __name__ == "__main__":
    main()