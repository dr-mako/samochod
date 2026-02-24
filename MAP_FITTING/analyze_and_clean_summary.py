import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

summary_path = r"C:\Users\Maciej Kozłowski\Desktop\Logi\2026-02-23-2\out_lane_aruco_bev\summary.csv"
out_path = summary_path.replace("summary.csv", "summary_clean.csv")

jump_threshold = 100  # px


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(summary_path)

print("Liczba klatek:", len(df))

# ============================================================
# COMPUTE JUMPS
# ============================================================

dx = df["marker_bev_x"].diff()
dy = df["marker_bev_y"].diff()

jump_mask = (
    (np.abs(dx) > jump_threshold) |
    (np.abs(dy) > jump_threshold)
)

print("Wykryte duże skoki:", jump_mask.sum())

# ============================================================
# MARK OUTLIERS
# ============================================================

df["is_outlier"] = jump_mask

# Wyzeruj landmark w outlierach
df.loc[jump_mask, ["marker_bev_x", "marker_bev_y"]] = np.nan
df.loc[jump_mask, ["raw_id", "smoothed_id"]] = np.nan

# ============================================================
# SAVE CLEAN VERSION
# ============================================================

df.to_csv(out_path, index=False)

print("Zapisano:", out_path)
print("Gotowe.")