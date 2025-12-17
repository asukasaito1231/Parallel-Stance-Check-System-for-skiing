import os
import numpy as np
import matplotlib.pyplot as plt

# ===== フォルダパス =====
angle_dir = "/run/media/s1300109/1B93-9EF4/DCIM/frame-angle"
label_dir = "/run/media/s1300109/1B93-9EF4/DCIM/frame-noAngle"

MAX_ANGLE = 25
MAX_FRAME = 494

# ===== 度数配列 =====
parallel_count = np.zeros(MAX_ANGLE + 1, dtype=int)
not_parallel_count = np.zeros(MAX_ANGLE + 1, dtype=int)

# ===== メインループ =====
for i in range(1, MAX_FRAME + 1):

    # ---------- 角度ファイル探索 ----------
    angle_filename = None
    prefix = f"フレーム番号{i}-"

    for f in os.listdir(angle_dir):
        if f.startswith(prefix) and f.endswith("度.png"):
            angle_filename = f
            break

    if angle_filename is None:
        continue

    # ---------- 角度抽出 ----------
    try:
        angle = int(
            angle_filename
            .replace(prefix, "")
            .replace("度.png", "")
        )
    except ValueError:
        continue

    if angle > MAX_ANGLE:
        continue

    # ---------- parallel / notParallel 判定 ----------
    if os.path.exists(
        os.path.join(label_dir, f"フレーム番号{i}-parallel.png")
    ):
        parallel_count[angle] += 1

    elif os.path.exists(
        os.path.join(label_dir, f"フレーム番号{i}-notParallel.png")
    ):
        not_parallel_count[angle] += 1


# ===== ヒストグラム描画 =====
angles = np.arange(0, MAX_ANGLE + 1)

plt.figure(figsize=(10, 6))
plt.bar(angles, parallel_count, label="parallel")
plt.bar(
    angles,
    not_parallel_count,
    bottom=parallel_count,
    label="notParallel"
)

plt.xlabel("Shin Angle (degrees)")
plt.ylabel("Number of Occurrences")
plt.title("Distribution of Parallel and Not Parallel by Shin Angle")
plt.legend()
plt.show()
