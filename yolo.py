import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル

# 元の動画を読み込み
cap = cv2.VideoCapture("far1.mp4")
if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# 動画の情報を取得
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 逆再生動画を保存するための設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('reversed.mp4', fourcc, fps, (width, height))

# フレームを配列に保存
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# フレームを逆順に保存
for frame in reversed(frames):
    out.write(frame)

# リソースを解放
cap.release()
out.release()

# 逆再生動画で動画キャプチャを初期化
cap = cv2.VideoCapture("reversed.mp4")

if not cap.isOpened():
    print("Error: 逆再生動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するscale
resize_scale = 1

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 出力動画の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_yolo.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

    # YOLOでポーズ推定を実行
    results = model(small_frame)

    # 検出結果を描画
    annotated_frame = results[0].plot()

    # フレームを保存
    # out.write(annotated_frame)

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()