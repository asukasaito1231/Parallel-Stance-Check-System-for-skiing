import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル

# 動画キャプチャの初期化
cap = cv2.VideoCapture("short.mp4")  # 動画ファイルを指定

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

# ウィンドウサイズを変更するscale
resize_scale = 1
# 拡大倍率
scale = 1

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# トラックバーのコールバック関数
def nothing(x):
    pass

# トラックバーの作成
cv2.createTrackbar('Scale', 'Pose Detection', 1, 5, nothing)

# 動画保存の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('expand.mp4', fourcc, fps, (width, height))

# 最初の人物の位置を保存する変数
first_person_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # トラックバーから現在のスケール値を取得
    scale = cv2.getTrackbarPos('Scale', 'Pose Detection')

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # 拡大処理
    # 縮小処理を防ぐ
    if int(scale) < 1:
        scale = 1

    # トリミング領域の幅と高さを計算
    roi_width = width / int(scale)
    roi_height = height / int(scale)

    # トリミング開始位置と終了位置を決める
    sabun_w1 = int((width - roi_width)/2)
    sabun_w2 = int((width + roi_width)/2)
    sabun_h1 = int((height - roi_height)/2)
    sabun_h2 = int((height + roi_height)/2)

    # フレームの中心領域をトリミング
    frame = frame[sabun_h1:sabun_h2, sabun_w1:sabun_w2]

    # トリミングしたフレームを元のサイズに拡大
    frame = cv2.resize(frame, (width, height))

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

    # YOLOでポーズ推定を実行
    results = model(small_frame)

    # 最初の人物の位置を保存
    if first_person_bbox is None and len(results[0].boxes) > 0:
        first_person_bbox = results[0].boxes[0].xyxy[0].cpu().numpy()

    # 検出結果を描画（最初の人物のみ）
    if first_person_bbox is not None:
        # 最初の人物の検出結果のみを保持
        results[0].boxes = results[0].boxes[0:1]
        results[0].keypoints = results[0].keypoints[0:1]
        annotated_frame = results[0].plot()
    else:
        annotated_frame = small_frame.copy()

    # フレームを保存
    out.write(annotated_frame)

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()