import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル

# 動画キャプチャの初期化
cap = cv2.VideoCapture("far.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するscale
resize_scale = 1

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Zoomed Detection', cv2.WINDOW_NORMAL)

# 動画保存の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_yolo.mp4', fourcc, fps, (width, height))

# 拡大倍率
zoom_scale = 2.0

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

    # 検出された人物のバウンディングボックスを取得
    if len(results[0].boxes) > 0:
        # 最初に検出された人物のバウンディングボックスを取得
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # バウンディングボックスの中心を計算
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 拡大する領域のサイズを計算
        box_width = x2 - x1
        box_height = y2 - y1
        
        # 拡大する領域の座標を計算（画像の端を考慮）
        zoom_x1 = max(0, center_x - int(box_width * zoom_scale / 2))
        zoom_y1 = max(0, center_y - int(box_height * zoom_scale / 2))
        zoom_x2 = min(width, center_x + int(box_width * zoom_scale / 2))
        zoom_y2 = min(height, center_y + int(box_height * zoom_scale / 2))
        
        # 拡大領域を切り出し
        zoomed_region = small_frame[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
        
        # 拡大表示
        if zoomed_region.size > 0:  # 領域が有効な場合のみ表示
            cv2.imshow('Zoomed Detection', zoomed_region)

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
