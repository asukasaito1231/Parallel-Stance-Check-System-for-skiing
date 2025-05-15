import cv2
from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO('yolov8m-pose.pt')  # より精度の高いモデルに変更

# 動画キャプチャの初期化
cap = cv2.VideoCapture("test.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するscale
resize_scale = 1

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 動画保存の設定
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

    # フレームを拡大（遠くの人物の検出精度向上のため）
    scale_factor = 1.0  # 拡大率
    enlarged_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

    # YOLOでポーズ推定を実行
    results = model(enlarged_frame)

    # 検出結果を描画
    annotated_frame = results[0].plot()

    # 元のサイズに戻す
    annotated_frame = cv2.resize(annotated_frame, (width, height))

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
