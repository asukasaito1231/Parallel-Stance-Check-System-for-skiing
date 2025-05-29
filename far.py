import cv2
from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル

# 動画キャプチャの初期化
cap = cv2.VideoCapture("far1.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# トラックバーのコールバック関数
def nothing(x):
    pass

# トラックバーの作成（初期値100、最大値200）
cv2.createTrackbar('Zoom', 'Pose Detection', 100, 200, nothing)

# 動画保存の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output-standard.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # トラックバーからズーム値を取得（100を基準に0.5から2.0の範囲）
    zoom_value = cv2.getTrackbarPos('Zoom', 'Pose Detection') / 100.0
    
    # デバッグ用：ズーム値を表示
    print(f"現在のズーム値: {zoom_value:.2f}")

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # ズーム値に基づいて新しいサイズを計算
    new_width = int(width * zoom_value)
    new_height = int(height * zoom_value)

    # フレームサイズをズーム値に基づいて調整
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # YOLOでポーズ推定を実行
    results = model(resized_frame)

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