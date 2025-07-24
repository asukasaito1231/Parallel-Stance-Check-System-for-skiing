import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def detectionResult(confidence):
    
    times = [t for t, s in confidence]
    scores = [s for t, s in confidence]
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker='o', linestyle='-')
    plt.ylim(0, 1)
    plt.xlabel('time(second)')
    plt.ylabel('confidence score')
    plt.title('Confidence Score per Frame')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('yolo-result.png')
    plt.close()

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル

# 動画キャプチャの初期化
cap = cv2.VideoCapture(r"E:\ski\data\clip.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# 動画保存の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('standard.mp4', fourcc, fps, (width, height))

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

confidence=[]

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width), int(height)))

    # YOLOでポーズ推定を実行
    results = model(small_frame)

    # 検出結果を描画
    annotated_frame = results[0].plot()

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    time = frame_number / fps

    if len(results[0].boxes) < 1 or len(results[0].boxes) > 1:
        score=0
        
    else:
        score = float(results[0].boxes.conf[0].cpu().numpy())

    confidence.append((time, score))

    # フレームを保存
    #out.write(annotated_frame)

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

detectionResult(confidence)

# --- 追加統計処理 ---
total_frames = len(confidence)
zero_score_frames = [score for t, score in confidence if score == 0.0]
num_zero_score_frames = len(zero_score_frames)
zero_score_percent = (num_zero_score_frames / total_frames) * 100 if total_frames > 0 else 0.0

# 信頼度スコア0が1秒以上続いた区間の個数をカウント
zero_streaks = 0
streak_length = 0
for t, score in confidence:
    if score == 0.0:
        streak_length += 1
    else:
        if streak_length >= fps:  # 1秒以上
            zero_streaks += 1
        streak_length = 0

# 最後が0で終わる場合
if streak_length >= fps:
    zero_streaks += 1

# 信頼度スコアが0より大きいフレームの平均信頼度スコア
positive_scores = [score for t, score in confidence if score > 0.0]
avg_positive_score = np.mean(positive_scores) if positive_scores else 0.0

print(f'総フレーム数: {total_frames}')
print(f'bbox検出無しのフレーム数: {num_zero_score_frames} ({zero_score_percent:.2f}%)')
print(f'bbox検出無しが1秒以上続いた区間の個数: {zero_streaks}')
print(f'bboxを検出した際の平均信頼度スコア: {avg_positive_score:.4f}')

# リソースを解放
cap.release()
#out.release()
cv2.destroyAllWindows()