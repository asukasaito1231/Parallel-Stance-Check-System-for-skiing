import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

confirm=0

# bbox検出結果表示関数
def detectionResult(confidence):
    
    '''
    # 通常プロット
    times = [t for t, s in confidence]
    scores = [s for t, s in confidence]


    # 通常プロット
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker='o', linestyle='-')
    plt.ylim(0, 1)
    plt.xlabel('time(second)')
    plt.ylabel('confidence score')
    plt.title('Confidence Score per Frame(YOLO)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('yolo-roi-result-usual.png')
    plt.close()
    

    # 時間軸反転プロット
    max_time = max(times)
    reversed_times = [max_time - t for t in times]
    plt.figure(figsize=(10, 5))
    plt.plot(reversed_times, scores, marker='o', linestyle='-')
    plt.ylim(0, 1)
    plt.xlabel('time(second)')
    plt.ylabel('confidence score')
    plt.title('Confidence Score per Frame (YOLO)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('YOLO-roi-result-reversed.png')
    plt.close()
    '''

def isParallel(keypoints, angle_threshold=35):
    global confirm
    
    try:
        # キーポイントの存在確認（YOLOv8は17個のキーポイント）
        if len(keypoints) < 17:
            print(f'キーポイントが不足: {len(keypoints)}個（期待値: 17個）')
            return False
            
        # 必要なキーポイントの存在確認
        required_indices = [13, 14, 15, 16]  # 左膝、右膝、左足首、右足首
        for idx in required_indices:
            if idx >= len(keypoints):
                print(f'キーポイント{idx}が存在しません')
                return False
            if np.any(np.isnan(keypoints[idx][:2])):
                print(f'キーポイント{idx}の座標が無効です')
                return False
                
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # 膝→足首ベクトル
        left_leg_vec = left_ankle - left_knee
        right_leg_vec = right_ankle - right_knee
        
        # ベクトルの長さ確認
        if (np.linalg.norm(left_leg_vec) < 1e-8 or 
            np.linalg.norm(right_leg_vec) < 1e-8):
            print('ベクトルの長さが0')
            return False
            
        angle = angle_between(left_leg_vec, right_leg_vec)
        confirm += 1
        
        return angle < angle_threshold
        
    except (IndexError, TypeError, ValueError) as e:
        print(f"isParallel関数でエラー: {e}")
        return False

# ベクトルのなす角度を計算
def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot) * 180 / np.pi
    return angle

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル（より高精度）

# 動画ファイルを指定
#cap = cv2.VideoCapture(r"E:\ski\data\parallel-judge.mp4")
cap = cv2.VideoCapture(r"E:\ski\data\expand-reversed.mp4")

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")

    exit()

# 動画の情報を取得
fps = cap.get(cv2.CAP_PROP_FPS)
'''
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r"E:\ski\data\previous.mp4", fourcc, fps, (width, height))

# 逆再生動画を保存するための設定
out = cv2.VideoWriter(r"E:\\ski\\data\\reversed.mp4", fourcc, fps, (width, height))

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

print('逆再生処理完了')

# リソースを解放
cap.release()
out.release()

# 逆再生動画で動画キャプチャを初期化
cap = cv2.VideoCapture(r"E:\\ski\\data\\reversed.mp4")

if not cap.isOpened():
    print("Error: 逆再生動画を開けませんでした。")
    exit()
'''
# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 最初のフレームで検出された人物の位置を保存
first_person_bbox = None
# 最初のフレームで検出された人物のキーポイントを保存
first_person_keypoints = None
# 現在のバウンディングボックスを保存
current_bbox = None

# 最初のフレームを読み込んで人物を検出
ret, first_frame = cap.read()

# フレームの高さと幅を取得
height, width, _ = first_frame.shape

black_frame = np.zeros((height, width, 3), dtype=np.uint8)

if ret:
    try:
        # YOLOでポーズ推定を実行
        first_results = model(first_frame)

        if len(first_results[0].boxes) > 0:

            # バウンディングボックスの座標を保存
            first_person_bbox = first_results[0].boxes[0].xyxy[0].cpu().numpy()

            # キーポイントの座標を保存
            first_person_keypoints = first_results[0].keypoints[0].data.cpu().numpy()

            # 現在のバウンディングボックスを設定
            current_bbox = first_person_bbox

            print("最初のフレームでの人物発見成功")

        else:
            print("最初のフレームで人物が検出されませんでした")
            exit()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        exit()

# 動画の最初に戻る
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

confidence=[]

notParallel=0

#フレーム読み込み開始
while True:
    ret, frame = cap.read()

    if not ret:
        print("動画の再生が終了しました。")
        break

    try:
        if current_bbox is not None:

            # バウンディングボックスの中心座標を計算
            center_x = int((current_bbox[0] + current_bbox[2]) / 2)
            center_y = int((current_bbox[1] + current_bbox[3]) / 2)

            # current_bboxの幅と高さを計算
            bbox_width = current_bbox[2] - current_bbox[0]
            bbox_height = current_bbox[3] - current_bbox[1]

            #多少余白を持たせることで確実にターゲットを検出 
            x_margin=bbox_width/2
            y_margin=bbox_height/5

            # ROIの範囲をcenter_x, center_yを中心にbboxより少し大きい大きさで設定
            roi_x1 = max(0, int(center_x - bbox_width / 2-x_margin))
            roi_y1 = max(0, int(center_y - bbox_height / 2-y_margin))
            roi_x2 = min(width, int(center_x + bbox_width / 2+x_margin))
            roi_y2 = min(height, int(center_y + bbox_height / 2+y_margin))

            roi=frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # ROI領域でYOLOを実行
            results = model(roi)

            # 2人以上あるいは検出無しだった場合はcurrent_bboxはそのまま
            if len(results[0].boxes) < 1 or len(results[0].boxes) > 1:
                curret_bbox = current_bbox

            # ただ1人のみ検出された場合はcurrent_bboxを更新
            else:
                detected_bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
                #ROI内での相対座標 " (0, 0)=ROIの左上 " を元画像での絶対座標に変換 "(0, 0)が画像の左上 "
                current_bbox = [
                    detected_bbox[0]+roi_x1,
                    detected_bbox[1]+roi_y1,
                    detected_bbox[2]+roi_x1,
                    detected_bbox[3]+roi_y1
                    ]

                annotated_frame = results[0].plot()

                # キーポイント取得
                keypoints = results[0].keypoints[0].data.cpu().numpy()

                print(f'検出されたキーポイント数: {len(keypoints)}')

                parallel = isParallel(keypoints)
                
                # パラレルが崩れたらnotParallelをカウントしROI領域を青色で塗りつぶす
                if parallel==False:

                    notParallel+=1

                    # ROI領域のみ青色で塗りつぶす
                    #annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2, 0] = 255  # B
                    #annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2, 1] = 0    # G
                    #annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2, 2] = 0    # R

                # ROI以外を黒く塗りつぶす
                annotated_frame = cv2.copyMakeBorder(
                    annotated_frame,
                    roi_y1, height - roi_y2,
                    roi_x1, width - roi_x2,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                    )

        # current_bboxがnoneの場合
        else:
          # YOLOでフレーム全体サイズのままポーズ推定を実行
          results = model(frame)
          annotated_frame = results[0].plot()

    except Exception as e:
        print('try文に突入しなかった')
        results = model(frame)
        annotated_frame = results[0].plot()

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    time = frame_number / fps

    if len(results[0].boxes) < 1 or len(results[0].boxes) > 1:
        score=0

    else:
        score = float(results[0].boxes.conf[0].cpu().numpy())

    confidence.append((time, score))

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    #out.write(annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

'''
# --- 追加統計処理 ---

# bbox検出成功のフレーム数
total_frames = len(confidence)
exit_score_frames = [score for t, score in confidence if score > 0.0]
num_exit_score_frames = len(exit_score_frames)
exit_score_percent = (num_exit_score_frames / total_frames) * 100 if total_frames > 0 else 0.0

# bbox検出無しが0.5秒以上続いた区間の個数
zero_streaks = 0
streak_length = 0
judge=fps/2

for t, score in confidence:
    if score == 0.0:
        streak_length += 1
    else:
        if streak_length >= judge:  # 1秒以上
            zero_streaks += 1
        streak_length = 0

# 最後が0で終わる場合
if streak_length >= judge:
    zero_streaks += 1

# bboxを検出した際の平均信頼度スコア
positive_scores = [score for t, score in confidence if score > 0.0]
avg_positive_score = np.mean(positive_scores) if positive_scores else 0.0

print(f'総フレーム数: {total_frames}')
print(f'bbox検出成功のフレーム数: {num_exit_score_frames} ({exit_score_percent:.2f}%)')
print(f'bbox検出無しが0.5秒以上続いた区間の個数: {zero_streaks}')
print(f'bboxを検出した際の平均信頼度スコア: {avg_positive_score:.4f}')
#print(f'パラレルが崩れた回数 : {notParallel}')

detectionResult(confidence)
'''

print(f'パラレルが崩れた回数 : {notParallel}')
print(f'角度計算はしてるのか : {confirm}')
cap.release()
#out.release()