import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# 角度表示関数-通常プロット
def angleTable(angles):
    
    times = [t for t, a in angles]
    angle = [a for t, a in angles]
    
    plt.figure(figsize=(10, 5))
    
    # 有効な角度データとNoneデータを分けて処理
    valid_times = []
    valid_angles = []
    none_times = []
    
    for i, (t, a) in enumerate(zip(times, angle)):
        if a is not None:
            valid_times.append(t)
            valid_angles.append(a)
        else:
            none_times.append(t)
    
    # 有効な角度データを青線でプロット
    if valid_times:
        plt.plot(valid_times, valid_angles, marker='o', linestyle='-')
    
    # Noneの場合は赤丸で表示（y=0の位置）
    if none_times:
        plt.plot(none_times, [0]*len(none_times), marker='o', linestyle='-',color='red')

    maxAngle=max(valid_angles)

    plt.ylim(0, maxAngle)
    plt.xlabel('Time(second)')
    plt.ylabel('Angle')
    plt.title('Angle per Frame(YOLO)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('yolo-roi-angle.png')
    plt.close()

# bbox検出結果表示関数-通常プロット
def detectionResult(confidence):

    times = [t for t, s in confidence]
    scores = [s for t, s in confidence]
    
    plt.figure(figsize=(10, 5))
    
    # 有効なscoreデータとNoneデータを分けて処理
    valid_times = []
    valid_scores = []
    none_times = []
    
    for i, (t, s) in enumerate(zip(times, scores)):
        if s is not None:
            valid_times.append(t)
            valid_scores.append(s)
        else:
            none_times.append(t)
    
    # 有効なscoreデータを青線でプロット
    if valid_times:
        plt.plot(valid_times, valid_scores, marker='o', linestyle='-')
    
    # Noneの場合は赤丸で表示（y=0の位置）
    if none_times:
        plt.plot(none_times, [0]*len(none_times), marker='o', linestyle='-',color='red')

    maxScore=max(valid_scores)

    plt.ylim(0, maxScore)
    plt.xlabel('Time(second)')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Score per Frame(YOLO)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('yolo-roi-bbox.png')
    plt.close()
    
def isParallel(keypoints, threshold=20):

    global confirm

    confirm=0
    
    try:
                
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # 膝→足首ベクトル
        left_leg_vec = left_ankle - left_knee
        right_leg_vec = right_ankle - right_knee
            
        angle = angle_between(left_leg_vec, right_leg_vec)
        
        confirm += 1
        
        return angle < threshold, angle
        
    except (IndexError, TypeError, ValueError) as e:
        print(f"isParallel関数でエラー: {e}")
        return False

# 足のなす角度を計算
def angle_between(v1, v2):

    # 各ベクトルを単位ベクトルに変換
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    # 内積を計算
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

    # 内積→ラジアン→度に変換
    angle = np.arccos(dot) * 180 / np.pi

    return angle

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル（より高精度）

# 動画ファイルを指定
cap = cv2.VideoCapture(r"E:\\ski\\data\\test.mp4")

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")

    exit()

# 動画の情報を取得
fps = cap.get(cv2.CAP_PROP_FPS)

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
angles=[]

notParallel=0

frame_idx=0

#フレーム読み込み開始
while True:
    ret, frame = cap.read()

    if not ret:
        print("動画の再生が終了しました。")
        break

    frame_idx+=1

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

            # ただ1人のみ検出された場合はcurrent_bboxを更新(ただし、それがターゲットとは限らない)
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
                keypoints_raw = results[0].keypoints[0].data.cpu().numpy()
                
                #elif len(keypoints_raw.shape) == 3 and keypoints_raw.shape[1] == 17:
                #(1検出人数, 17キーポイント, 3座標とスコア)
                #検出された1人目のキーポイントを取得
                keypoints = keypoints_raw[0]

                parallel, angle = isParallel(keypoints)
                
                # パラレルが崩れたらnotParallelをカウントしROI領域を青色で塗りつぶす
                if not parallel:

                    notParallel+=1

                    '''
                    # ROI領域に薄い赤色のオーバーレイを適用
                    roi_overlay = annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                    roi_overlay[:, :, 0] = 0    # B (青を0にする)
                    roi_overlay[:, :, 1] = 0    # G (緑を0にする)  
                    roi_overlay[:, :, 2] = 255  # R (赤を最大にする)
                    # 元のフレームと赤色オーバーレイをブレンド（透明度0.3で薄い赤色）
                    annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                        annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2], 0.7, roi_overlay, 0.3, 0)
                    '''

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

        score=None
        angle=None
        
    else:

        score = float(results[0].boxes.conf[0].cpu().numpy())

    #print(f'角度{angle}')
          
    confidence.append((time, score))
    angles.append((time, angle))

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    #out.write(annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

'''
# 統計処理

# bbox検出成功のフレーム数
total_frames = frame_idx
exit_score_frames = [score for t, score in confidence]
num_exit_score_frames = len(exit_score_frames)
exit_score_percent = (num_exit_score_frames / total_frames) * 100

# bboxを検出した際の平均信頼度スコア
positive_scores = [score for t, score in confidence]
avg_positive_score = np.mean(positive_scores)

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

# 信頼度スコア配列、角度配列に含まれるNoneの個数を計上
failOfConf = sum(1 for t, score in confidence if score is None)
failOfAngle = sum(1 for t, angle in angles if angle is None)

print('YOLO')
print(f'総フレーム数: {total_frames}')
print()
print(f'bbox検出成功のフレーム数: {num_exit_score_frames} ({exit_score_percent:.2f}%)')
print()
print(f'bboxを検出した際の平均信頼度スコア: {avg_positive_score:.4f}')
print()
print(f'bbox検出無しが0.5秒以上続いた区間の個数: {zero_streaks}')
print()
print(f'bbox検出失敗(2人以上、あるいは検出無し)のフレーム数: {failOfConf}')
print()
print(f'足のなす角度検出失敗(2人以上、あるいは検出無し)のフレーム数: {failOfAngle}')

detectionResult(confidence)
angleTable(angles)
'''

#print(f'パラレルが崩れた回数 : {notParallel}')
#print(f'角度計算はしてるのか : {confirm}')
cap.release()