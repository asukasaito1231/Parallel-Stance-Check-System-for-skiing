import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# 角度グラフ表示関数-時間軸反転プロット
def angleGraph(angles):
    
    sub_times = [t for t, a in angles]
    sub_angle = [a for t, a in angles]

    minTime=min(sub_times)
    maxTime=max(sub_times)

    re = [((minTime + maxTime) - t, a) for t, a in zip(sub_times, sub_angle)]

    times  = [t for t, a in re]
    angle = [a for t, a in re]

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

    maxAngle=max(valid_angles)

    # 有効な角度データを青線でプロット
    if valid_times:
        plt.plot(valid_times, valid_angles, marker='o', linestyle='-', color='blue')
    
    # Noneの場合は赤丸で表示（y=0の位置）
    if none_times:
        plt.plot(none_times, [maxAngle]*len(none_times), marker='x', linestyle='-',color='red')
    print(none_times)

    plt.ylim(0, maxAngle)
    plt.xlabel('Time(second)', fontsize=21)
    plt.ylabel('Angle(degree)', fontsize=21)

    # 軸の数字のフォントサイズを大きくする
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # 横軸（時間）を1秒刻みに設定
    if times:
        max_time = max(times)
        tick_positions = np.arange(0, max_time, 1)
        plt.xticks(tick_positions)

    # 縦軸（角度）を5度刻みに設定
    if angle:
        tick_positions = np.arange(0, maxAngle, 5)
        plt.yticks(tick_positions)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('angle-roi-rewind.png')
    plt.close()

# bbox検出結果グラフ表示関数-時間軸反転プロット
def scoreGraph(confidence):

    sub_times = [t for t, s in confidence]
    sub_scores = [s for t, s in confidence]

    minTime=min(sub_times)
    maxTime=max(sub_times)

    re = [((minTime + maxTime) - t, s) for t, s in zip(sub_times, sub_scores)]

    times  = [t for t, s in re]
    scores = [s for t, s in re]

    plt.figure(figsize=(10, 5))
    
    # 有効な角度データとNoneデータを分けて処理
    valid_times = []
    valid_scores = []
    none_times = []
    
    for i, (t, s) in enumerate(zip(times, scores)):

        if s is not None:
            valid_times.append(t)
            valid_scores.append(s)

        else:
            none_times.append(t)

    # 有効な角度データを青線でプロット
    if valid_times:
        plt.plot(valid_times, valid_scores, marker='o', linestyle='-', color='blue')
    
    # Noneの場合は赤丸で表示（y=0の位置）
    if none_times:
        plt.plot(none_times, [0]*len(none_times), marker='o', linestyle='-',color='red')

    plt.ylim(0, 1)
    plt.xlabel('Time(second)', fontsize=21)
    plt.ylabel('Confidence Score', fontsize=21)

    # 軸の数字のフォントサイズを大きくする
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    # 横軸（時間）を1秒刻みに設定
    if times:
        max_time = max(times)
        tick_positions = np.arange(0, max_time, 1)
        plt.xticks(tick_positions)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('confidence-roi-rewind.png')
    plt.close()

def isParallel(keypoints, threshold=20):
    
    parallel=True

    try:
        # 腰の座標
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        
        # 膝の座標
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        
        # 足首の座標
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # 太ももベクトル（腰→膝）
        left_thigh_vec = left_knee - left_hip
        right_thigh_vec = right_knee - right_hip
        
        # 脛ベクトル（膝→足首）
        left_shin_vec = left_ankle - left_knee
        right_shin_vec = right_ankle - right_knee
        
        # 太ももの角度計算
        thigh_angle = angle_between(left_thigh_vec, right_thigh_vec)
        
        # 脛の角度計算
        shin_angle = angle_between(left_shin_vec, right_shin_vec)
        
        angle=(thigh_angle+shin_angle)/2

        if(angle > threshold):
            parallel=False

        return angle,parallel
        
    except (IndexError, TypeError, ValueError) as e:
        print(f"isParallel関数でエラー: {e}")
        return False, (None, None)

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
object_detection_model = YOLO('yolo11n.pt')
detect_skeleton_model=YOLO('yolo11n-pose.pt')

cap = cv2.VideoCapture(r"E:\\ski\\far\\far1.mp4")

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_length=total_frames/fps

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(r"E:\\ski\\1級検定本番\\ポスター用.mp4", fourcc, fps, (width, height))

# 逆再生動画を保存するための設定
out = cv2.VideoWriter(r"E:\\ski\\far\\far1-reversed.mp4", fourcc, fps, (width, height))

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

cap = cv2.VideoCapture(r"E:\\ski\\far\\far1-reversed.mp4")

if not cap.isOpened():
    print("Error: 逆再生動画を開けませんでした。")
    exit()

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 現在のバウンディングボックスを保存
current_bbox = None

# 最初のフレームを読み込んで人物を検出
ret, first_frame = cap.read()

# フレームの高さと幅を取得
height, width, _ = first_frame.shape

if ret:
    try:
        # YOLOでポーズ推定を実行（人間クラスのみ検出）
        first_results = object_detection_model(first_frame, classes=[0])

        if len(first_results[0].boxes)==1:

            # バウンディングボックスの座標を保存
            current_bbox = first_results[0].boxes[0].xyxy[0].cpu().numpy()
            
            print("最初のフレームでの人物発見成功")

        else:
            print("最初のフレームで人物が検出されませんでした(2人以上の検出、あるいは検出無し)")
            exit()

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        exit()

# 動画の最初に戻る
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

confidence=[]

angles=[]

total_frames=0

#フレーム読み込み開始
while True:

    ret, frame = cap.read()

    if not ret:
        print("動画の再生が終了しました。")
        print()
        break

    total_frames+=1

    time = total_frames / fps

    try:
        if current_bbox is not None:

            # バウンディングボックスの中心座標を計算
            center_x = int((current_bbox[0] + current_bbox[2]) / 2)
            center_y = int((current_bbox[1] + current_bbox[3]) / 2)

            # current_bboxの幅と高さを計算
            bbox_width = current_bbox[2] - current_bbox[0]
            bbox_height = current_bbox[3] - current_bbox[1]

            #多少余白を持たせることで確実にターゲットを検出 
            x_margin=bbox_width
            y_margin=bbox_height

            # ROIの範囲をcenter_x, center_yを中心にbboxより少し大きい大きさで設定
            bbox_roi_x1 = max(0, int(center_x - bbox_width / 2-x_margin))
            bbox_roi_y1 = max(0, int(center_y - bbox_height / 2-y_margin))
            bbox_roi_x2 = min(width, int(center_x + bbox_width / 2+x_margin))
            bbox_roi_y2 = min(height, int(center_y + bbox_height / 2+y_margin))

            bbox_roi=frame[bbox_roi_y1:bbox_roi_y2, bbox_roi_x1:bbox_roi_x2]

            # ROI領域でスキーヤー検出を実行
            bbox_results = object_detection_model(bbox_roi, classes=[0])

            # 2人以上あるいは検出無しだった場合はcurrent_bboxはそのまま
            if len(bbox_results[0].boxes) < 1 or len(bbox_results[0].boxes) > 1:

                curret_bbox = current_bbox

                # ROI以外を黒く塗りつぶした画像をannotated_frameに格納
                annotated_frame = cv2.copyMakeBorder(
                    bbox_roi,
                    bbox_roi_y1, height - bbox_roi_y2,
                    bbox_roi_x1, width - bbox_roi_x2,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                score=None

                angle=None

            #ただ1人のみ検出された場合はcurrent_bboxを更新(ただし、それがターゲットとは限らない)
            #bboxと骨格を描画 
            else:

                detected_bbox = bbox_results[0].boxes[0].xyxy[0].cpu().numpy()

                #ROI内での相対座標 " (0, 0)=ROIの左上 " を全体フレームでの絶対座標に変換 "(0, 0)が全体フレームの左上 "
                current_bbox = [
                    detected_bbox[0]+bbox_roi_x1,
                    detected_bbox[1]+bbox_roi_y1,
                    detected_bbox[2]+bbox_roi_x1,
                    detected_bbox[3]+bbox_roi_y1
                    ]

                #検出されたbboxの座標を取得
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = current_bbox

                skeleton_margin = 200

                if(time > (video_length/2)):
                    skeleton_margin=100

                skeleton_roi_x1 = max(0, int(bbox_x1 - (skeleton_margin/2)))
                skeleton_roi_y1 = max(0, int(bbox_y1 - skeleton_margin))
                skeleton_roi_x2 = min(width, int(bbox_x2 + (skeleton_margin/2)))
                skeleton_roi_y2 = min(height, int(bbox_y2 + skeleton_margin))

                # 骨格検出用のROI領域を抽出
                skeleton_roi = frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2]

                skeleton_results=detect_skeleton_model(skeleton_roi, classes=[0])

                #annotated_frame = bbox_results[0].plot()
                annotated_frame = skeleton_results[0].plot()

                # キーポイント取得
                keypoints_raw = skeleton_results[0].keypoints[0].data.cpu().numpy()
                
                #elif len(keypoints_raw.shape) == 3 and keypoints_raw.shape[1] == 17:
                #(1検出人数, 17キーポイント, 3座標とスコア)

                #検出された1人目のキーポイントを取得
                keypoints = keypoints_raw[0]

                score = float(bbox_results[0].boxes.conf[0].cpu().numpy())

                angle, parallel = isParallel(keypoints)

                if parallel == False:
                    # フレームを薄い赤で色付けする
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, np.full(annotated_frame.shape, (0, 0, 255), dtype=np.uint8), 0.3, 0)

                # ROI以外を黒く塗りつぶす
                annotated_frame = cv2.copyMakeBorder(
                    annotated_frame,
                    skeleton_roi_y1, height - skeleton_roi_y2,
                    skeleton_roi_x1, width - skeleton_roi_x2,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                    )

    except Exception as e:
        print('try文に突入しなかった')
        bbox_results = object_detection_model(frame, classes=[0])
        annotated_frame = bbox_results[0].plot()
        score=None
        angle=None

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
exit_score_frames = [score for t, score in confidence if score is not None]
num_exit_score_frames = len(exit_score_frames)
if total_frames > 0:
    exit_score_percent = (num_exit_score_frames / total_frames) * 100
else:
    exit_score_percent = 0.0

# bboxを検出した際の平均信頼度スコア
positive_scores = [score for t, score in confidence if score is not None]
if positive_scores:
    avg_positive_score = np.mean(positive_scores)
else:
    avg_positive_score = 0.0

# bbox検出無しが0.5秒以上続いた区間の個数
zero_streaks = 0
streak_length = 0
judge=fps/2

for t, score in confidence:
    if score is None:
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

print('【YOLO】 - far26.mp4')
print()
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
#print(f'足のなす角度検出失敗(2人以上、あるいは検出無し)のフレーム数: {failOfAngle}')
'''
scoreGraph(confidence)
angleGraph(angles)

cap.release()
out.release()