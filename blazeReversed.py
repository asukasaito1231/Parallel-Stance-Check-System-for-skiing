import cv2
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
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
    plt.savefig('blaze-angle-roi-rewind.png')
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
    plt.savefig('blaze-confidence-roi-rewind.png')
    plt.close()

def isParallel(keypoints, threshold=20):
    
    parallel=True

    try:

        # MediaPipeのLandmarkオブジェクトから3D座標を取得
        left_hip = np.array([keypoints[11].x, keypoints[11].y, keypoints[11].z])
        right_hip = np.array([keypoints[12].x, keypoints[12].y, keypoints[12].z])

        left_knee = np.array([keypoints[13].x, keypoints[13].y, keypoints[13].z])
        right_knee = np.array([keypoints[14].x, keypoints[14].y, keypoints[14].z])

        left_ankle = np.array([keypoints[15].x, keypoints[15].y, keypoints[15].z])
        right_ankle = np.array([keypoints[16].x, keypoints[16].y, keypoints[16].z])
        
        # 太ももベクトル（腰→膝）
        left_thigh_vec = left_knee - left_hip
        right_thigh_vec = right_knee - right_hip
        
        # 脛ベクトル（膝→足首）
        left_shin_vec = left_ankle - left_knee
        right_shin_vec = right_ankle - right_knee
        
        # 太ももの角度計算（3D対応）
        thigh_angle = angle_between(left_thigh_vec, right_thigh_vec)
        if thigh_angle > 30:
            return 50, False
        
        # 脛の角度計算（3D対応）
        shin_angle = angle_between(left_shin_vec, right_shin_vec)
        if shin_angle > 30:
            return 50, False

        # 左右の足の平行度を測定（太ももと脛の平均）
        angle = (thigh_angle + shin_angle) / 2
        
        # デバッグ: 角度情報を表示
        print(f"BlazePose - 太もも角度: {thigh_angle:.2f}度, 脛角度: {shin_angle:.2f}度, 平均角度: {angle:.2f}度")

        if(angle > threshold):
            parallel=False

        return angle,parallel
        
    except (IndexError, TypeError, ValueError) as e:
        print(f"isParallel関数でエラー: {e}")
        return False, (None, None)

# 3Dベクトルのなす角度を計算
def angle_between(v1, v2):
    """
    3Dベクトル間の角度を計算する関数
    
    Args:
        v1: 3Dベクトル1 [x, y, z]
        v2: 3Dベクトル2 [x, y, z]
    
    Returns:
        angle: 角度（度）
    """
    # 各ベクトルを単位ベクトルに変換
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

    # 内積を計算（3D対応）
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)

    # 内積→ラジアン→度に変換
    angle = np.arccos(dot) * 180 / np.pi

    return angle

# 検出結果の描画
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
 
    return annotated_image

def draw_skeleton(annotated_frame, pose_landmarker_result, roi_frame):
    """
    骨格を描画する関数
    
    Args:
        annotated_frame: 描画対象のフレーム
        pose_landmarker_result: 姿勢推定結果
        roi_frame: ROIフレーム
    
    Returns:
        annotated_frame: 骨格が描画されたフレーム
    """

    # フレームの高さと幅を取得
    height, width, _ = annotated_frame.shape

    if pose_landmarker_result.pose_landmarks:
        for pose_landmarks in pose_landmarker_result.pose_landmarks:

            # ランドマークを描画
            for i, landmark in enumerate(pose_landmarks):
                x = int(landmark.x * roi_frame.shape[1])
                y = int(landmark.y * roi_frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
            
            # 骨格の接続線を描画
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24),
                (23, 25), (25, 27), (27, 29), (29, 31),
                (24, 26), (26, 28), (28, 30), (30, 32),
                (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
                (16, 18), (16, 20), (16, 22), (18, 20), (20, 22)
            ]
            
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start_point = pose_landmarks[start_idx]
                    end_point = pose_landmarks[end_idx]
                    
                    sx = int(start_point.x * roi_frame.shape[1])
                    sy = int(start_point.y * roi_frame.shape[0])
                    ex = int(end_point.x * roi_frame.shape[1])
                    ey = int(end_point.y * roi_frame.shape[0])
                    
                    cv2.line(annotated_frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
    
    return annotated_frame

# YOLOモデルの読み込み
object_detection_model = YOLO('yolo11n.pt')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1)

#BlazePoseモデルの読み込み
detect_skeleton_model=PoseLandmarker.create_from_options(options)

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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                roi_frame = frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2]

                roi_rgb_frame = rgb_frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2]

                # ここでshapeチェック
                if roi_rgb_frame.size == 0:
                    continue

                roi_rgb_frame = np.ascontiguousarray(roi_rgb_frame)

                mp_image = mp.Image(mp.ImageFormat.SRGB, roi_rgb_frame)

                skeleton_results=detect_skeleton_model.detect(mp_image)

                # annotated_frameを初期化（ROIフレームをコピー）
                #annotated_frame = roi_frame.copy()

                annotated_frame = bbox_results[0].plot()

                # 解析結果を描画
                #annotated_frame = draw_skeleton(annotated_frame, skeleton_results, roi_frame)

                # 骨格検出用のROI領域を抽出 
                #skeleton_roi = frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2]
                
                #annotated_frame = bbox_results[0].plot()
                #annotated_frame = skeleton_results[0].plot()

                score = float(bbox_results[0].boxes.conf[0].cpu().numpy())

                angle, parallel = isParallel(skeleton_results.pose_world_landmarks[0])

                #if parallel == False:
                    # フレームを薄い赤で色付けする
                    #annotated_frame = cv2.addWeighted(annotated_frame, 0.7, np.full(annotated_frame.shape, (0, 0, 255), dtype=np.uint8), 0.3, 0)

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
detect_skeleton_model.close()