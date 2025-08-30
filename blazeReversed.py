import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

# 角度表示関数-時間軸反転プロット
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
    plt.title('Angle per Frame(BlazePose)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('blaze-reversed-angle.png')
    plt.close()

# bbox検出結果表示関数-時間軸反転プロット
def scoreGraph(confidence):

    sub_times = [t for t, s in confidence]
    sub_scores = [s for t, s in confidence]

    minTime=min(sub_times)
    maxTime=max(sub_scores)

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
        plt.plot(valid_times, valid_scores, marker='o', linestyle='-')
    
    # Noneの場合は赤丸で表示（y=0の位置）
    if none_times:
        plt.plot(none_times, [0]*len(none_times), marker='o', linestyle='-',color='red')

    maxScore=max(valid_scores)

    plt.ylim(0, maxScore)
    plt.xlabel('Time(second)')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Score per Frame(BlazePose)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('blaze-reversed-bbox.png')
    plt.close()

def isParallel(keypoints, angle_threshold=20):

    global confirm

    confirm=0
    
    try:
        # MediaPipeのLandmarkオブジェクトから座標を取得
        left_knee = np.array([keypoints[13].x, keypoints[13].y])
        right_knee = np.array([keypoints[14].x, keypoints[14].y])
        left_ankle = np.array([keypoints[15].x, keypoints[15].y])
        right_ankle = np.array([keypoints[16].x, keypoints[16].y])
        
        # 膝→足首ベクトル
        left_leg_vec = left_ankle - left_knee
        right_leg_vec = right_ankle - right_knee
            
        angle = angle_between(left_leg_vec, right_leg_vec)
        
        confirm += 1
        
        return angle < angle_threshold, angle
        
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

# 全身のvisibility平均スコアを計算する関数
def calculate_full_body_visibility(landmarks):
    # 全身のランドマークインデックス（0-32の全て）
    full_body_indices = list(range(33))  # 0から32まで

    visibility_scores = []
    for index in full_body_indices:
        if index < len(landmarks):
            visibility_scores.append(landmarks[index].visibility)

    # 平均visibilityを計算
    if visibility_scores:
        return np.mean(visibility_scores)
    else:
        return 0.0
    
def draw_skeleton(annotated_frame, pose_landmarker_result, roi_coords=None, roi_frame=None, width=None, height=None):
    """
    骨格を描画する関数
    
    Args:
        annotated_frame: 描画対象のフレーム
        pose_landmarker_result: 姿勢推定結果
        roi_coords: ROI座標 (x, y)
        roi_frame: ROIフレーム
        width: 元のフレーム幅
        height: 元のフレーム高さ
    
    Returns:
        annotated_frame: 骨格が描画されたフレーム
    """
    if pose_landmarker_result.pose_landmarks:
        for pose_landmarks in pose_landmarker_result.pose_landmarks:
            # 各ランドマークを描画
            for i, landmark in enumerate(pose_landmarks):
                x = int(landmark.x * (roi_frame.shape[1] if roi_coords else width))
                y = int(landmark.y * (roi_frame.shape[0] if roi_coords else height))
                if roi_coords:
                    x += roi_coords[0]
                    y += roi_coords[1]
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
                    
                    sx = int(start_point.x * (roi_frame.shape[1] if roi_coords else width))
                    sy = int(start_point.y * (roi_frame.shape[0] if roi_coords else height))
                    ex = int(end_point.x * (roi_frame.shape[1] if roi_coords else width))
                    ey = int(end_point.y * (roi_frame.shape[0] if roi_coords else height))
                    
                    if roi_coords:
                        sx += roi_coords[0]
                        sy += roi_coords[1]
                        ex += roi_coords[0]
                        ey += roi_coords[1]
                    
                    cv2.line(annotated_frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
    
    return annotated_frame

# 3D描画に使うかも
def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []
    for index, landmark in enumerate(landmarks):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])
    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]
    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))
    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))
    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))
    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))
    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))
    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))
    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

def get_bbox_from_landmarks(landmarks, width, height):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x = max(0, int(min(xs) * width))
    max_x = min(width, int(max(xs) * width))
    min_y = max(0, int(min(ys) * height))
    max_y = min(height, int(max(ys) * height))
    return [min_x, min_y, max_x, max_y]

def main():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1)

    # 動画ファイルの読み込み
    cap = cv2.VideoCapture(r"E:\\ski\\data\\expand-reversed.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter('blaze.mp4', fourcc, fps, (width, height))

    # --- 逆再生動画の作成 ---
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    out_rev = cv2.VideoWriter(r"E:\\ski\\data\\reversed.mp4", fourcc, fps, (width, height))
    for frame in reversed(frames):
        out_rev.write(frame)
    out_rev.release()
    print('逆再生処理完了')
    # 逆再生動画で動画キャプチャを初期化
    cap = cv2.VideoCapture(r"E:\\ski\\data\\reversed.mp4")

    # 信頼度スコアを保存するリスト
    confidence = []
    angles=[]

    # ウィンドウ作成
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    # 最初のフレームで検出された人物のバウンディングボックスを保存
    first_person_bbox = None
    current_bbox = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        # 最初のフレームで人物検出
        ret, first_frame = cap.read()
        if not ret:
            print('動画が読み込めません')
            return
        rgb_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_first)
        pose_landmarker_result = landmarker.detect(mp_image)
        if pose_landmarker_result.pose_landmarks:
            first_person_bbox = get_bbox_from_landmarks(pose_landmarker_result.pose_landmarks[0], width, height)
            current_bbox = first_person_bbox
            print('最初のフレームで人物検出成功')
        else:
            print('最初のフレームで人物が検出されませんでした')
            current_bbox = None
            exit()

        # 動画の最初に戻る
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        notParallel=0
        full_body_score=0

        while True:

            ret, frame = cap.read()

            if not ret:
                print("動画の再生が終了しました。")
                print()
                break

            frame_idx += 1

            
            # annotated_frameを初期化
            annotated_frame = frame.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_frame = frame
            roi_rgb = rgb_frame
            roi_coords = None
            # current_bboxがある場合はROIで検出
            if current_bbox is not None:
                cx = int((current_bbox[0] + current_bbox[2]) / 2)
                cy = int((current_bbox[1] + current_bbox[3]) / 2)
                bbox_w = current_bbox[2] - current_bbox[0]
                bbox_h = current_bbox[3] - current_bbox[1]
                x_margin = bbox_w*2
                y_margin = bbox_h
                roi_x1 = int(max(0, cx - bbox_w // 2 - x_margin))
                roi_y1 = int(max(0, cy - bbox_h // 2 - y_margin))
                roi_x2 = int(min(width, cx + bbox_w // 2 + x_margin))
                roi_y2 = int(min(height, cy + bbox_h // 2 + y_margin))
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_rgb = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
                # ここでshapeチェック
                if roi_rgb.size == 0:
                    continue
                roi_rgb = np.ascontiguousarray(roi_rgb)
            mp_image = mp.Image(mp.ImageFormat.SRGB, roi_rgb)
            pose_landmarker_result = landmarker.detect(mp_image)

            # バウンディングボックス更新
            if pose_landmarker_result.pose_landmarks:

                # 2人以上あるいは検出無しだった場合はcurrent_bboxはそのまま
                if len(pose_landmarker_result.pose_landmarks) < 1 or len(pose_landmarker_result.pose_landmarks) > 1:
                    current_bbox = current_bbox

                # ただ1人のみ検出された場合はcurrent_bboxを更新
                else:

                    detected_bbox = get_bbox_from_landmarks(pose_landmarker_result.pose_landmarks[0], roi_frame.shape[1], roi_frame.shape[0])
                    # ROI→元画像座標に変換
                    current_bbox = [
                        detected_bbox[0] + roi_coords[0],
                        detected_bbox[1] + roi_coords[1],
                        detected_bbox[2] + roi_coords[0],
                        detected_bbox[3] + roi_coords[1]
                    ]
            
                    annotated_frame = draw_skeleton(annotated_frame, pose_landmarker_result, roi_coords, roi_frame, width, height)

                # ROI以外を黒く塗りつぶす
                if roi_coords:
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
                    annotated_frame = cv2.copyMakeBorder(
                        annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2],
                        roi_y1, height - roi_y2,
                        roi_x1, width - roi_x2,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]
                    )

            time = frame_idx / fps

            if len(pose_landmarker_result.pose_landmarks) < 1 or len(pose_landmarker_result.pose_landmarks) > 1:
                
                score=None

                angle=None
                
            else:
                
                score = calculate_full_body_visibility(pose_landmarker_result.pose_world_landmarks[0])

                for marks in pose_landmarker_result.pose_world_landmarks:
                    parallel, angle = isParallel(marks)
          
            confidence.append((time, score))
            angles.append((time, angle))
            
            # 結果を表示
            cv2.imshow('Pose Detection', annotated_frame)
            #out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    cv2.destroyAllWindows()

    # 統計処理
    
    # bbox検出成功のフレーム数
    total_frames = frame_idx
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

    print('Blaze Pose')
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

    scoreGraph(confidence)
    angleGraph(angles)
    
    cap.release()

if __name__ == "__main__":
    main()