import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

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

# 信頼度スコアのグラフを作成する関数
def detectionResult(confidence_data):

    times = [t for t, s in confidence_data]
    scores = [s for t, s in confidence_data]
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker='o', linestyle='-')
    plt.ylim(0, 1)
    plt.xlabel('time(second)')
    plt.ylabel('confidence score')
    plt.title('Full Body Visibility Score per Frame')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('blaze-result')
    plt.close()

# 検出結果の描画
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
    cap = cv2.VideoCapture(r"E:\ski\data\clip.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 3D骨格動画の出力設定
    #out_3d = cv2.VideoWriter(r'E:\ski\data\3d-output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 骨格検出結果を描画した動画の出力設定
    #out_pose = cv2.VideoWriter(r'E:\ski\data\pose-detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 信頼度スコアを保存するリスト
    confidence_scores = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # BGR→RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_landmarker_result = landmarker.detect(mp_image)

            # 現在のフレーム番号と時間を取得
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = frame_number / fps

            # 全身のvisibility平均スコアを計算
            full_body_score = 0.0
            if pose_landmarker_result.pose_world_landmarks:
                for marks in pose_landmarker_result.pose_world_landmarks:
                    full_body_score = calculate_full_body_visibility(marks)
                    break  # 最初の人物のみ処理
            
            # 信頼度スコアを保存
            confidence_scores.append((current_time, full_body_score))

            # 骨格検出結果を動画に描画
            annotated_frame = frame.copy()
            if pose_landmarker_result.pose_landmarks:
                for pose_landmarks in pose_landmarker_result.pose_landmarks:
                    # 骨格を描画（新しいAPI用）
                    for i, landmark in enumerate(pose_landmarks):
                        # キーポイントを描画
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # 骨格の接続線を描画（主要な接続のみ）
                    connections = [
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 肩と腕
                        (11, 23), (12, 24), (23, 24),  # 肩と腰
                        (23, 25), (25, 27), (27, 29), (29, 31),  # 右足
                        (24, 26), (26, 28), (28, 30), (30, 32),  # 左足
                        (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),  # 右手
                        (16, 18), (16, 20), (16, 22), (18, 20), (20, 22)   # 左手
                    ]
                    
                    for connection in connections:
                        start_idx, end_idx = connection
                        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                            start_point = pose_landmarks[start_idx]
                            end_point = pose_landmarks[end_idx]
                            
                            start_x = int(start_point.x * width)
                            start_y = int(start_point.y * height)
                            end_x = int(end_point.x * width)
                            end_y = int(end_point.y * height)
                            
                            cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            
            # 骨格検出動画を保存
            #out_pose.write(annotated_frame)

            # 3D骨格描画
            fig = plt.figure(figsize=(width/100, height/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
            if pose_landmarker_result.pose_world_landmarks:
                for marks in pose_landmarker_result.pose_world_landmarks:
                    plot_world_landmarks(
                        plt,
                        ax,
                        marks,
                    )
            #plt.show()  # 表示は省略
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_pil = Image.open(buf)
            img_np = np.array(img_pil)
            # PNGはRGBAなのでBGRに変換
            if img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # サイズ調整（念のため）
            #img_np = cv2.resize(img_np, (width, height))
            #out_3d.write(img_np)
            
    cap.release()
    #out_3d.release()
    #out_pose.release()
    print('3D骨格動画を保存しました: 3d-output.mp4')
    print('骨格検出動画を保存しました: pose-detection.mp4')
    
    # 信頼度スコアのグラフを作成
    detectionResult(confidence_scores)

    # --- 追加統計処理 ---
    total_frames = len(confidence_scores)
    zero_score_frames = [score for t, score in confidence_scores if score == 0.0]
    num_zero_score_frames = len(zero_score_frames)
    zero_score_percent = (num_zero_score_frames / total_frames) * 100 if total_frames > 0 else 0.0

    # 信頼度スコア0が1秒以上続いた区間の個数をカウント
    zero_streaks = 0
    streak_length = 0
    for t, score in confidence_scores:
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
    positive_scores = [score for t, score in confidence_scores if score > 0.0]
    avg_positive_score = np.mean(positive_scores) if positive_scores else 0.0

    print(f'総フレーム数: {total_frames}')
    print(f'bbox検出無しのフレーム数: {num_zero_score_frames} ({zero_score_percent:.2f}%)')
    print(f'bbox検出無しが1秒以上続いた区間の個数: {zero_streaks}')
    print(f'bboxを検出した際の平均信頼度スコア: {avg_positive_score:.4f}')
    
if __name__ == '__main__':
    main()