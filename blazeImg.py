import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'

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

def isParallel(keypoints, angle_threshold=35):
    
    # 13=左膝, 14=右膝, 15=左足首, 16=右足首のランドマークを取得
    left_knee = np.array([keypoints[13].x, keypoints[13].y, keypoints[13].z])
    right_knee = np.array([keypoints[14].x, keypoints[14].y, keypoints[14].z])
    left_ankle = np.array([keypoints[15].x, keypoints[15].y, keypoints[15].z])
    right_ankle = np.array([keypoints[16].x, keypoints[16].y, keypoints[16].z])

    # 膝→足首ベクトル
    left_leg_vec = left_ankle - left_knee
    right_leg_vec = right_ankle - right_knee

    # 左右の脚ベクトルのなす角度を計算
    angle = angle_between(left_leg_vec, right_leg_vec)

    return angle < angle_threshold, angle

def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot) * 180 / np.pi
    return angle

def main():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1)
    with PoseLandmarker.create_from_options(options) as landmarker:
        # 画像ファイルの読み込み
        mp_image = mp.Image.create_from_file('two.jpg')
        # 解析実行
        pose_landmarker_result = landmarker.detect(mp_image)
        
        # 解析結果を描画
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
        for marks in pose_landmarker_result.pose_world_landmarks:
            # パラレル診断
            is_parallel, angle = isParallel(marks)
            if is_parallel:
                result_text = f"脚は平行です（なす角度: {angle:.2f}度）"
            else:
                result_text = f"脚は平行ではありません（なす角度: {angle:.2f}度）"
            print(result_text)
            # すべての骨格座標をターミナルに表示
            #for i, landmark in enumerate(marks):
                #print(f"Landmark {i}: x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility}")
            plot_world_landmarks(
                plt,
                ax,
                marks,
            )
        # 画像内に診断結果を描画
        plt.figtext(0.5, 0.05, result_text, ha='center', fontsize=16, color='red', bbox={'facecolor':'white', 'alpha':0.7, 'pad':5})
        plt.savefig('result.jpg')
        plt.show()
    
if __name__ == '__main__':
    main()