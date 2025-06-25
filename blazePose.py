import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import io

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
    # 入力動画ファイルと出力動画ファイル
    input_video_path = 'btest.mp4'
    output_video_path = 'blaze_pose_output.mp4'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1)

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイル {input_video_path} を開けませんでした。")
            return

        # 3Dプロットの準備と出力動画の解像度取得
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
        
        # ダミープロットで出力画像のサイズを取得
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plot_img = cv2.imdecode(img_arr, 1)
        height, width, _ = plot_img.shape
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        timestamp_ms = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_landmarker_result.pose_world_landmarks:
                for marks in pose_landmarker_result.pose_world_landmarks:
                    plot_world_landmarks(
                        plt,
                        ax,
                        marks,
                    )

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                buf.close()
                plot_img = cv2.imdecode(img_arr, 1)
                
                # リアルタイム表示
                cv2.imshow('BlazePose 3D Visualization', plot_img)
                
                # out.write(plot_img)
            else:
                # ポーズが検出されない場合は黒い画面を表示
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.imshow('BlazePose 3D Visualization', black_frame)
                out.write(black_frame)

            # 'q'キーまたはウィンドウの×ボタンで終了
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('BlazePose 3D Visualization', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        plt.close(fig)
        print(f"動画を {output_video_path} に保存しました。")

if __name__ == '__main__':
    main()