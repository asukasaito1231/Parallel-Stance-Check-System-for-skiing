import cv2
import mediapipe as mp

# Mediapipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 動画キャプチャの初期化
cap = cv2.VideoCapture("static.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するscale
resize_scale = 1

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

    # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Mediapipeで骨格検出を実行
    result = pose.process(frame_rgb)

    # 検出結果を描画
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # 縮小されたフレームを表示
    cv2.imshow('Pose Detection', small_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()


