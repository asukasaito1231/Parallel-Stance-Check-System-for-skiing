import cv2
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO('yolov8n-pose.pt')  # ポーズ推定用のYOLOv8モデル（より高精度）

# 動画キャプチャの初期化

cap = cv2.VideoCapture(r"E:\ski\data\expand.mp4")  # 動画ファイルを指定

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")

    exit()

# 動画の情報を取得
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 逆再生動画を保存するための設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r"E:\ski\data\reversed.mp4", fourcc, fps, (width, height))

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
cap = cv2.VideoCapture(r"E:\ski\data\reversed.mp4")

if not cap.isOpened():
    print("Error: 逆再生動画を開けませんでした。")
    exit()

# ウィンドウを作成
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# 動画保存の設定
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('previous.mp4', fourcc, fps, (width, height))

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
        # フレームサイズを縮小
        small_first_frame = cv2.resize(first_frame, (int(width), int(height)))
        # YOLOでポーズ推定を実行
        first_results = model(small_first_frame)

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
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# 動画の最初に戻る
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#フレーム読み込み開始
while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました。")
        break

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

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

            # ROIを元のサイズに拡大
            #roi = cv2.resize(roi, (width, height))

            # ROI領域でYOLOを実行
            results = model(roi)

                # 2人以上検出された場合は最も信頼度スコアが高い人物を描画
            if len(results[0].boxes) > 1:
                # 信頼度スコアを取得
                confidence_scores = results[0].boxes.conf.cpu().numpy()
                # 最も信頼度の高い人物のインデックスを取得
                best_person_idx = np.argmax(confidence_scores)
                # 最も信頼度の高い人物のみを選択
                results[0].boxes = results[0].boxes[best_person_idx:best_person_idx+1]
                results[0].keypoints = results[0].keypoints[best_person_idx:best_person_idx+1]

            # 検出されたバウンディングボックスをcurrent_bboxに設定
            detected_bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
            #ROI内での相対座標(0, 0)がROIの左上を元画像での絶対座標に変換(0, 0)が画像の左上
            current_bbox = [
                detected_bbox[0]+roi_x1,
                detected_bbox[1]+roi_y1,
                detected_bbox[2]+roi_x1,
                detected_bbox[3]+roi_y1
            ]
            
            annotated_frame = results[0].plot()

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
          print('current_bboxがnoneの場合')
          # YOLOでフレーム全体サイズのままポーズ推定を実行
          results = model(frame)
          annotated_frame = results[0].plot()

    except Exception as e:
        print('try文に突入しなかった')
        results = model(frame)
        annotated_frame = results[0].plot()

    # 結果を表示
    cv2.imshow('Pose Detection', annotated_frame)

    #out.write(annotated_frame)

    # 'q'キーまたはウィンドウの×ボタンで終了
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# リソースを解放
cap.release()
out.release()
# cv2.destroyAllWindow()