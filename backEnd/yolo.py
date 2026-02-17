import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

def smoothing(angles):

    b=angles

    for i in range(1, len(angles) - 1):

        left  = angles[i - 1]
        mid   = angles[i]
        right = angles[i + 1]

        if mid==None:
            continue

        # 前後が白なら、真ん中も白
        if left < 14 and mid >= 14  and right < 14:
            b[i] = ((left+right)/2)

        # 前後が黄なら、真ん中も黄（真ん中が白バージョン）
        elif 14 <= left and left < 16 and  mid < 14 and 14 <= right and right < 16:
            b[i] = ((left+right)/2)

        # 前後が黄なら、真ん中も黄（真ん中が赤バージョン）
        elif 14 <= left and left < 16 and  mid > 16 and 14 <= right and right < 16:
            b[i] = ((left+right)/2)

        # 前後が赤なら、真ん中も赤
        elif left >= 16 and mid < 16 and right >= 16:
            b[i] = ((left+right)/2)

        else:
            continue

    return b

def colorize_frames_by_angle(angles, frames, alpha=0.3):
    """
    angles: [(time, angle), ...]
    frames: [frame, frame, ...]  (BGR, OpenCV image)
    alpha : 色の濃さ（0〜1、小さいほど薄い）
    """

    assert len(angles) == len(frames), "angles と frames の長さが一致しません"

    colored_frames = []

    for i, frame in enumerate(frames):

        angle = angles[i]

        # 何もしない場合
        if angle is not None and angle < 14:
            colored_frames.append(frame)
            continue

        overlay = frame.copy()

        if angle is None:
            # 薄い黒
            color = (0, 0, 0)
        elif 14 <= angle < 16:
            # 薄い黄色 (BGR)
            color = (0, 255, 255)
        else:  # angle >= 16
            # 薄い赤 (BGR)
            color = (0, 0, 255)

        # フレーム全体を塗る
        overlay[:] = color

        # 半透明合成
        colored = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        colored_frames.append(colored)

    return colored_frames

# 角度グラフ表示関数-時間軸反転プロット
def angleGraph(angles):

    times  = [t for t, a in angles]
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
    plt.savefig('before-smoothing.png')
    plt.close()

# bbox検出結果グラフ表示関数-時間軸反転プロット
def scoreGraph(confidence):

    times  = [t for t, s in confidence]
    scores = [s for t, s in confidence]

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
    plt.savefig('YOLO-confidence.png')
    plt.close()

def isParallel(keypoints):

    try:
        
        # 膝の座標
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        
        # 足首の座標
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # 脛ベクトル（膝→足首）
        left_shin_vec = left_ankle - left_knee
        right_shin_vec = right_ankle - right_knee
    
        # 脛の角度計算
        angle = angle_between(left_shin_vec, right_shin_vec)

        return angle
        
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

def main(filename, first_y1, first_y2, first_x1, first_x2, start, end):
    
    print(start)
    print(end)

    # YOLOモデルの読み込み
    object_detection_model = YOLO('yolo12n.pt')
    pose_estimation_model=YOLO('yolo11x-pose.pt')

    cap = cv2.VideoCapture(rf"C:\Users\asuka\thesis\ps_check_system\static\uploads\{filename}.mp4")

    if not cap.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        return False, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_x=width/320
    scale_y=height/180

    first_x1=int(first_x1*scale_x)
    first_y1=int(first_y1*scale_y)
    first_x2=int(first_x2*scale_x)
    first_y2=int(first_y2*scale_y)

    print(scale_x)
    print(scale_y)
    print(first_x1)
    print(first_y1)
    print(first_x2)
    print(first_y2)
    
    # 動画のスタート位置
    start_frame = int(fps * start)

    # 動画の再生開始位置をセット
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) 

    # 現在のバウンディングボックスを保存
    current_bbox = None

    # 最初のフレームを読み込んで人物を検出
    ret, first_frame = cap.read()

    # (x1, y1)は左上、(x2, y2)が右下
    first_ROI=first_frame[first_y1:first_y2, first_x1:first_x2]

    # 物体検出を実行（人間のみ検出）
    first_results = object_detection_model(first_ROI, classes=[0])

    if len(first_results[0].boxes)==1:
        
        # 相対座標→絶対座標の変換、ROIの初期値を設定
        detected_bbox = first_results[0].boxes[0].xyxy[0].cpu().numpy()

        #ROI内での相対座標 " (0, 0)=ROIの左上 " を全体フレームでの絶対座標に変換 "(0, 0)が全体フレームの左上 "
        current_bbox = [
            detected_bbox[0]+first_x1,
            detected_bbox[1]+first_y1,
            detected_bbox[2]+first_x1,
            detected_bbox[3]+first_y1
            ]
                
        print("最初のフレームでの人物発見成功")

    else:
        print('first fail')
        return False, 0

    frames=[]

    #confidence=[]

    angles=[]

    current_index=0

    # 動画の再生開始位置をセット
    # pos=position,
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 動画の終了位置
    end_frame = int(fps * end)

    need_frame=end_frame - start_frame

    parallel=0

    #フレーム読み込み開始
    while True:

        ret, frame = cap.read()

        current_index+=1

        if current_index > need_frame:
            print("終了フレームに達しました。")
            break

        #time = (start_frame+current_index)/fps

        try:
            if current_bbox is not None:

                # ターゲット検出フェーズ

                # バウンディングボックスの中心座標を計算
                center_x = int((current_bbox[0] + current_bbox[2]) / 2)
                center_y = int((current_bbox[1] + current_bbox[3]) / 2)

                # current_bboxの幅と高さを計算
                bbox_width = current_bbox[2] - current_bbox[0]
                bbox_height = current_bbox[3] - current_bbox[1]

                #多少余白を持たせることで確実にターゲットを検出 
                x_margin=bbox_width/2
                y_margin=bbox_height/2

                # ROIの範囲をcenter_x, center_yを中心にbboxより少し大きい大きさで設定
                bbox_roi_x1 = max(0, int(center_x - bbox_width / 2-x_margin))
                bbox_roi_y1 = max(0, int(center_y - bbox_height / 2-y_margin))
                bbox_roi_x2 = min(width-1, int(center_x + bbox_width / 2+x_margin))
                bbox_roi_y2 = min(height-1, int(center_y + bbox_height / 2+y_margin))

                bbox_roi=frame[bbox_roi_y1:bbox_roi_y2, bbox_roi_x1:bbox_roi_x2]

                # ROI領域でターゲット検出を実行
                bbox_results = object_detection_model(bbox_roi, classes=[0])

                # 2人以上あるいは検出無しだった場合はcurrent_bboxはそのまま
                if len(bbox_results[0].boxes) < 1 or len(bbox_results[0].boxes) > 1:

                    # ROI以外を白く塗りつぶした画像をannotated_frameに格納
                    annotated_frame = cv2.copyMakeBorder(
                        annotated_frame,
                        bbox_roi_y1, height - bbox_roi_y2,
                        bbox_roi_x1, width - bbox_roi_x2,
                        cv2.BORDER_CONSTANT,
                        value=[255, 255, 255]
                    )

                    #score=None

                    angle=None

                #ただ1人のみ検出された場合はcurrent_bboxを更新
                #bboxと骨格を描画 
                else:

                    # 姿勢推定フェーズ
                    
                    detected_bbox = bbox_results[0].boxes[0].xyxy[0].cpu().numpy()

                    #ROI内での相対座標 " (0, 0)=ROIの左上 " を全体フレームでの絶対座標に変換 "(0, 0)が全体フレームの左上 "
                    current_bbox = [
                        detected_bbox[0]+bbox_roi_x1,
                        detected_bbox[1]+bbox_roi_y1,
                        detected_bbox[2]+bbox_roi_x1,
                        detected_bbox[3]+bbox_roi_y1
                        ]
                    
                    # bboxの中心座標を計算
                    center_x = int((current_bbox[0] + current_bbox[2]) / 2)
                    center_y = int((current_bbox[1] + current_bbox[3]) / 2)

                    # current_bboxの幅と高さを計算
                    bbox_width = current_bbox[2] - current_bbox[0]
                    bbox_height = current_bbox[3] - current_bbox[1]

                    #多少余白を持たせることで確実にターゲットを検出 
                    x_margin=bbox_width/2
                    y_margin=bbox_height/2

                    # ROIの範囲をcenter_x, center_yを中心にbboxより少し大きい大きさで設定
                    skeleton_roi_x1 = max(0, int(center_x - bbox_width / 2-x_margin))
                    skeleton_roi_y1 = max(0, int(center_y - bbox_height / 2-y_margin))
                    skeleton_roi_x2 = min(width-1, int(center_x + bbox_width / 2+x_margin))
                    skeleton_roi_y2 = min(height-1, int(center_y + bbox_height / 2+y_margin))

                    # 姿勢推定用のROI領域を抽出
                    skeleton_roi = frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2]
                    
                    # ROI領域で姿勢推定を実行
                    skeleton_results=pose_estimation_model(skeleton_roi, classes=[0])
                
                    annotated_frame = skeleton_results[0].plot()

                    # キーポイント取得
                    keypoints_raw = skeleton_results[0].keypoints[0].data.cpu().numpy()
                    
                    #elif len(keypoints_raw.shape) == 3 and keypoints_raw.shape[1] == 17:
                    #(1検出人数, 17キーポイント, 3座標とスコア)

                    #検出された1人目のキーポイントを取得
                    keypoints = keypoints_raw[0]

                    #score = float(bbox_results[0].boxes.conf[0].cpu().numpy())

                    angle = isParallel(keypoints)

                    if angle < 14:
                        parallel+=1

                    white_frame = np.full_like(frame, 255, dtype=np.uint8)

                    # ROI部分だけ元の画像からコピー
                    white_frame[bbox_roi_y1:bbox_roi_y2, bbox_roi_x1:bbox_roi_x2] = annotated_frame

                    # 完成：ROI以外は白、サイズは変わらない
                    annotated_frame = white_frame
                    '''
                    text = f"{angle}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 7
                    color = (255, 0, 0) # BGR
                    thickness = 7

                    # 文字サイズを取得して右下の座標を計算
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_w, text_h = text_size

                    # フレームサイズを取得
                    h, w, _ = annotated_frame.shape

                    # 右下に配置（マージン付き）
                    x = bbox_roi_x2
                    y = bbox_roi_y2
                    
                    # テキスト描画
                    cv2.putText(annotated_frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                    '''
        except Exception as e:
            print('try文に突入しなかった')
            # ROI以外を白く塗りつぶした画像をannotated_frameに格納
            annotated_frame = cv2.copyMakeBorder(
                bbox_roi,
                bbox_roi_y1, height - bbox_roi_y2,
                bbox_roi_x1, width - bbox_roi_x2,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
                )
            #score=None
            angle=None

        #confidence.append((time, score))
        angles.append(angle)
        frames.append(annotated_frame)

    cap.release()

    angles = smoothing(angles)

    frames=colorize_frames_by_angle(angles, frames)
    
    fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")  # H.264
    out = cv2.VideoWriter(r".\static\result_video\ps_check_result.mp4", fourcc, fps, (width, height))

    # フレームを保存
    for frame in frames:
        out.write(frame)

    # リソースを解放
    out.release()

    #confidence=[]
    angle=[]
    frames=[]
    
    #scoreGraph(confidence)
    #angleGraph(angles)

    success=int((parallel/need_frame)*100)

    cv2.destroyAllWindows()

    return True, success
