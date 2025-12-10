import os
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
import gc

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
    plt.savefig('YOLO-angle-roi-rewind.png')
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
    plt.savefig('YOLO-confidence-roi-rewind.png')
    plt.close()

def isParallel(keypoints):

    try:
        '''
        # 腰の座標
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        '''
        
        # 膝の座標
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        
        # 足首の座標
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        '''
        # 太ももベクトル（腰→膝）
        left_thigh_vec = left_knee - left_hip
        right_thigh_vec = right_knee - right_hip
        '''
        # 脛ベクトル（膝→足首）
        left_shin_vec = left_ankle - left_knee
        right_shin_vec = right_ankle - right_knee
        '''
        # 太ももの角度計算
        thigh_angle = angle_between(left_thigh_vec, right_thigh_vec)
        if thigh_angle > 30:
            return 50, False
        '''
        # 脛の角度計算
        angle = angle_between(left_shin_vec, right_shin_vec)
        '''
        if shin_angle > 30:
            return 50, False
        '''

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

def smoothing(angles):

    b=angles

    for i in range(1, len(angles) - 1):

        z, left  = angles[i - 1]
        time,  mid   = angles[i]
        z, right = angles[i + 1]

        if mid==None:
            continue

        # 左右が14以上、中間が13以下なら14に修正
        if left >= 14 and mid <= 13 and right >= 14:
            b[i] = (time, 14)

        # 左右が13以下、中間が14以上なら13に修正
        elif left <= 13 and mid >= 14 and right <= 13:
            b[i] = (time, 13)

    return b

def main(filename, first_y1, first_y2, first_x1, first_x2, start, end):

    #filenameは拡張子無し

    # YOLOモデルの読み込み
    object_detection_model = YOLO('yolo12n.pt')
    detect_skeleton_model=YOLO('yolo11n-pose.pt')

    cap_original = cv2.VideoCapture(rf"C:\Users\asuka\thesis\ps_check_system\static\uploads\{filename}.mp4")

    if not cap_original.isOpened():
        print("Error: カメラまたは動画を開けませんでした。")
        exit()

    fps = cap_original.get(cv2.CAP_PROP_FPS)
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_x=width/320
    scale_y=height/180

    first_x1=int(first_x1*scale_x)
    first_y1=int(first_y1*scale_y)
    first_x2=int(first_x2*scale_x)
    first_y2=int(first_y2*scale_y)

    print('yoloReversed')
    print(filename)
    print(first_x1)
    print(first_y1)
    print(first_x2)
    print(first_y2)
    print(fps)
    print(width)
    print(height)

    # 既に逆再生ファイルがあるなら
    if os.path.exists(rf"D:\DCIM\MOVIE\far\{filename}-reversed.mp4"):

        cap_reverse = cv2.VideoCapture(rf"D:\DCIM\MOVIE\far\{filename}-reversed.mp4")
        print("既に逆再生ファイルが存在します")

    # 逆再生ファイルが無いなら作る
    else:

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 逆再生動画を保存するための設定

        r_out = cv2.VideoWriter(rf"D:\DCIM\MOVIE\far\{filename}-reversed.mp4", fourcc, fps, (width, height))

        # フレームを配列に保存
        temps = []
        while True:
            ret, temp = cap_original.read()
            if not ret:
                break
            temps.append(temp)

        # フレームを逆順に保存
        for temp in reversed(temps):
            r_out.write(temp)

        r_out.release()

        temps=[]

        cap_reverse = cv2.VideoCapture(rf"D:\DCIM\MOVIE\far\{filename}-reversed.mp4")
        print("逆再生ファイルが存在しないので作りました")

    if not cap_reverse.isOpened():
        print("Error: 逆再生動画を開けませんでした。")
        exit()

    # 動画のスタート位置➡逆再生するので動画の終了時間のフレームを見る
    end_frame = int(fps * end)-10

    # 動画の再生開始位置をセット
    cap_original.set(cv2.CAP_PROP_POS_FRAMES, end_frame)    

    # 現在のバウンディングボックスを保存
    current_bbox = None

    # 最初のフレームを読み込んで人物を検出
    ret, first_frame = cap_original.read()

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

    confidence=[]

    angles=[]

    current_index=0

    # 総フレーム数
    total_frames = int(cap_reverse.get(cv2.CAP_PROP_FRAME_COUNT))

    # 逆再生する場合の開始フレームex180,6s
    reverse_start_frame=total_frames - end_frame

    start_frame=int(start*fps)+10

    # 逆再生する場合の終了フレームex240,8s
    reverse_end_frame=total_frames - start_frame

    # 動画の再生開始位置をセット
    # pos=position,
    cap_reverse.set(cv2.CAP_PROP_POS_FRAMES, reverse_start_frame)

    first_frame=reverse_start_frame

    need_frame=reverse_end_frame-reverse_start_frame

    notParallel=0

    #フレーム読み込み開始
    while True:

        ret, frame = cap_reverse.read()

        current_index+=1

        if current_index > need_frame:
            print("終了フレームに達しました。")
            break

        time = (first_frame+current_index)/fps

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
                y_margin=bbox_height/2

                # ROIの範囲をcenter_x, center_yを中心にbboxより少し大きい大きさで設定
                bbox_roi_x1 = max(0, int(center_x - bbox_width / 2-x_margin))
                bbox_roi_y1 = max(0, int(center_y - bbox_height / 2-y_margin))
                bbox_roi_x2 = min(width-1, int(center_x + bbox_width / 2+x_margin))
                bbox_roi_y2 = min(height-1, int(center_y + bbox_height / 2+y_margin))

                bbox_roi=frame[bbox_roi_y1:bbox_roi_y2, bbox_roi_x1:bbox_roi_x2]

                # ROI領域でスキーヤー検出を実行
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

                    angle = isParallel(keypoints)

                    if angle >= 5:
                        notParallel+=1
            
                    judge_frame = np.full_like(frame, 255, dtype=np.uint8)

                    if angle >= 5:
                        judge_frame=np.full_like(frame, (200, 200, 255), dtype=np.uint8)

                    # ROI部分だけ元の画像からコピー
                    judge_frame[skeleton_roi_y1:skeleton_roi_y2, skeleton_roi_x1:skeleton_roi_x2] = annotated_frame

                    # 完成：ROI以外は白、サイズは変わらない
                    annotated_frame = judge_frame
                    
                    angle=int(angle)
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
                    x = skeleton_roi_x2
                    y = skeleton_roi_y2

                    # テキスト描画
                    cv2.putText(annotated_frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

                    #if f"{angle}.png is exists":
                        #angle=angle+100

                    #cv2.imwrite(f"{angle}.png", annotated_frame)
                    
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
            score=None
            angle=None

        confidence.append((time, score))
        angles.append((time, angle))
        frames.append(annotated_frame)

    cap_reverse.release()

    #angles = smoothing(angles)
    '''
    for i in range(len(angles)):

        if angles[i][1] is not None and angles[i][1] >= 14:

            overlay = frames[i].copy()
            red = np.full_like(overlay, (0, 0, 255))  # 赤
            alpha = 0.3
            cv2.addWeighted(red, alpha, overlay, 1 - alpha, 0, dst=frames[i])
    '''
    fourcc = cv2.VideoWriter_fourcc("H", "2", "6", "4")
    final_out = cv2.VideoWriter(r".\static\result_video\ps_check_result.mp4", fourcc, fps, (width, height))

    # フレームを逆順に保存
    for frame in reversed(frames):
        final_out.write(frame)

    frames=[]

    # リソースを解放
    final_out.release()
    
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

    print('【YOLO】 - far2.mp4')
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
    #scoreGraph(confidence)
    #angleGraph(angles)

    confidence=[]
    angle=[]

    parallel_frame=need_frame-notParallel

    success=int((parallel_frame/need_frame)*100)

    cv2.destroyAllWindows()
    return True, success

    