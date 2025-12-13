import os
from yoloReversed import main as yoloReversed_main
from yolo import main as yolo_main
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/", methods=["GET", "POST"])
def index():

    # 送信ボタンを押した時
    if request.method == "POST":

        video = request.files.get("video")

        # 1. 動画ファイルが選択されてるかチェック
        if not video:
            error_msg = "スキー動画が選択されていません。"
            return render_template("index.html", error=error_msg)

        duration = float(request.form.get("duration"))

        trim_start_raw = request.form.get("trim_start")
        trim_end_raw   = request.form.get("trim_end")

        # Noneだった場合は０と動画の長さをセット、数値がある場合はフロートに変換
        trim_start = float(trim_start_raw) if trim_start_raw else 0.0
        trim_end   = float(trim_end_raw)   if trim_end_raw   else duration

        # 2-1 . 
        if trim_start > trim_end:
            error_msg = "動画の開始時間が終了時間より後です。"
            return render_template("index.html", error=error_msg)

        # 2-2 . 
        if trim_start == trim_end:
            error_msg = "動画の開始時間と終了時間が同じです。"
            return render_template("index.html", error=error_msg)

        # 2-3 . 
        if trim_start < 0 or trim_end < 0:
            error_msg = "動画の開始時間と終了時間は負の数になってはいけません。"
            return render_template("index.html", error=error_msg)

        # 2-4
        if trim_end > duration:
            error_msg = "動画の終了時間が動画の長さを超えてます。。"
            return render_template("index.html", error=error_msg)

        # 3. 撮影方法が選択されてるかチェック
        shooting_method = request.form.get("shooting_method")

        if not shooting_method:
            error_msg = "撮影方法が選択されてません。"
            return render_template("index.html", error=error_msg)

        first_x1 = request.form.get("first_x1")
        first_y1 = request.form.get("first_y1")
        first_x2 = request.form.get("first_x2")
        first_y2 = request.form.get("first_y2")

        # 4. 初期ROIが設定されてるかチェック
        if not all([first_x1, first_y1, first_x2, first_y2]):
            if shooting_method == "back":
                error_msg = "スキー動画の開始時間時にターゲットがどこにいるか設定されてません。"
            elif shooting_method == "front":
                error_msg = "スキー動画の終了時間時にターゲットがどこにいるか設定されてません。"
            return render_template("index.html", error=error_msg)

        # エラーが無ければ撮影方法に応じたプログラムに、ファイル名、初期ROI、開始時間、終了時間を渡す
        first_x1 = int(float(first_x1))
        first_y1 = int(float(first_y1))
        first_x2 = int(float(first_x2))
        first_y2 = int(float(first_y2))

        name, ext = os.path.splitext(video.filename)
        filename = name
            
        videopath = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        video.save(videopath)

        if shooting_method == "back":
            fullCheck, success = yolo_main(filename, first_y1, first_y2, first_x1, first_x2, trim_start, trim_end)
        elif shooting_method == "front":
            fullCheck, success = yoloReversed_main(filename, first_y1, first_y2, first_x1, first_x2, trim_start, trim_end)

        if fullCheck==False:
            error_msg="ターゲットを追跡できませんでした。"
            return render_template("index.html", error=error_msg)

        return redirect(url_for("result", success=success))

    # GET の場合
    return render_template("index.html")

@app.route("/result")
def result():
    success = request.args.get("success")
    return render_template("result.html", success=success)

@app.route("/howto")
def howto():
    return render_template("howto.html")

@app.route("/developer")
def developer():
    return render_template("developer.html")

if __name__ == "__main__":
    app.run(debug=True)


