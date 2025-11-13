import os
import sys
sys.path.append(r"C:\Users\asuka\thesis\backGround")
#from yoloReversed import main
#from yolo import main
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/", methods=["GET", "POST"])
def index():

    error_msg = ""
    ps_result_video_name=""

    if request.method == "POST":

        # ファイルを受け取る
        file = request.files.get("video")
        
        # 撮影方法を受け取る
        shooting_method = request.form.get("shooting_method")

        # ファイルが選択されていない場合
        if not file or file.filename == "":
            error_msg = "動画ファイルが選択されていません"

        # 動画以外がアップロードされた場合
        elif not file.mimetype.startswith("video/"):
            error_msg = "動画ファイルのみアップロード可能です"

        elif not shooting_method:
            error_msg = "撮影方法が選択されてません"

        else:
            # 保存先パス
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

            file.save(filepath)  # 今は保留

            '''
            if shooting_method == "後ろから撮影":
                ps_result_video_name=yolo.main(initial_position)

            else:
                ps_result_video_name=yoloReversed.main(initial_position)
            '''
            # /result にリダイレクト
            return redirect(url_for("result", ps_result_video_name=ps_result_video_name.filename))

    return render_template("index.html", error=error_msg)

@app.route("/result")
def result():

    ps_result_video_name = request.args.get("ps_result_video_name")

    return render_template("result.html", ps_result_video_name=ps_result_video_name.filename)

if __name__ == "__main__":

    app.run(debug=True)
