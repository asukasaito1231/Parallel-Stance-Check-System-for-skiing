import os
import sys
from yoloReversed import main as yoloReversed_main
from yolo import main as yolo_main
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/", methods=["GET", "POST"])
def index():

    error_msg = ""
    initial_position=""

    if request.method == "POST":

        # ファイルを受け取る
        file = request.files.get("video")
        
        name, ext = os.path.splitext(file.filename)

        filename=name

        # 撮影方法を受け取る
        shooting_method = request.form.get("shooting_method")

        # ファイルが選択されていない場合
        if not file or file.filename == "":
            error_msg = "動画ファイルが選択されていません"

        elif not shooting_method:
            error_msg = "撮影方法が選択されてません"

        else:
            # 保存先パス
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

            file.save(filepath)

            if shooting_method == "後ろから撮影":
                yolo_main(filename)

            else:
                yoloReversed_main(filename)

            # /result にリダイレクト
            return redirect(url_for("result"))

    return render_template("index.html", error=error_msg)

@app.route("/result")
def result():

    return render_template("result.html")

if __name__ == "__main__":

    app.run(debug=True)

