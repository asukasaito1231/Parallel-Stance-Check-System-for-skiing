import os
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ダミー解析関数
def check_parallel(video_path):
    # 本当はここでOpenCVとかで解析する
    # いまはダミーで「崩れたフレーム数」を返す
    return 5  # 仮に5枚あったことにする

@app.route("/", methods=["GET", "POST"])
def index():

    error_msg = ""  # エラーメッセージ用

    if request.method == "POST":

        # ファイルを受け取る
        file = request.files.get("video")

        # ファイルが選択されていない場合
        if not file or file.filename == "":
            error_msg = "動画ファイルが選択されていません"

        # 動画以外がアップロードされた場合
        elif not file.mimetype.startswith("video/"):
            error_msg = "動画ファイルのみアップロード可能です"

        else:
            # 保存先パス
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

            file.save(filepath)  # 今は保留

            # ダミー解析呼び出し
            error_frames = check_parallel(filepath)

            # /result にリダイレクト
            return redirect(url_for("result", filename=file.filename, errors=error_frames))

    return render_template("index.html", error=error_msg)

@app.route("/result")
def result():

    filename = request.args.get("filename")

    errors = request.args.get("errors", 0)

    return render_template("result.html", filename=filename, errors=errors)


if __name__ == "__main__":

    app.run(debug=True)
