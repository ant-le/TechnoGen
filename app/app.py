import sys, yaml

from pathlib import Path

sys.path.append(".")

from flask import Flask, render_template, request, redirect
from utils import encode, decode, sample, refresh

app = Flask(__name__)

MODEL_VERSION = "baseline"
with open(f"config/{MODEL_VERSION}.yml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
audio_path = Path.cwd() / "app" / "static" / "audio"
audio_path.mkdir(exist_ok=True)

config["dataset"]["path"] = audio_path


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html", path=audio_path)


@app.route("/compress", methods=["POST", "GET"])
def compress():
    refresh(config)
    if request.method == "POST":
        request.files["file"].save(audio_path / "input.wav")
        encode(config)
        decode(config)
    return render_template(
        "compress.html",
        path=audio_path,
        sr=config["dataset"]["sample_rate"],
        seconds=config["dataset"]["hop_size"],
        compression_rate=config["model"]["stride"] ** config["model"]["layers"],
        codebooks=config["model"]["codebook_size"],
    )


@app.route("/generator")
def generator():
    sample(config)
    return redirect("home")


if __name__ == "__main__":
    app.run(debug=True)
