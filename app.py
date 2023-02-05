#!/usr/bin/env python

from os import getenv

import PIL.Image
from dotenv import load_dotenv
from autotagger import Autotagger
from base64 import b64encode
from flask import Flask, request, render_template, jsonify, abort
from werkzeug.exceptions import HTTPException

load_dotenv()
model_path = getenv("MODEL_PATH", "models/model.onnx")
autotagger = Autotagger(model_path)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_PRETTYPRINT_REGULAR"] = True

@app.route("/", methods=["GET"])
def index():
  return render_template("index.html")

@app.route("/evaluate", methods=["POST"])
def evaluate():
  files = request.files.getlist("file")
  threshold = float(request.values.get("threshold", 0.1))
  general_threshold = float(request.values.get("general_threshold", threshold))
  character_threshold = float(request.values.get("character_threshold", threshold))
  output = request.values.get("format", "html")
  limit = int(request.values.get("limit", 50))

  images = [PIL.Image.open(file) for file in files]
  predictions = autotagger.predict(images, general_threshold=threshold, character_threshold=character_threshold, limit=limit)

  if output == "html":
    for file in files:
      file.seek(0)

    base64data = [b64encode(file.read()).decode() for file in files]
    return render_template("evaluate.html", predictions=zip(base64data, predictions))
  elif output == "json":
    predictions = [{ "filename": file.filename, "tags": tags } for file, tags in zip(files, predictions)]
    return jsonify(predictions)
  else:
    abort(400)
