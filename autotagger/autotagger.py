import functools
import html
import os

import cv2
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL.Image

def make_square(img, target_size):
  old_size = img.shape[:2]
  desired_size = max(*old_size, target_size)

  delta_w = desired_size - old_size[1]
  delta_h = desired_size - old_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)

  color = [255, 255, 255]
  new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

  return new_im

def smart_resize(img, size):
  if img.shape[0] > size:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
  elif img.shape[0] < size:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
  return img

def process_image(image, height, width):
  image = image.convert("RGBA")
  new_image = PIL.Image.new("RGBA", image.size, "WHITE")
  new_image.paste(image, mask=image)
  image = new_image.convert("RGB")
  image = np.asarray(image)

  image = image[:, :, ::-1]

  image = make_square(image, height)
  image = smart_resize(image, height)
  image = image.astype(np.float32)
  image = np.expand_dims(image, 0)

  return image


class Autotagger:
  def __init__(self, model_path="models/model.onnx", tags_path="models/selected_tags.csv"):
    self.model = rt.InferenceSession(model_path)
    self.load_labels(tags_path)

  def load_labels(self, tags_path):
    df = pd.read_csv(tags_path)
    self.tag_names = df["name"].tolist()
    self.rating_indices = list(np.where(df["category"] == 9)[0])
    self.general_indices = list(np.where(df["category"] == 0)[0])
    self.character_indices = list(np.where(df["category"] == 4)[0])

  def predict(self, images, general_threshold=0.35, character_threshold=0.8, limit=50):
    inputs = self.model.get_inputs()[0]
    _, height, width, _ = inputs.shape
    input_name = inputs.name
    label_name = self.model.get_outputs()[0].name

    filenames = [image.filename for image in images]
    images = [process_image(image, height, width) for image in images]
    results = []

    for filename, image in zip(filenames, images):
      probs = self.model.run([label_name], {input_name: image})[0]
      labels = list(zip(self.tag_names, probs[0].astype(float)))

      ratings = pd.DataFrame([
        ("rating:" + labels[i][0], labels[i][1])
        for i in self.rating_indices
      ], columns=["tag", "score"]).sort_values("score", ascending=False).head(1)
      gentags = pd.DataFrame([x for x in [labels[i] for i in self.general_indices] if x[1] > general_threshold], columns=["tag", "score"]).sort_values("score", ascending=False)
      chartags = pd.DataFrame([x for x in [labels[i] for i in self.character_indices] if x[1] > character_threshold], columns=["tag", "score"]).sort_values("score", ascending=False)

      tags = pd.concat([ratings, gentags, chartags]).head(limit)

      yield dict(zip(tags.tag, tags.score))
