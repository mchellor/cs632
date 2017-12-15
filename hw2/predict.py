# coding: utf-8

import keras

from keras.models import load_model

model = load_model('cats_vs_dogs_32_32_3_model.h5')

import sys
import numpy as np
import random
import os

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

TEST_FILE = sys.argv[1]

data = np.load(TEST_FILE).item()

images = data["images"]
ids = data["labels"]

OUT_FILE = "predictions.txt"


out = open(OUT_FILE, "w")
for i, image in enumerate(images):

    image_id = ids[i]
    image = np.expand_dims(image, axis=0)

    prediction = model.predict_classes(image, verbose=0)

    line = str(image_id) + " " + str(prediction) + "\n"
    out.write(line)

out.close()