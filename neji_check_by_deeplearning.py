# coding:utf-8
import neji_keras3_use_mizumashi as nejikeras
import sys, os
from PIL import Image
import numpy as np

if len(sys.argv) <= 1:
	print ("neji_check_by_deepl.py (filename_please)")
	quit()

image_size = 50
categories = ["large", "mini"] #large, miniの並び方重要


X = []
files = []
for filename in sys.argv[1:]:
	img = Image.open(filename)
	img = img.convert("RGB")
	img = img.resize((image_size,image_size))
	in_data = np.asarray(img)
	X.append(in_data)
	files.append(filename)
X = np.array(X)

model = nejikeras.build_model(X.shape[1:])
model.load_weights("neji_model.hdf5")

pre = model.predict(X)
for i,p in enumerate(pre):
	y = p.argmax()
	print(files[i])
	print(categories[y])