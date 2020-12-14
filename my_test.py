import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import os
import shutil
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

print("test")

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file(origin=dataset_url,
#                                   fname='flower_photos',
#                                   untar=True)

#print(os.getcwd())

#print(data_dir)

paths, dirs, files = next(os.walk(os.getcwd()))
print(dirs)
for directory in dirs:
    if "apple" in directory:
        applepath = os.getcwd() + "\\" + directory
        print(applepath)
        # tmpapplepath = applepath + "\\" + "*.png"
        # print(Path(tmpapplepath))
        # print(Path.cwd())
data_dir = pathlib.Path(applepath)

path, dirs, files = next(os.walk(applepath))
for directory in dirs:
    if "apple" in directory:
        subapplepath = applepath + "\\" + directory
        print("sub: " + subapplepath)
path, dirs, files = next(os.walk(subapplepath))
print("-------------------")
print(subapplepath)
print("-------------------")
applepics = []
for file in files:
    if "crop.png" in file:
        picpath = subapplepath + "\\" + file
        applepics.append(picpath)

print(len(applepics))
print(applepics[0])
#image_count = len(list(data_dir.glob('*/*.jpg')))
#print(image_count)

#apples = list(data_dir.glob(data_dir))
# PIL.Image.open(str(apples[0]))
#print(str(apples[0]))

os.mkdir(applepath + "\\crop")
os.mkdir(applepath + "\\crop\\apples")

print(applepics[0])
for pic in applepics:
    if "_crop.png" in pic:
        Path(pic).rename(pic.replace("apple_1", "crop\\apples\\", 1))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  Path(applepath + "\\crop"),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  Path(applepath + "\\crop"),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
count = 0
for images, labels in train_ds.take(1):
    if not count:
        #print(train_ds.)
        pass
    for i in range(9):
        # ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.show()
        # plt.title(class_names[labels[i]])
        plt.axis("off")
