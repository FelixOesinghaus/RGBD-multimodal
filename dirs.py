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

#paths, dirs, files = next(os.walk(os.getcwd()))

#paths, dirs, files = os.walk(os.getcwd()+ "\\" +"rgbd-dataset")

def create_crop_directory(paths, dirs, files):
	main_crop_path = paths + "\\crop"
	os.mkdir(main_crop_path)
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)

	for directory in dirs:
		if "rgbd-dataset" in directory:
			dataset_path = os.getcwd() + "\\" + directory
			print(dataset_path)

	data_dir = pathlib.Path(dataset_path)

	paths, dirs, files = next(os.walk(dataset_path))
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)

	for directory in dirs:
		print(directory)
		os.mkdir(main_crop_path + "\\" + directory)
		sub_path = paths + "\\" + directory
		print(sub_path)
		sub_data_dir = pathlib.Path(sub_path)
		
		subpaths, subdirs, subfiles = next(os.walk(sub_path))
		print("subpaths: ", subpaths)
		print("subdirs: ", subdirs)
		#print("subfiles: ", subfiles)
		for subdirectory in subdirs:
			sub_sub_path = sub_path + "\\" + subdirectory
			print(sub_sub_path)
			objectpaths, objectdirs, objectfiles = next(os.walk(sub_sub_path))
			
			object_pics = []
			for file in objectfiles:
				if "_crop.png" in file:
					picpath = sub_sub_path + "\\" + file
					object_pics.append(picpath)
			print(len(object_pics))
			current_path = "rgbd-dataset\\" + directory + "\\" + subdirectory
			new_path = "crop\\" + directory
			for pic in object_pics:
				if "_crop.png" in pic:
					Path(pic).rename(pic.replace(current_path, new_path, 1))


paths, dirs, files = next(os.walk(os.getcwd()))
# if crop does not exist, use create_crop_directory
if Path.exists(Path(os.getcwd() + "\\crop")):
	print("./crop existiert bereits.")
else:
	create_crop_directory(paths, dirs, files)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\crop"),
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\crop"),
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

class_names = train_ds.class_names

# plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
	for i in range(9):
		# ax = plt.subplot(3, 3, i + 1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[labels[i]])
		plt.axis("off")
		plt.show()
