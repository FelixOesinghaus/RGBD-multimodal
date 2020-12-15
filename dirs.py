import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import os
import shutil
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
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
# for images, labels in train_ds.take(1):
# 	for i in range(9):
# 		# ax = plt.subplot(3, 3, i + 1)
# 		plt.imshow(images[i].numpy().astype("uint8"))
# 		plt.title(class_names[labels[i]])
# 		plt.axis("off")
# 		plt.show()

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
# num_classes = 5

model = tf.keras.Sequential([
	layers.experimental.preprocessing.Rescaling(1./255),
	layers.Conv2D(32, 3, activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, activation='relu'),
	layers.MaxPooling2D(),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])

model.compile(
	optimizer='adam',
	loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=3
)

model.save(os.getcwd())

history = model.history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(3)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

