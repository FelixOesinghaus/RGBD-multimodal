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

paths, dirs, files = next(os.walk(os.getcwd()))
main_crop_path = paths + "\\crop"

os.mkdir(main_crop_path)

print("paths: ", paths)
print("dirs: ",dirs)
print("files: ", files)

for directory in dirs:
    if "rgbd-dataset" in directory:
        dataset_path = os.getcwd() + "\\" + directory
        print(dataset_path)

data_dir = pathlib.Path(dataset_path)

paths, dirs, files = next(os.walk(dataset_path))
print("paths: ", paths)
print("dirs: ",dirs)
print("files: ", files)


for directory in dirs:
	print(directory)
	os.mkdir(main_crop_path +"\\"+ directory)
	sub_path = paths + "\\" + directory
	print(sub_path)
	sub_data_dir = pathlib.Path(sub_path)
	
	subpaths, subdirs, subfiles = next(os.walk(sub_path)
	print("subpaths: ", subpaths)
	print("subdirs: ", subdirs)
	print("subfiles: ", subfiles)

	


