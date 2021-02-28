import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import os
import random
# import shutil
import PIL
import PIL.Image
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
# import tensorflow_datasets as tfds

print(tf.__version__)

def create_train_directory(paths, dirs, files):
	main_train_path = paths + "\\train"
	os.mkdir(main_train_path)
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
		#os.mkdir(main_train_path + "\\" + directory)
		sub_path = paths + "\\" + directory
		print(sub_path)
		sub_data_dir = pathlib.Path(sub_path)
		
		subpaths, subdirs, subfiles = next(os.walk(sub_path))
		print("subpaths: ", subpaths)
		print("subdirs: ", subdirs)
		#print("subfiles: ", subfiles)
		for subdirectory in subdirs:
			os.mkdir(main_train_path  + "\\" + subdirectory )
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
			new_path = "train\\" + subdirectory
			for pic in object_pics:
				if "_crop.png" in pic:
					Path(pic).rename(pic.replace(current_path, new_path, 1))

def create_test_directory(paths, dirs, files):
	main_train_path = paths + "\\train"
	main_test_path = paths + "\\test"
	os.mkdir(main_test_path)
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)

	paths, dirs, files = next(os.walk(main_train_path))
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)
	
	for directory in dirs:
		print(directory)
		os.mkdir(main_test_path + "\\" + directory)
		sub_path = paths + "\\" + directory
		print(sub_path)
		sub_data_dir = pathlib.Path(sub_path)
		
		subpaths, subdirs, subfiles = next(os.walk(sub_path))
		object_pics = []
		for file in subfiles:
			picpath = sub_path + "\\" + file
			object_pics.append(picpath)
		print(len(object_pics))
		ten_percent = int(len(object_pics)*0.1)
		current_path = "train\\" + directory 
		new_path = "test\\" + directory
		
		for i in random.sample(object_pics,ten_percent):
			Path(i).rename(i.replace(current_path, new_path, 1))
		
		
		
def create_val_directory(paths, dirs, files):
	main_train_path = paths + "\\train"
	main_val_path = paths + "\\validation"
	os.mkdir(main_val_path)
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)

	paths, dirs, files = next(os.walk(main_train_path))
	print("paths: ", paths)
	print("dirs: ", dirs)
	print("files: ", files)
	
	for directory in dirs:
		print(directory)
		os.mkdir(main_val_path + "\\" + directory)
		sub_path = paths + "\\" + directory
		print(sub_path)
		sub_data_dir = pathlib.Path(sub_path)
		
		subpaths, subdirs, subfiles = next(os.walk(sub_path))
		object_pics = []
		for file in subfiles:
			picpath = sub_path + "\\" + file
			object_pics.append(picpath)
		print(len(object_pics))
		twenty_percent = int(len(object_pics)*0.1)
		current_path = "train\\" + directory 
		new_path = "validation\\" + directory
		
		for i in random.sample(object_pics,twenty_percent):
			Path(i).rename(i.replace(current_path, new_path, 1))
		
		

paths, dirs, files = next(os.walk(os.getcwd()))
# if train does not exist, use create_train_directory
if Path.exists(Path(os.getcwd() + "\\train")):
	print("./train existiert bereits.")
else:
	create_train_directory(paths, dirs, files)


# if test does not exist, use create_test_directory
if Path.exists(Path(os.getcwd() + "\\test")):
	print("./test existiert bereits.")
else:
	create_test_directory(paths, dirs, files)


# if validation does not exist, use create_train_directory
#if Path.exists(Path(os.getcwd() + "\\validation")):
#	print("./validation existiert bereits.")
#else:
#	create_val_directory(paths, dirs, files)

