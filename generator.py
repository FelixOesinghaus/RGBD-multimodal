# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
#from pyimagesearch.resnet import ResNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-a", "--augment", type=int, default=-1,
	help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


import glob



object_category_names = sorted(os.listdir('train'))
object_categories = {}
for i, name in enumerate(object_category_names):
    object_categories[i] = name



# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-1
BS = 8
EPOCHS = 5
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename, load the image, and
	# resize it to be a fixed 64x64 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

	
# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)	
	
aug = ImageDataGenerator()

if args["augment"] > 0:
	print("[INFO] performing 'on the fly' data augmentation")
	aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)

#model = ResNet.build(64, 64, 3, 2, (2, 3, 4),(32, 64, 128, 256), reg=0.0001)
num_classes = len(object_categories)
	# num_classes = 5
	#factor = 0.5
model = tf.keras.Sequential([
	#layers.experimental.preprocessing.Rescaling(1./255),
	#layers.Conv2D(16*factor, 3, activation='relu'),
	#layers.MaxPooling2D(),
	#layers.Activation('sigmoid'),
	layers.Conv2D(32, 3, activation='relu'),
	layers.MaxPooling2D(),
	#layers.Activation('sigmoid'),
	layers.Conv2D(64, 3, activation='relu'),
	#layers.Activation('sigmoid'),
	layers.MaxPooling2D(),
	layers.Conv2D(128, 3, activation='relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(256, 3, activation='relu'),
	layers.MaxPooling2D(),
	#layers.Dropout(.2),
	#layers.Activation('sigmoid'),
	#layers.Conv2D(128, 3, activation='relu'),
	#layers.Conv2D(512*factor, 3, activation='relu'),
	#layers.MaxPooling2D(),
	#layers.Activation('sigmoid'),
	layers.Flatten(),
	#layers.Flatten(),
	layers.Dense(1024, activation='relu', kernel_regularizer="l2"),
	layers.Dropout(0.30), 
	layers.Dense(1024, activation='relu'),
	layers.Dropout(0.10), 
	#layers.Softmax(),
	layers.Dense(num_classes)
])	

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)
	
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])