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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
# import tensorflow_datasets as tfds

import seaborn as sns
import glob


#args = -1
batch_size = 128
img_height = 200
img_width = 200


object_category_names = sorted(os.listdir('train'))
object_categories = {}
for i, name in enumerate(object_category_names):
    object_categories[i] = name



def display_sample(sample_images, sample_labels, sample_predictions=None, num_rows=8, num_cols=8,
                   plot_title=None, fig_size=None):
    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction 
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    from PIL import Image
    import seaborn as sns
    assert len(sample_images) == num_rows * num_cols

    # a dict to help encode/decode the labels
    global object_categories
    
    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((14, 12) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.02, "hspace": 0.25}, squeeze=True)
        #fig = ax[0].get_figure()
        f.tight_layout()
        f.subplots_adjust(top=0.93)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                pil_image = sample_images[image_index] #Image.fromarray(sample_images[image_index])
                ax[r, c].imshow(pil_image)
                
                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%s" % object_categories[sample_labels[image_index]])
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_label = sample_predictions[image_index]
                    prediction_matches_true = (sample_labels[image_index] == sample_predictions[image_index])
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = object_categories[true_label]
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = '%s/%s' % (object_categories[true_label][:5], object_categories[pred_label][:5])
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()



train_datagen = ImageDataGenerator(
	rescale=1.0/255,
	height_shift_range=0.2,
	width_shift_range=0.2,
	rotation_range=20,
	horizontal_flip=True,
	fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
	'train',
	target_size=(img_height,img_width),
	batch_size=batch_size,
	shuffle=True,
	seed=123,
	class_mode='categorical')
	
validation_generator = validation_datagen.flow_from_directory(
	'test',
	target_size=(img_height,img_width),
	batch_size=batch_size,
	shuffle=True,
	seed=123,
	class_mode='categorical')	
	
test_generator = validation_datagen.flow_from_directory(
	'test',
	target_size=(img_height,img_width),
	batch_size=batch_size,
	shuffle=False,
	seed=123,
	class_mode=None)	


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\validation"),
	labels="inferred",
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

#val_ds_steps = int(val_ds.cardinality() - 1)
#normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

#normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#normalized_val_ds = normalized_val_ds.repeat()

train_steps = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size
test_steps = test_generator.n // batch_size

print('Training set size %d, step-size %d' % (train_generator.n, train_steps))
print('Validation set size %d, step-size %d' % (validation_generator.n, validation_steps))
print('Test set size %d, step-size %d' % (test_generator.n, test_steps))

#images, labels = train_generator.next()
#print(labels)
#print(images)
#labels = np.argmax(labels, axis=1)
#print(labels)
#display_sample(images[:16], labels[:16], num_rows=4, num_cols=4, plot_title='Sample Training Data', fig_size=(16,20))

if Path.exists(Path(os.getcwd() + "\\generator_saved_model")):
	print("./generator_saved_model existiert bereits.")
	# +"\\generator_saved_model_0"+str(args)+".pb"
	#Path(os.getcwd()+"\\saved-models\\generator_saved_model_0"+str(args)+".pb").\
	#	rename(os.getcwd()+"\\generator_saved_model.pb")
	#model = keras.models.load_model(Path(os.getcwd()))
	# model.load_weights()
	#Path(os.getcwd()+"\\generator_saved_model.pb").\
	#	rename(os.getcwd()+"\\saved-models\\generator_saved_model_0"+str(args)+".pb")
	#if Path(os.getcwd()+"\\saved-models\\generator_saved_model.pb").exists():
	#	os.remove(os.getcwd()+"\\saved-models\\generator_saved_model.pb")
	model = tf.keras.models.load_model('Q:/rgbd-data/RGBD multimodal/generator_saved_model')

	print("Modell geladen.")
		

else:

	
	#images, labels = train_generator.next()
	#labels = np.argmax(labels, axis=1)
	#display_sample(images[:64], labels[:64], num_rows=8, num_cols=8, plot_title='Sample Training Data', fig_size=(16,20))

	
	#num_elements = 0
	#for element in packed_ds:
	#	num_elements += 1
		
	#class_names = train_ds.class_names
	#N_VALIDATION = int(num_elements * 0.2)
	#N_TRAIN = int(num_elements * 0.8)
	#print("\n",train_ds.__len__(),"\n")
	#val_ds = packed_ds.take(N_VALIDATION).repeat()
	#train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).repeat()
	
	# plt.figure(figsize=(10, 10))
	# for images, labels in train_ds.take(1):
	# 	for i in range(9):
	# 		# ax = plt.subplot(3, 3, i + 1)
	# 		plt.imshow(images[i].numpy().astype("uint8"))
	# 		plt.title(class_names[labels[i]])
	# 		plt.axis("off")
	# 		plt.show()
	
	print(object_categories)
	#normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

	#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	#image_batch, labels_batch = next(iter(normalized_ds))
	#first_image = image_batch[0]
	# Notice the pixels values are now in `[0,1]`.
	# print(np.min(first_image), np.max(first_image))
	#
	# AUTOTUNE = tf.data.experimental.AUTOTUNE
	#
	# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
	#train_ds.repeat()
	#val_ds.repeat()
	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	epochs = 6
	
	faster_steps = train_steps // 4
	
	#csv_logger = CSVLogger('training.log', separator=',', append=False)
	#csv_logger_test = CSVLogger('test.log', separator=',', append=False)
	#validation_data=validation_generator,
	model.fit_generator(
		train_generator,
		steps_per_epoch=faster_steps,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=validation_steps
	)

	model.save(
		'Q:/rgbd-data/RGBD multimodal/generator_saved_model',
		overwrite=True,
		include_optimizer=True,
		save_format=None,
		signatures=None,
		options=None,
		save_traces=True,
	)
model.summary()

loss, acc = model.evaluate_generator(validation_generator,
                                     steps=math.ceil(validation_generator.samples / batch_size),
                                     verbose=0,
                                     workers=1)

y_pred = model.predict_generator(validation_generator,
                                 steps=math.ceil(validation_generator.samples / batch_size),
                                 verbose=0,
                                 workers=1)

y_pred = np.argmax(y_pred, axis=-1)
y_test = validation_generator.classes[validation_generator.index_array]

print('loss: ', loss, 'accuracy: ', acc) # loss:  0.47286026436090467 accuracy:  0.864
print('accuracy_score: ', accuracy_score(y_test, y_pred)) # accuracy_score:  0.095
	
#loss, acc = model.evaluate_generator(train_generator, steps=train_steps, verbose=1)
#print('Training data  -> loss: %.3f, acc: %.3f' % (loss, acc))
#loss, acc = model.evaluate_generator(validation_generator, steps=validation_steps, verbose=1)
#print('Cross-val data -> loss: %.3f, acc: %.3f' % (loss, acc))
#loss, acc = model.evaluate_generator(test_generator, steps=test_steps, verbose=1)
#print('Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))

all_predictions, all_labels = [], []
for i in range(test_steps):
    images, labels = test_generator.next()
    labels = np.argmax(labels, axis=1)
    y_pred = np.argmax(model.predict(images), axis=1)
    all_predictions.extend(y_pred.astype('int32'))
    all_labels.extend(labels.astype('int32'))

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

print('First 25 results:')
print('  - Actuals    : ', all_labels[:25])
print('  - Predictions: ', all_predictions[:25])
correct_pred_count = (all_labels == all_predictions).sum()
test_acc = correct_pred_count / len(all_labels)
print('We got %d of %d correct (or %.3f accuracy)' % (correct_pred_count, len(all_labels), test_acc))

# display sample predictions
images, labels = test_generator.next()
labels = np.argmax(labels, axis=1)
predictions = np.argmax(model.predict(images), axis=1)
display_sample(images[:64], labels[:64], sample_predictions=predictions[:64], num_rows=8, num_cols=8, 
               plot_title='Sample Predictions on Test Dataset', fig_size=(18,20))



