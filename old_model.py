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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import tensorflow_datasets as tfds
import seaborn as sns
import glob
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
object_category_names = sorted(os.listdir('train'))
object_categories = {}
for i, name in enumerate(object_category_names):
    object_categories[i] = name


#result = object_categories.items() 
#data = list(result) 

#object_labels_two = np.array(data)
object_labels_two = np.delete(np.array(list(object_categories.items())),0,1)
object_labels = object_labels_two.reshape(len(object_categories),)

print(type(object_categories))
print(object_categories)
print()
print(len(object_categories))

#print(type(object_labels_two))
#print(object_labels_two)
#print(object_labels_two.shape)




print(object_labels)
print(object_labels.shape)


#sys.exit()





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
                ax[r, c].imshow(pil_image.astype('uint8'))
                
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



#args = -1
batch_size = 128
img_height = 200
img_width = 200

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\train"),
	labels="inferred",
	color_mode='grayscale',
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\train"),
	labels="inferred",
	color_mode='grayscale',
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
	Path(os.getcwd() + "\\test"),
	labels="inferred",
	color_mode='grayscale',
	seed=123,
	shuffle=True,
	image_size=(img_height, img_width),
	batch_size=batch_size)

	
#train_elements = 168457
#train_elements = 170326
#train_elements = 136261
#val_elements = 34065
#test_elements = 20767
#test_elements = 20767

#train_steps = train_elements // batch_size
#validation_steps = val_elements // batch_size
#test_steps = test_elements // batch_size

train_steps = int(train_ds.cardinality() - 1)
validation_steps = int(val_ds.cardinality() - 1)
test_steps = int(test_ds.cardinality() - 1)

print('Training set size %d, step-size %d' % (train_steps*batch_size, train_steps))
print('Validation set size %d, step-size %d' % (validation_steps*batch_size, validation_steps))
print('Test set size %d, step-size %d' % (test_steps*batch_size , test_steps))
class_names = train_ds.class_names	
train_ds = train_ds.repeat()
val_ds = val_ds.repeat()



normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

images, labels = iter(normalized_train_ds).next()
#print(labels)
nd_labels = labels.numpy()
nd_images = images.numpy()
#labels = np.argmax(labels, axis=1)

display_sample(nd_images[:16], nd_labels[:16], num_rows=4, num_cols=4, plot_title='Sample Training Data', fig_size=(16,20))

if Path.exists(Path(os.getcwd() + "\\saved_model")):
	print("./saved_model existiert bereits.")
	# +"\\saved_model_0"+str(args)+".pb"
	#Path(os.getcwd()+"\\saved-models\\saved_model_0"+str(args)+".pb").\
	#	rename(os.getcwd()+"\\saved_model.pb")
	#model = keras.models.load_model(Path(os.getcwd()))
	# model.load_weights()
	#Path(os.getcwd()+"\\saved_model.pb").\
	#	rename(os.getcwd()+"\\saved-models\\saved_model_0"+str(args)+".pb")
	#if Path(os.getcwd()+"\\saved-models\\saved_model.pb").exists():
	#	os.remove(os.getcwd()+"\\saved-models\\saved_model.pb")
	model = tf.keras.models.load_model('Q:/rgbd-data/RGBD multimodal/saved_model')

	print("Modell geladen.")



else:


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
	
	print(class_names)
	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

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

	num_classes = len(class_names)
	# num_classes = 5
	#factor = 0.5
	#regularizer = contrib.layers.l2_regularizer(scale=0.1)
	model = tf.keras.Sequential([
		layers.experimental.preprocessing.Rescaling(1./255),
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
		layers.Dropout(0.10), 
		layers.Dense(1024, activation='relu', kernel_regularizer="l2"),
		layers.MaxPooling2D(),
		layers.Dropout(0.10), 
		#layers.Dropout(.2),
		#layers.Activation('sigmoid'),
		#layers.Conv2D(128, 3, activation='relu'),
		#layers.Conv2D(512*factor, 3, activation='relu'),
		#layers.MaxPooling2D(),
		#layers.Activation('sigmoid'),
		layers.Flatten(),
		#layers.Flatten(),
		layers.Dense(1024, activation='relu', kernel_regularizer="l2"),
		layers.Dropout(0.10), 
		#layers.Softmax(),
		layers.Dense(num_classes)
	])
	
	model.compile(
		optimizer='adam',
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])

	epochs = 15
	#steps_per_epoch = 10

	# if epochs * steps_per_epoch > 138 * 400:
	# 	print("Zu viele Epochenschritte für zu wenige Daten!", file=sys.stderr)
	# 	exit(-1)
	#

	faster_steps = train_steps // 4
	csv_logger = CSVLogger('training.log', separator=',', append=False)
	#csv_logger_test = CSVLogger('test.log', separator=',', append=False)
	
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		callbacks=[csv_logger],
		epochs=epochs,
		steps_per_epoch=faster_steps,
		validation_steps=validation_steps
		
		
	)

	#steps_per_epoch= 1000,,
	#use_multiprocessing=True
	#validation_steps= 1000
	#validation_steps=25
	model.save(
		'Q:/rgbd-data/RGBD multimodal/saved_model',
		overwrite=True,
		include_optimizer=True,
		save_format=None,
		signatures=None,
		options=None,
		save_traces=True,
	)
	
if Path.exists(Path(os.getcwd() + "\\saved_model")):
	log_data = pd.read_csv('training.log', sep=',', engine='python')
	history = log_data
	acc = history['accuracy']
	val_acc = history['val_accuracy']
	loss = history['loss']
	val_loss = history['val_loss']

else:
	history = model.history

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

# acc = model.history['accuracy']
# val_acc = model.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	

# loss = model.history['loss']
# val_loss = model.history['val_loss']

# epochs_range = range(epochs)


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.summary()
plt.clf()
#images, labels = iter(test_ds).next()
#predict_labels = model.predict(images)
#predict_labels = np.argmax(predict_labels, axis=1)

#nd_labels = labels.numpy()
#nd_images = images.numpy()
#wrong_labels = 0
#for i in range(len(nd_labels)):
#	if nd_labels[i] != predict_labels[i]:
#		wrong_labels +=1
	
#print("\n Amount of wrong labels :",wrong_labels)
#print(predict_labels)

#print("\n Actual label index:\n", nd_labels)
#print("\n Predicted label index:\n", predict_labels)
all_predictions, all_labels = [], []
#all_predictions = np.empty(test_steps * batch_size, dtype=np.int)
#all_labels = np.empty(test_steps * batch_size, dtype=np.int)
	

print("Total test batches to predict: ",test_steps)

for i in range(test_steps):
	print("Testing batch nr:",i,"out of",test_steps)
	images, labels = iter(test_ds).next()
	predict_labels = model.predict(images)
	predict_labels = np.argmax(predict_labels, axis=1)
	nd_labels = labels.numpy()
	nd_images = images.numpy()

	all_predictions.extend(predict_labels)
	all_labels.extend(nd_labels)


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
#np.savetxt("test_labels.csv", all_labels, delimiter=',')test_steps
#np.savetxt("test_predictions.csv", all_predictions, delimiter=',')

def int_label_to_string(x):
  return object_labels[x]



print("\n Confusion Matrix:")
cm = confusion_matrix(int_label_to_string(all_labels),int_label_to_string(all_predictions), labels=object_labels,normalize='true')
np.savetxt("cm.csv", cm, delimiter=",")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=object_labels)
#plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation='vertical') 
for labels in disp.text_.ravel():
    labels.set_fontsize(6)

#plt.savefig('confusion_matrix.png',dpi=300)
#plt.show()


#number_of_samples = np.zeros((len(object_categories),), dtype=np.uint64)
#wrong_labels = np.zeros((len(object_categories),), dtype=np.uint64)
#accuracies = np.zeros((len(object_categories),), dtype=np.uint64)
#for i in range(len(all_labels)):
#	number_of_samples[all_labels[i]] += 1
#	if all_labels[i] != all_predictions[i]:
#		wrong_labels[all_labels[i]] +=1


#print("Per object accuracy:")
for i in range(len(object_categories)):
#	correct_amount = number_of_samples[i] - wrong_labels[i]
#	accuracies[i] = correct_amount / number_of_samples[i]
#	print(object_categories[i]," accuracy: ", correct_amount / number_of_samples[i])
	print(object_categories[i]," accuracy: ", cm[i,i])



#print("Most mismatches :",object_categories[np.argmax(wrong_labels)])
#print('First 25 results:')
#print('  - Actuals    : ', all_labels[:25])
#print('  - Predictions: ', all_predictions[:25])
correct_pred_count = (all_labels == all_predictions).sum()
test_acc = correct_pred_count / len(all_labels)
print('\n We got %d of %d correct (or %.3f accuracy)' % (correct_pred_count, len(all_labels), test_acc))
#loss, acc = model.evaluate(test_ds, steps=test_steps, verbose=1)
#print('Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))


images, labels = iter(test_ds).next()

#print(labels)
nd_labels = labels.numpy()
nd_images = images.numpy()
#labels = np.argmax(labels, axis=1)

#display_sample(nd_images[:16], nd_labels[:16], num_rows=4, num_cols=4, plot_title='Sample Training Data', fig_size=(16,20))
predict_labels = model.predict(images)
predict_labels = np.argmax(predict_labels, axis=1)

#print("\n Actual label index:\n", nd_labels)
#print("\n Predicted label index:\n", predict_labels)

# display sample predictions

display_sample(nd_images[:64], nd_labels[:64], sample_predictions=predict_labels[:64], num_rows=8, num_cols=8, 
               plot_title='Sample Predictions on Test Dataset', fig_size=(18,20))



#normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
#images, labels = iter(normalized_test_ds).next()
#images, labels = iter(normalized_test_ds).next()

#score = model.evaluate(test_ds,verbose=1)
#print(score)
print()
loss, acc = model.evaluate(train_ds, steps=train_steps, verbose=1)
print('Training data  -> loss: %.3f, acc: %.3f' % (loss, acc))
loss, acc = model.evaluate(val_ds, steps=validation_steps, verbose=1)
print('Cross-val data -> loss: %.3f, acc: %.3f' % (loss, acc))
loss, acc = model.evaluate(test_ds, steps=test_steps, verbose=1)
print('Testing data   -> loss: %.3f, acc: %.3f' % (loss, acc))


#labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#ones = np.ones(len(object_categories))




