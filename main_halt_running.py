from tensorflow.keras import layers as layers
from tensorflow.python.keras.engine import base_layer_utils
import tensorflow as tf
from tensorflow.keras import Sequential
from sklearn.utils import shuffle
import tensorflow.keras.optimizers as Optimizer
import matplotlib.pyplot as plot
import numpy as np
import os , os.path
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing import image
from random import randint
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import time
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

#Control pannel
##############################################
epochtime = 100                    #epoch time
save = 0                           #save model
load = 0                           #load model
filepath = './saved_model'         #model path
load_filepath = '*path*'
 

##############################################
# Check the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
timenow=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
# Load the images and their labels into a data set
def getImages(dataset_dir, img_size):

    dataset_array = []
    dataset_labels = []
    class_counter = 0
    class_names = os.listdir(dataset_dir)

    for current_class_name in class_names:
        # Get class directory
        class_dir = os.path.join(dataset_dir, current_class_name)

        # Keep track of the class that is being extracted
        images_in_class = os.listdir(class_dir)
        print("Class index", class_counter, ", ", current_class_name, ":", len(images_in_class))
        for image_file in images_in_class:
            if image_file.endswith(".png"):
                image_file_dir = os.path.join(class_dir, image_file)
                img = tf.keras.preprocessing.image.load_img(image_file_dir, target_size=(img_size, img_size))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                dataset_array.append(img_array)
                dataset_labels.append(class_counter)

        # Increase the counter when we're done with a class
        class_counter += 1

    # Shuffle both lists the same way
    dataset_array, dataset_labels = shuffle(dataset_array, dataset_labels, random_state=817328462)

    # Transform to a numpy array
    dataset_array = np.array(dataset_array)
    dataset_labels = np.array(dataset_labels)
    return dataset_array, dataset_labels, class_names


# Get the training data set
train_ds, train_classes, class_names = getImages('./data/seg_train/seg_train/', 150)
print("Training Data Array Shape :", train_ds.shape)
print('Classes Shape: ', train_classes.shape)

'''
# Visualise the data we're working with
plot.figure(figsize=(10, 10))
for i in range(9):
    img = train_ds[i]
    img_label = class_names[train_classes[i]]

    # Create a subplot for the image on the canvas
    ax = plot.subplot(3, 3, i + 1)
    plot.imshow(img)
    plot.title(img_label)
    plot.axis('off')

'''
# Creating the model
model = Sequential()
model.add(layers.Conv2D(200, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(180, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(5, 5))
model.add(layers.Conv2D(180, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(140, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(50, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(5, 5))
model.add(layers.Flatten())
model.add(layers.Dense(180, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(3, activation='softmax'))
# add predictions layer

# Compile the model
model.compile(optimizer=Optimizer.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
trained = model.fit(train_ds, train_classes, batch_size=8, epochs=epochtime, validation_split=0.30)
model.summary()

# # Save the model
if save == 1:
    save_model(model, f"{filepath}_{timenow}")

# Visualise the model accuracy
acc = trained.history['accuracy']
val_acc = trained.history['val_accuracy']
loss = trained.history['loss']
val_loss = trained.history['val_loss']
epochs_range = range(epochtime)


# Create the graphs
plot.figure(figsize=(10, 10))
plot.subplot(1, 2, 1)
plot.title('Training & Validation Acc.')
plot.plot(epochs_range, acc, label='Training Acc.')
plot.plot(epochs_range, val_acc, label='Validation Acc.')
plot.legend(['Train', 'Test'], loc='lower right')

plot.subplot(1, 2, 2)
plot.title('Training & Validation Loss')
plot.plot(epochs_range, loss, label='Training Loss')
plot.plot(epochs_range, val_loss, label='Validation Loss')
plot.legend(['Train', 'Test'], loc='upper right')

# ..and show the graphs
plot.show()

# Load the model
if load == 1 :
    loaded_model = load_model(load_filepath, compile = True)

# get prdeict image
pred_ds, pred_classes, pred_class_names  = getImages('./data/seg_pred/',150)
DIR = '/home/karma/git_home/intel-image-classification.git/data/seg_pred/seg_pred'
Predition_data_count=len([chunk for chunk in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, chunk))])
print ("Predition data count",Predition_data_count)

good_data_count     = 0
bad_data_count      = 0
off_data_count      = 0
invalid_value_count = 0
pred_list           = []

for i in range(Predition_data_count):
    pred_img =np.array([pred_ds[i]])
    pred_prob = model.predict(pred_img)
    processed_pred_prob = pred_prob[0]
    first_index  = processed_pred_prob[0]
    second_index = processed_pred_prob[1]
    third_index  = processed_pred_prob[2]
    if  first_index  > 0.7:
        good_data_count     += 1
        pred_list.append("Running")
    elif second_index > 0.7:
        bad_data_count      += 1
        pred_list.append("Halt")
    elif third_index  > 0.7:
        off_data_count      += 1
        pred_list.append("No_input")
    else:
        invalid_value_count += 1
        pred_list.append("invalid_value")
print("good size",good_data_count)
print("bad size" ,bad_data_count)
print("off size" ,off_data_count)
print("invalid value size" ,invalid_value_count)
valid_value    = good_data_count + bad_data_count + off_data_count
invalid_value  = invalid_value_count

find_max = max(good_data_count,bad_data_count,off_data_count)
precision = find_max / Predition_data_count
precision_round_number = round(precision, 2)
print("The precision is :", precision_round_number*100," % ")

all_count      = [good_data_count,bad_data_count,off_data_count,invalid_value]
outer = gridspec.GridSpec(10, 10, wspace=1, hspace=1)
labels = 'Running', 'Halt', 'No_input', 'Invalid value'
labels_pos =[0,1,2,3]
x = np.array(labels_pos)
y = np.array(all_count)
plot.bar(x,y)
plot.title(f"The precision is {precision_round_number*100} % ")
plot.xticks(labels_pos,labels)
plot.show()

'''
    # Create a subplot for the image on the canvas
    ax = plot.subplot(math.ceil(math.sqrt(Predition_data_count)), math.ceil(math.sqrt(Predition_data_count)), i + 1)
    ax.imshow(pred_img.squeeze())
    plot.axis('off') 
plot.show()

'''


'''
fig = plot.figure(figsize=(25, 25))
outer = gridspec.GridSpec(10, 10, wspace=1, hspace=1)
# get prdeict image
pred_ds, pred_classes, pred_class_names  = getImages('./data/seg_pred/',150)

for i in range(Predition_data_count):
    pred_img =np.array([pred_ds[i]])
    pred_prob = model.predict(pred_img)
    conpro = np.concatenate(pred_prob) 
    labels = 'Good', 'Off', 'Bad'
    labels_pos =[0,1,2]
    x = np.array(labels_pos)
    y = np.array(conpro)
    plot.subplot(math.ceil(math.sqrt(Predition_data_count)), math.ceil(math.sqrt(Predition_data_count)), i + 1)
    plot.bar(x,y)
    plot.xticks(labels_pos,labels)
    anser= ('The '+ repr(i+1)+' picture')
    plot.title(anser) 
plot.show()

'''

'''
find_max = max(good_data_count,bad_data_count,off_data_count)
precision = find_max / Predition_data_count
print("The precision is :", precision*100," % ")
'''
'''
def Confusion_matrix_flag():
	if find_max == good_data_count:
		return "good"
	if find_max == bad_data_count:
                return "bad"
	if find_max == off_data_count:
                return "off"
flag = Confusion_matrix_flag()
flag_list = [flag]*Predition_data_count

y_true = flag_list
y_pred = pred_list
conmatrix = confusion_matrix(y_true, y_pred, labels=["good", "bad", "off","invalid_value"])
print(conmatrix)
df_cm = pd.DataFrame(conmatrix, index = [i for i in labels],columns = [i for i in labels])
plot.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
plot.xlabel("Predicted Class")
plot.ylabel("True Class")
plot.show()
'''
