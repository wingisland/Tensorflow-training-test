# Classifying the images

In this document I'm going to walk you trough my thought process and how I build the model. Let's start with introduction of the problem. I'm working with a small-medium batch of natural images that showcase things like mountains, beaches, cities etc. We are going to try predicting classes of the images by building a CNN. In my case, I'm going to use Keras but you can of course use any other similar library. I have a few goals in mind while going into this project:

- I want to see how big of a difference in accuracy using a tuner can make
- I want to see if using strides instead of pooling really gives a better result, and if not, in which cases it would



## Importing the libraries

Before making the model, we first have to load all needed libraries (there is a requirements.txt file in the repository that you can use) as well as our images.

```python
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory as idfd
import matplotlib.pyplot as plot
```



_Note: I will be using the in-build Keras method for loading the images but it is possible to use it with vanilla python if you feel like it!_

```Python
# Load the images and their labels into a data set
train_ds = idfd(
    directory='data/seg_train/seg_train',
    labels='inferred',
    label_mode='int',
    seed=124
)

val_ds = idfd(
    directory='data/seg_test/seg_test',
    labels='inferred',
    label_mode='int',
    seed=124
)

class_names = train_ds.class_names
print('Classes: ', class_names)
```

```Python
>> Classes:  ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
```



## Exploring the images

Lovely, we know our classes now but I would also like to see how the images look like. Perhaps that insight could help us with creating the model later, who knows? Let's give it a try. I won't include the plots code here since I think it's not necessary to our storytelling, the plots themselves will provide enough of it.



![Images with their classes](plots\classes.png)



## Creating the base model

Now I'm going to create our basic model and evaluate its accuracy. Later I'm going to use this model with the Keras tuner to see if I can achieve more accuracy by tuning the hyperparameters as well as do some other tests with it. I'm going to use three conv2d layers, some pooling to reduce and then flatten the data. After that we only have our basic dense layers with a dropout. If you don't know what these layers are doing, let me explain it in short. I will also leave some links to resources by the end of this file.

- The **Conv2D layers** transform the input image into a very abstract representation, they basically take a group of x by x pixels and bring their values into one pixel with the average of all values
- The **MaxPooling2D layers** down samples the input representation, it takes the maximum value trough a window defined by pool_size
- The **Flatten layer** transforms our data into a one-dimensional one since the dense layer can accept it only in that form



Before defining the model, I also added a few lines of code that make sure we don't pollute the memory too much. I also made sure to standarize the data before training.

```Python
# Tune the data to not pollute the memory while training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardizing the data before training
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds.map(lambda x, y: (normalization_layer(x), y))
```

```Python
# Creating the model
model = Sequential()
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(140, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(100, kernel_size=(3, 3), activation='relu'))
model.add(layers.Conv2D(60, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(160, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Alright! We created our model so the next step will be the fun part - testing it. I created a few functions to visualise our accuracy measures so we can see it getting plotted nicely later. I also chose 20 epochs to start with.