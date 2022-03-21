from tensorflow.keras import layers as layers
from tensorflow.keras import Sequential
import tensorflow.keras.optimizers as Optimizer
import kerastuner as kt
from kerastuner import HyperModel, RandomSearch
from main import train_ds, train_classes, getImages

class RegressionHyperModel(HyperModel):
    def build(self, hp):
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
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=100,
                                            max_value=180,
                                            step=40), activation='relu'))
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=80,
                                            max_value=100,
                                            step=10), activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(3, activation='softmax'))


        model.compile(optimizer=Optimizer.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

# Initialize the model
hypermodel = RegressionHyperModel()

# Run the random search
tuner_rs = RandomSearch(
            hypermodel,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=1)

tuner_rs.search_space_summary()
tuner_rs.search(train_ds, train_classes, batch_size=8, epochs=100, validation_split=0.30)

# Get the best model
best_model = tuner_rs.get_best_models(num_models=1)[0]

# Evaluate
val_ds, val_classes = getImages('../input/intel-image-classification/seg_test/seg_test/', 100)
best_model.evaluate(val_ds, val_classes, verbose=1)
