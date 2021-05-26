# Tuning the model

When it comes to tuning this model, we can go many different ways. For the personal research purposes I'm going to test three different ways that I picked up from Kaggle competitions so far. We're going to use pretrained models as well as tune them to get the highest possible accuracy. I did some reading on notebooks that other people created for the same dataset and the accuracy usually comes down to 86 - 88%. After looking at the numbers and ways people used to get them, I think we can at most bet on 89-90%. That being said, I won't give up on looking for ways to make it higher in the future.



## 1) Using Keras tuner

Keras tuner is a tool that allows us to easily tune the models, it works similarly to grid search but we can also use different ways of searching for the best parameters. In our case I'm going to use the random search.

```Python
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
                                        step=20), activation='relu'))
        model.add(layers.Dense(units=hp.Int('units',
                                        min_value=80,
                                        max_value=100,
                                        step=10), activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(6, activation='softmax'))
        
        
        model.compile(optimizer=Optimizer.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    
# Initialize the model
hypermodel = RegressionHyperModel()
```

```Python
# Run the random search
tuner_rs = RandomSearch(
            hypermodel,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=3)

tuner_rs.search_space_summary()
tuner_rs.search(train_ds, train_classes, epochs=20, validation_split=0.3)
```

Now it's time to see what kind of results we got from the summary and runtime information that printed out. I went for a little bit less epochs here because I thought it might take some time to build and evaluate everything I wanted to check. If you have a bit more time, feel free to experiment with other parameters too.

```python
# Here we have the summary of what we've build so far
Search space summary
Default search space size: 2
units (Int)
{'default': 100, 'conditions': [], 'min_value': 80, 'max_value': 100, 'step': 1, 'sampling': None}
learning_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}
```

```
Best val_accuracy So Far: 0.8330562710762024
Total elapsed time: 00h 53m 35s
```



## 2) Using VGG

*To be finished.*



## 3) VGG model tuning

*To be finished.*

