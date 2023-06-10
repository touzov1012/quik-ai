# Quik-AI
Quick Unifying Infrastructure Kit for Machine Learning and AI.

An easy-to-use interface for building deep learning models in TensorFlow. Supports modern neural network architectures as well as automatic hyperparameter tuning to ease the time to production for machine learning tasks. Build regression, classification, and density estimation models on mixed input data using only a few simple commands.

## Usage
1. Create the data driver for all processes involving passing data through our model.
```python
import quik_ai as qa

# Feed your pandas dataframes.
driver = qa.Driver(training_data, validation_data, test_data)
```

2. Specify the objective of our deep learning task.
```python
# We aim to fit a gaussian mixture to the response.
head = qa.heads.GaussianMixture()
```

3. Specify the predictors used for model fitting.
```python
# All the predictors for the model.
predictors = [
    qa.predictors.NumericalPredictor('numerical'),
    qa.predictors.CategoricalPredictor('categorical'),
    qa.predictors.PeriodicPredictor('periodic', 24.0),
]
```

4. Select the model architecture.
```python
# Select the architecture and feed the response column, and other
# necessary data for training, testing the model.
model = qa.models.Transformer('response', head, predictors, driver)
```

5. Train the model.
```python
# Train the model, any parameters which are not specified in the
# above predictors, model, driver, and head are automatically tuned
# using the specified hyperparameter optimizer. We default to use
# a custom implementation of the BOHB tuner.
model.train()
```

For additional options and tutorials see the samples in the `./samples` folder.
