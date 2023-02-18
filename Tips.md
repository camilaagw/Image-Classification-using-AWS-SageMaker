# Tips to save your credit and time when doing projects 3 and 4
 If you restart your development notebook kernel, instead of re-running the costly training jobs or hyperparameter tuning jobs from scratch you can load (attach) completed jobs to a variable.
For training jobs, use this code to attach a complted training job:
`estimator = PyTorch.attach(<training_job_name>)`

For hyperparameter tuning jobs:
`tuner = HyperparameterTuner.attach(<tuning_job_name>)`

If you have an already deployed endpoint and you want to use it, you can attach it using the following code:
`predictor = sagemaker.pytorch.model.PyTorchPredictor(endpoint_name)`

Instead of launching a training job with a training script, you can develop your train_model.py script locally inside a jupyter notebook. You can simulate the training instance environment variables and the command line arguments using the following code:
```python
import sys
import os


# Simulate passing hayperparameters as environment variables
sys.argv = [
        "script.py", 
        "--batch-size","32",
        "--learning-rate","0.001", 
        "--epochs", "1", 
       ]

# Simulate passing data channels and model saving path as environmet variables:
os.environ['SM_CHANNEL_TRAIN'] = "./dogImages/train"
os.environ['SM_CHANNEL_VALID'] = "./dogImages/valid"
os.environ['SM_CHANNEL_TEST'] = "./dogImages/test"
os.environ['SM_MODEL_DIR'] = "/opt/ml/model"


# Crate a model directory and take ownership 
!sudo mkdir /opt/ml/model
!sudo chown -R 1000:1000 /opt/ml/model
```

For the starting phase of the creating your project, you just want to focus on the correctness of your code. Use a subset of the dataset to speed your training and testing. You can use a break statement inside the training or testing loop. For example:
```
for step, (inputs, labels) in enumerate(data_loader):
    ...
    dataset_length = len(data_loader.dataset)
    running_samples = (step + 1) * len(inputs)
    proportion = 0.2 # we will use 20% of the dataset
    if running_samples>(proportion*dataset_length):
        break
```