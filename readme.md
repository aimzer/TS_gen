# Time Series Generation

This project serves as a tensorflow project template complemented with an example of use.

## Template details

This template will simplify your research work developing new architectures or implementing research papers. It is adapted from [an open source Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template).

Its advantages are:
- Traceability: every experiment is saved with a configuration file, checkpoints and logs.
- Ease of use and debugging.
- Flexibility: creating a model that doesn't depend on input size allow can train on multiple datasets.
- Ease of training parallelisation.

The main components of the template are:

### Data Loader
- Inherits from BaseDataloader. 
- Preprocesses data for training.
- May include functions for loading batches and evaluating results. 

### Trainer
- Inherits from BaseTrainer. 
- Trains provided model on data, tracks losses and saves checkpoints.
- Can resume from checkpoint.

### Model
- Inherits from BaseModel. 
- Initialises placeholders, uses architectures to build the model, creates loss and training operations.
- Includes functions of inference, for saving and loading models.

### Architecture
- Defines neural network architectures as a hierarechy of blocks.

### Configuration file
Json files that specifies dataset, model and trainer parameters parameters seperately.

### Main file
Takes the configuration file as input, initializes main components and trains or tests the model on the data.


## Example Project

The goal of this project is to generate fixed length time series using GAN and WAE.

### Dataset

The dataset used for this experiment is UK's smartmeter energy consumption data for 5567 london households. Clients are seperated into different Acorn groups which will be used for conditioning the generation. The dataset is cleaned and the curves are segmented into weeks of 336 datapoints. Data is unskewed and scaled as a preprocessing for training and inference. The dataset is characterised by statistical indicators that will determine the quality of the generated data.

In order to test a training on the dataset it's necessary to [download the data](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households) and process it using the provided notebook.

### Models
The models tested for this task are:
- [WAE Wasserstein AutoEncoder](https://arxiv.org/abs/1711.01558v3): Adapted from the [this github repository](https://github.com/tolstikhin/wae/blob/master/wae.py)
- [CWGAN Conditional Wasserstein Generative Adversarial Networks](https://cameronfabbri.github.io/papers/conditionalWGAN.pdf): adapted from [this github repository](https://github.com/cameronfabbri/cWGANs)

### Running a training
```shell
python run_config.py -c configs/wgan_smartmeter.json
```

### Hyperparameter optimization
The script gan_skopt.py run a hyperparameter optimisation on GAN using bayesian optimisation. Training on multiple GPUs will speed up hyperparameter optimization.

### Requirements

This project run on tensorflow 1.13 and hyperparameter optimization requires the development version of scikit-optimize.

# Acknowledgements
The template is adapted from [Ahmkel](https://github.com/Ahmkel)'s [open source Keras Project Template](https://github.com/Ahmkel/Keras-Project-Template) and [MrGemy95](https://github.com/MrGemy95)'s [open source Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template).
