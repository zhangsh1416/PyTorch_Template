# PyTorch Neural Network Training Pipeline

This project provides a modular and flexible pipeline for training neural networks using PyTorch. It includes data preprocessing, model training, evaluation, and logging functionalities. The pipeline is highly configurable through a YAML configuration file.

## Features

- **Modular Code Structure**: Separate modules for data loading, model definition, and training.
- **Data Augmentation**: Common augmentation techniques for improving model generalization.
- **Early Stopping and Learning Rate Scheduler**: Configurable early stopping and learning rate scheduling.
- **Model Checkpointing**: Save model checkpoints at specified intervals.
- **Evaluation Metrics**: Support for various evaluation metrics including accuracy and F1 score.
- **Logging and Visualization**: Use TensorBoard and `tqdm` for logging and visualizing training progress.
- **YAML Configuration**: Centralized configuration management using a YAML file.

## Installation

### Prerequisites

- Python 3.7 or above
- `pip` package manager

### Steps

1. **Clone the repository**:
    ```sh
    git clone git@github.com:zhangsh1416/PyTorch_Template.git
    cd your-repo
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate  # On Windows
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

All configurable parameters are located in the `config.yaml` file. You can adjust the dataset path, batch size, number of epochs, learning rate, and other hyperparameters as needed.

### Training

Run the training script to start training the model:
    ```sh
    python train.py
    ```

This script will:
- Load and preprocess the data.
- Initialize and test the model.
- Train the model using the specified configurations.
- Save model checkpoints.
- Log training progress and metrics.

### Evaluation

After training, you can evaluate the model's performance using the test dataset.

### Project Structure

    ```plaintext
    .
    ├── checkpoints
    ├── config.yaml
    ├── data_utils.py
    ├── models
    │   └── model.py
    ├── requirements.txt
    ├── train.py
    ├── test_data.py
    ├── test_model.py
    ├── utils.py
    └── logs
    ```

### Example Configuration (config.yaml)

    ```yaml
    dataset_path: './data'
    batch_size: 64
    validation_split: 0.2
    num_epochs: 10
    learning_rate: 0.01
    log_filename: 'logs/training.log'
    early_stopping: true
    lr_scheduler: true
    save_checkpoint: 5
    data_augmentation: true
    evaluate_accuracy: true
    evaluate_f1: false
    tensorboard_logging: true
    ```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
