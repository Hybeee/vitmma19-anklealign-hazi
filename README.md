# vitmma19-anklealign-hazi

## Submission Instructions

[Delete this entire section after reading and following the instructions.]

### Project Levels

**Basic Level (for signature)**
*   Containerization
*   Data acquisition and analysis
*   Data preparation
*   Baseline (reference) model
*   Model development
*   Basic evaluation

**Outstanding Level (aiming for +1 mark)**
*   Containerization
*   Data acquisition and analysis
*   Data cleansing and preparation
*   Defining evaluation criteria
*   Baseline (reference) model
*   Incremental model development
*   Advanced evaluation
*   ML as a service (backend) with GUI frontend
*   Creative ideas, well-developed solutions, and exceptional performance can also earn an extra grade (+1 mark).

### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [ ] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Szaniszló Csongor Ádám
- **Aiming for +1 Mark**: Yes

### Solution Description

The current solution solves a computer vision problem, where the goal is to correctly classify the input image into one of three categories: Neutral, Pronation and Supination. These classifications can be made by observing the ankle's alignment on the image. If it leans outwards the correct label is supination. If it leans inwards the correct label is pronation. Otherwise the correct label is neutral.

The implementation includes 5 stages; the first is data preparation which is implemented in `src/data_pipeline/data_preparing.py`. This consists of reading the raw images from the input `data/all_data` folder. This is followed by a data cleaning step - implemented in `src/data_pipeline/data_cleaning.py` -, which removes duplicates and images that do not meet a given quality criteria. The data is then finally split into train, validation and test datasets in the third stage implemented in `src/data_pipeline/data_processing.py`. The fourth stage is training, which includes early stopping to prevent overfitting. The fifth and final stage is evaluation. This includes multiple steps, such as calculating classification metrics, confusion matrices as well as plots that display a model's focus on the input image - the latter is done by using Grad-CAM or Attention Rollout based on the model type. The implementations can be found in `src/train.py` and `src/evaluation.py` respectively. 

I've implemented 4 models for the task; a dummy baseline which predicts majority class - calculated from the train dataset -, as well as a simple, medium and complex model. I have also evaluated the performance of a Vision Transformer. Note that the solution uses the ViT model for the pipeline by default. During evaluation the Vision Transformer outperformed all other models.

The solution also includes a simple MLaaS - the implementations can be found in `api.py` and `ui.py`.

### Extra Credit Justification

The following aspects deserve an extra mark, in my opinion:
- Implemented automated data cleaning
- Implemented and evaluated several architectures of different complexities
- Used advanced AI explainability - Grad-CAM/Attention Rollout - to further analyze model behaviour

### Data Preparation

Follow these steps to prepare the data for the pipeline:
1. Select all NEPTUN folders inside the following folder on sharepoint: `sharepoint_root/anklealign`.
2. Download zip.
3. Extract zip into `data/all_data`.
    - Note: This is the same `data/` folder that should be mounted to docker.

**NOTE**: There's a new zip at `sharepoint_root/` called `anklealign.zip`. This should be the exact same zip as the one downloaded in step 1. Make sure the folder structure looks like the following after extracting the files: `data/all_data/{NEPTUN}/...`

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t your-project-name:version .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

To run the solution, use the following command. You must mount your local data and outputs directory to `/app/data` and `/app/outputs` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm --gpus all ` -v /absolute/path/to/your/local/data:/app/data ` ` -v /absolute/path/to/your/local/outputs:/app/outputs ` your-project-name:version > training_log.txt 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - **`data_pipeline/`**:
        - `data_cleaning.py`: The script that implements the data cleaning stage. The output of this script is placed into `data/cleaned_numpy_data`
        - `data_preparing.py`: The script that's responsible for loading raw images and converting them into an `.npy` file as well as reading their corresponding labels. The outputs - images and labels separately - are placed into `data/numpy_all_data`.
        - `data_processing.py`: The script that contains the `torch.utils.data.Dataset` class as well as other data related utility functions such as returning data loaders and splitting data. When called explicitly in the pipeline - `run.sh` - its main purpose is to create the splits from the cleaned data. The splits are saved to `data/splits` after the script finished running
    - **`vit_pytorch`**: A PyTorch implementation of Vision Transformers used for the model architecture.
        - Source: https://github.com/jeonsworld/ViT-pytorch
        - Attribution: Reimplementation of the original paper _An Image is Worth 16x16 Words (Dosovitskiy et al.)_
    - `api.py`: Implementation of the backend of the MLaaS.
    - `config.py`: Contains an args class that's responsible for storing the arguments that are used by the entire pipeline such as data preparation parameters (e.g.: _similarity threshold_) or training hyperparameters (e.g.: _epochs_)
    - `models.py`: Responsible for instantiating the correct models - training stage - as well as loading the pre-trained weights for the models - evaluation stage. The script also contains the model definitions for the following models:
        - Dummy Baseline
        - Simple model
        - Medium model
        - Complex model
    - `run.ps1`: Powershell equivalent of `run.sh`. Used for local testing.
    - `train.py`: The main script for defining the model and executing the training loop.
    - `ui.py`: Implementation of the frontend of the MLaaS
    - `utils.py`: Helper functions and utilities used across different scripts. Used for operations such as setting up the optimizer, setting up the logger or plotting data.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run. Always contains the log of the latest run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `LICENSE`: License file.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `training_log.txt`: Example output after running the container.
