# vitmma19-anklealign-hazi

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Szaniszló Csongor Ádám
- **Aiming for +1 Mark**: Yes

### Solution Description

The current solution solves a computer vision problem, where the goal is to correctly classify the input image into one of three categories: Neutral, Pronation and Supination. These classifications can be made by observing the ankle's alignment on the image. If it leans outwards the correct label is supination. If it leans inwards the correct label is pronation. Otherwise the correct label is neutral.

The implementation includes 5 stages; the first is data preparation which is implemented in `src/data_pipeline/data_preparing.py`. This consists of reading the raw images from the input `data/all_data` folder. This is followed by a data cleaning step - implemented in `src/data_pipeline/data_cleaning.py` -, which removes duplicates and images that do not meet a given quality criteria. The data is then finally split into train, validation and test datasets in the third stage implemented in `src/data_pipeline/data_processing.py`. The fourth stage is training, which includes early stopping to prevent overfitting. The fifth and final stage is evaluation. This includes multiple steps, such as calculating classification metrics, confusion matrices as well as plots that display a model's focus on the input image - the latter is done by using Grad-CAM or Attention Rollout based on the model type. The implementations can be found in `src/train.py` and `src/evaluation.py` respectively. 

I've implemented 4 models for the task; a dummy baseline which predicts majority class - calculated from the train dataset -, as well as a simple, medium and complex model. I have also evaluated the performance of a Vision Transformer. Note that the solution uses the ViT model for the pipeline by default. In this case the model's weights are downloaded from: [https://storage.googleapis.com/vit_models/imagenet21k/{vit_model}.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) - where `vit_model=args.vitargs.vit_model`. During evaluation the Vision Transformer outperformed all other models.

The solution also includes a simple MLaaS - the implementations can be found in `api.py` and `ui.py`.

### Extra Credit Justification

The following aspects deserve an extra mark, in my opinion:
- Implemented automated data cleaning
    - Excluding duplicate images
    - Excluding images that are of low quality
- Implemented and evaluated several architectures of different complexities
- Used advanced AI explainability - Grad-CAM/Attention Rollout - to further analyze model behaviour

### Data Preparation

Follow these steps to prepare the data for the pipeline:
1. Select all NEPTUN folders inside the following folder on sharepoint: `sharepoint_root/anklealign`.
2. Download zip.
3. Extract zip into `data/all_data`.
    - Note: This is the same `data/` folder that should be mounted to docker.
4. Check all folders in `data/all_data`. All images must be directly inside `data/all_data/{NEPTUN}`. If you find any subfolders (e.g, `.../{NEPTUN}/extra_folder/img.jpg`), move the images up to the main NEPTUN folder and delete the empty subfolder.
    - _Warning_: Images left in sub-subfolders will be ignored by the pipeline.

**NOTE**: There's a new zip at `sharepoint_root/` called `anklealign.zip`. This should be the exact same zip as the one downloaded in step 1. Make sure the folder structure looks like the following after extracting the files: `data/all_data/{NEPTUN}/...`

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t your-project-name:version .
```

The following sessions shows you how to run the solution. The first _Run_ section gives instructions on how to run the standard pipeline, whilst the second gives instructions on how to run the pipeline with the MLaaS demo. For more information about the MLaaS demo please refer to [ML as a Service (MLaaS)](#ml-as-a-service-mlaas).

#### Run (Standard Pipeline Only) 

To run the (pipeline only) solution, follow the instructions below. You must mount your local data and outputs directory to `/app/data` and `/app/outputs` inside the container.

```bash
docker run --rm --gpus all ` -v /absolute/path/to/your/local/data:/app/data ` ` -v /absolute/path/to/your/local/outputs:/app/outputs ` your-project-name:version bash run.sh > logs/run.log 2>&1
```

- Replace `/absolute/path/to/your/local/data` with the actual path to your data folder that [data preparation](#data-preparation) uses.
- Replace `/absolute/path/to/your/local/outputs` with the actual path to your outputs folder on your host machine. This is where you'll see the outputs of the pipeline.
- The `> logs/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `logs/run.log`.
- The container is configured to run every step (data preprocessing, training, evaluation, inference).

#### Run (MLaaS Demo: Pipeline + UI)

To run the (MLaaS demo) solution, follow the instructions below. You must mount your local data and outputs directory to `/app/data` and `/app/outputs` inside the container.

```bash
docker run --rm --gpus all -p your_backend_port:8000 -p your_frontend_port:8501 ` -v /absolute/path/to/your/local/data:/app/data ` ` -v /absolute/path/to/your/local/outputs:/app/outputs ` your-project-name:version > logs/run.log 2>&1
```

- Replace `your_backend_port` and `your_frontend_port` with your desired ports.
    - `-p your_backend_port:8000`: Maps the Backend API
    - `-p your_frontend_port:8501`: Maps the Streamlit UI. Access it at [http://localhost:8501](http://localhost:8501) if `your_frontend_port = 8501`
- Replace `/absolute/path/to/your/local/data` with the actual path to your data folder that [data preparation](#data-preparation) uses.
- Replace `/absolute/path/to/your/local/outputs` with the actual path to your outputs folder on your host machine. This is where you'll see the outputs of the pipeline.
- The `> logs/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `logs/run.log`.
- The container is configured to run every step (data preprocessing, training, evaluation, inference and MLaaS demo).

Please note that during local testing `Ctrl + C` did not stop the container from running - the frontend and backend kept running. In this case you may want to manually stop the container via Docker Desktop or by running `docker rm -f {your_container's_id}`. The ID can be found by running `docker ps`.

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
    - `evaluation.py`: Script for evaluating the trained model on test dta and generating metrics.
    - `inference.py`: Scrip that runs the model on new, unseen data to generate predictions.
    - `models.py`: Responsible for instantiating the correct models - training stage - as well as loading the pre-trained weights for the models - evaluation stage. The script also contains the model definitions for the following models:
        - Dummy Baseline
        - Simple model
        - Medium model
        - Complex model
    - `run.ps1`: Powershell equivalent of `run.sh`. Used for local testing.
    - `train.py`: The main script for defining the model and executing the training loop.
    - `ui.py`: Implementation of the frontend of the MLaaS
    - `utils.py`: Helper functions and utilities used across different scripts. Used for operations such as setting up the optimizer, setting up the logger or plotting data.

- **`logs/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run. Always contains the log of the latest run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `LICENSE`: License file.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `training_log.txt`: Example output after running the container.

### ML as a Service (MLaaS)
The project also includes a simple MLaaS demo which consists of two components:
- Frontend: For the implementation of the frontend/UI the streamlit library was used. Details can be seen in `src/ui.py`
- Backend: For the implementation of the backend fastapi and uvicorn was used. Details can be seen in `src/api.py`

The backend exposes an endpoint called `/predict`. The endpoint expects the raw image data sent in the post request. The frontend reaches the aforementioned endpoint over the network - current implementations uses localhost for this purpose.

The demo can be started by following the steps listed below:
- Open two terminals
- Install requirements using: `pip install -r requirements.txt` (using one of the terminals).
- Navigate to the project root in both terminals
- Enter the following commands:
    - frontend: `streamlit run src/ui.py`
    - backend: `python src/api.py`
- Streamlit should automatically open the website on localhost
- Upload an image from your file browser
    - The uploaded image should appear on the UI
- Click `Classify Image`
- View results
    - Predicted class and its confidence, as well as per-class confidence can be viewed on the output.

Note: The backend automatically loads a pre-trained from `outputs/latest_folder_by_time` - where `latest_folder_by_time` is the output folder of the latest run. Thus you need to run the pipeline first or acquire appropriate model weights for one of the models used in this project to run the demo.