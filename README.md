# TexSense - Running

The project aims to predict continuous 3D knee moments during various running conditions using pressure insoles (PI) 
and inertial measurement unit (IMU) data. This repo includes implementations of the neural network models, 
data preprocessing utilities, and evaluation metrics for assessing model performance.
The goal is to develop accurate models that can assist in biomechanical analysis and injury prevention in sports and 
rehabilitation contexts.

The project TexSense is funded within the context of WISS 2025, der Wissenschafts- und Innovationsstrategie 2025, by the federal state of Salzburg.<br><br>
Publications:
- Höschler, L., Halmich, C., Schranz, C., Fritz, J., Koelewijn, AD., Schwameder, H. (2024). TOWARDS REAL-TIME ASSESMENT: WEARABLE-BASED ESTIMATION OF 3D KNEE KINETICS IN RUNNING AND THE INFLUENCE OF PREPROCESSING WORKFLOWS. *ISBS Proceedings Archive:* Vol. 37: Iss.2, ISBS 2024 Conference in Salzburg, 15th - 20th July, 2024.
## Data Set

The repo comes with an example dataset containing a subset of 10 participants of our original data (n=19). 
Only a single shoe condition is provided.
The data consists of multiple 10 second - samples of continuous (synchronized and preprocessed) IMU, PI and 3D Knee moments data from each running condition.
Detailed information regarding preprocessing are stated below.

DOI: 10.5281/zenodo.11119845
<br>Link to example data: [example dataset](https://doi.org/10.5281/zenodo.11119845)<br>

### Preprocessing Information
#### 3D Knee Moments: 
- 3D Inverse Dynamics during stance phases
- flight phases filled with zero-values
- fs = 100 Hz
- normalized to body mass
- tensors with shape [1000 x 3]
<br>

#### PI Data
- "raw" pressure readings from XSensor sensors (resolution: 31 x 11)
- fs = 100 Hz
- normalized to body mass
- flattened tensors with shape [1000 x 341]
<br>

#### IMU Data
- 3D Accelerometer and Gyroscope readings from 4 body positions (foot, shank, thigh, pelvis)
- filtered using a 2nd order Butterworth low-pass filter with custom cut-off frequencies
- each channel is z-score normalized
- tensors with shape [4000 x 24]
<br>

### Conditions
#### Treadmill Slopes: 
- 0 % (level)
- 5 % incline (up)
- 5 % decline (down)<br>

#### Treadmill Speeds:
- self-selected speed
- self-selected speed - 1 km/h
- self-selected speed + 1 km/h<br>
#### Sides:
- left
- right<br>
#### Slices:
- non-overlapping 10-second windows over 60 seconds of running<br>
#### Naming convention: *ParticipantID_Shoe_Slope_Speed_Side_Slice*



## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [File Structure](#file-structure)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/luceskywalker/TexSense.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation
1. Ensure you have downloaded the [example dataset](https://doi.org/10.5281/zenodo.11119845) into the `CONT/` directory. 

### Model Training and Evaluation
1. Train a model using the provided script or notebook files in the `main/` directory.
2. Evaluate the trained model using the `model_statistics()` function in `model_stats.py`.
3. Plot the training and validation loss using `plot_loss()` function in `plot.py`
4. Plot the predictions loss using `plot_prediction()` function in `plot.py`

### Run own experiments
1. Alter the model architecture and training parameters by editing the `CONT_config.txt` in `config/`


## File Structure

```
projectname/
│
├── CONT/
│   ├── config/ 
│   │   └── CONT_config.txt    # Configuration File
│   │
│   ├── main/
│   │   ├── main_CONT.ipynb    # main as .ipynb
│   │   └── main_CONT.py       # main as .py
│   │
│   ├── utils/
│   │   ├── data.py            # dataset and dataloader
│   │   ├── loss.py            # custom loss function
│   │   ├── model.py           # cnn model
│   │   ├── plot.py            # plot functions
│   │   ├── training.py        # train and eval routines
│   │   ├── utils.py           # preprocessing and evaluation utils
│   │
│   └── example_dataset/       # Raw dataset files
│
├── README.md                  # Project README file
├── requirements.txt           # Project dependencies
├── LICENSE_code.md            # License information (code)
└── LICENSE_data.md            # License information (data)
```


## Contributing

Contributions to the project are welcome! Please reach out to us via [e-mail](mailto:lucas.hoeschler@googlemail.com). 

## License
Please note that the TexSense repository is dual-licensed.
<br>
The TexSense dataset is available under the terms of the Creative Commons BY-NC 4.0 license.
<br>
The provided example code is licensed under the MIT license. 

Please see the respective LICENSE files for more details.

## Contact
Lucas Höschler, [lucas.hoeschler@googlemail.com](mailto:lucas.hoeschler@googlemail.com)

