# FFT-BIS
A deep learning model that applies the attention mechanism to Fourier-transformed EEG signals for predicting and elucidating the Bispectral Index (BIS) in an end-to-end manner.

## Overal Descriptions
### Data
Due to policy constraints, the data used in our research cannot be disclosed. Instead, we provide code to generate synthetic data for training the proposed model and performing post-hoc analysis. You can find the code in [DataGeneration.ipynb](https://github.com/JunetaeKim/FFT-BIS/blob/main/DataSimulation/DataGeneration.ipynb) within the [DataSimulation](https://github.com/JunetaeKim/FFT-BIS/tree/main/DataSimulation) folder. The generated data are stored in the [ProcessedData](https://github.com/JunetaeKim/FFT-BIS/tree/main/ProcessedData) folder.

### Main model
The source code for developing the main model is included in [ModelTraining.py](https://github.com/JunetaeKim/FFT-BIS/blob/main/MainModel/ModelTraining.py), which can be found in the [MainModel](https://github.com/JunetaeKim/FFT-BIS/tree/main/MainModel) folder.
The model weights, which were obtained by running ModelTraining.py during the authors' research, are saved in the hdf5 format in the [ModelResults](https://github.com/JunetaeKim/FFT-BIS/tree/main/ModelResults) folder. The provided file in the folder contains the results of training the model based on the research data used by the authors. You can run python ModelTraining.py in the terminal to train the model anew. Please ensure you have the necessary data prepared before training.

### Main results
Post-evaluation involves the model's performance and visualization of explanatory power. [MainResult.ipynb](https://github.com/JunetaeKim/FFT-BIS/blob/main/MainResult.ipynb) displays the results based on the actual data used in the study, but it is provided for read-only purposes, as the data cannot be disclosed. Alternatively, we offer an environment for performing post-evaluation using simulated data in [MainResultSIM.ipynb](https://github.com/JunetaeKim/FFT-BIS/blob/main/MainResultSIM.ipynb). You can use the model weights learned by the researcher by loading the provided file in the [ModelResults](https://github.com/JunetaeKim/FFT-BIS/tree/main/ModelResults) folder.

## Notice
The results from [MainResult](https://github.com/JunetaeKim/FFT-BIS/blob/main/MainResult.ipynb) and [MainResultSIM](https://github.com/JunetaeKim/FFT-BIS/blob/main/MainResultSIM.ipynb) are <u>significantly different</u>. The results in MainResult are identical to those published in the authors' paper. This difference is due to the use of synthetic data. The actual data were obtained from EEG and BIS at [https://vitaldb.net/](https://vitaldb.net/). 


## Development Contributorship
[Junetae Kim](https://github.com/JunetaeKim) developed, trained, and tuned the model. 
[Eugene Hwang](https://github.com/joyce0215) tuned the model and assessed its performance and interpretability. 
[Jinyoung Kim](https://github.com/sacross93) refactored and structured the code.
