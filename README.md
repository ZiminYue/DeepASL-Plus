# DeepASL Plus
## Description
DeepASL Plus, developed based on Cesar Almendarez's [DeepASL](https://github.com/cesarealmendarez/DeepASL), is an American Sign Language (ASL) interpretation project powered by Python code, PyTorch-based model, and MediaPipe. It takes input from the webcam video feed and outputs texts and a skeleton-like animation generated from the captured gesture in real-time. 

Compared to the original project, DeepASL Plus includes several key updates: a new dataset for an expanded vocabulary, a new model for gesture recognition trained using Convolutional Neural Networks (CNNs), and a user interface built with Streamlit.

## Project Video 
[DeepASL Plus Intro (Panapto)](https://ual.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=88dbc0d0-4ecd-476e-81ba-b2a300c59092)

## Setup instructions

⚠IMPORTANT: Due to compatibility issues with OpenCV's fullscreen window handling on macOS, the main function of the application could not run successfully during testing on macOS systems. However, you can still view the Streamlit interface on Mac by following the steps below. 

All functions can run successfully on Windows.😉

#
### 1. Set up a conda environment

This project requires the `terminal` or `command prompt` (the program `Anaconda Command Prompt` on Windows).

(1) Download and install `Anaconda` from https://www.anaconda.com/download/success

(2) Open the software program `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac).

(3) Navigate to `C:\Users\<YourUsername>\` (on Windows) `/Users/<YourUsername>/`(on Mac) 
folder and run the following commands, one after the other:

```
conda create --name aim python=3.10
```
```
conda activate aim
```
```
pip install ipython jupyter
```

#
### 2. Clone this repository

(1) You can clone the repository with GitHub Desktop or download ZIP (remember to extract the files!).  

(2) Run the following command in your `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac):
```
cd "<path-to-your-download>/DeepASL-Plus"
```
(replace `<path-to-your-download>` with the actual path where you downloaded the project)

❗All necessary files, including the model, are included in the repository and will be downloaded automatically when cloning.

#
### 3. Install the required libraries

Run the following command in `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac):

```
pip install -r requirements.txt
```

#
### 4. Place the `.streamlit` folder and `streamlitImage` folder in your home directory

Place the `.streamlit` folder and `streamlitImage` folder in
`C:\Users\<YourUsername>\`(on Windows) or `/Users/<YourUsername>/`(on Mac)

❗If the `.streamlit` folder is invisible:

On Windows, make sure to enable `Hidden items` in `File Explorer` by checking the `Hidden items` box under the `View` tab.

On macOS, you can reveal hidden files by pressing `Command + Shift + .`.

#
### 5. Run DeepASL Plus on your computer!

(1) Open the software program `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac).

(2) Run the following command.

Basically, you will be able to launch the app only with the last command below. However, if you are trying to rerun the app after closing the terminal, or if images are not loading correctly, please run the following commands in this order:

If the terminal command is not starting with `(aim)`, please run:
```
conda activate aim
```
If the images are not loading correctly, please run:
```
cd "<path-to-your-download>/DeepASL-Plus/DeepASL_Plus/StreamlitAPP/"
```
(replace `<path-to-your-download>` with the actual path where you downloaded the project)

Launch the app:
```
streamlit run "<path-to-your-download>/DeepASL-Plus/DeepASL_Plus/StreamlitAPP/Welcome_to_DeepASL_Plus_🤟.py"
```
(replace `<path-to-your-download>` with the actual path where you downloaded the project)
