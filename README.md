# AI-4-Media-Project-Template (24/25)

Follow the steps in [the project repository setup instructions](https://moodle.arts.ac.uk/mod/page/view.php?id=1374587) on how to setup the repository for your AI 4 Media mini-project.


## Student name: Zimin Yue
## Student number: 24004556
## Project title: DeepASL Plus
## Link to project video recording: 

# Setup instructions:

Instructions for setting up the conda environment, any files that need downloading, and the specific technical instructions for how to run your code project go here:

### 1. Set up a conda environment

This project requires the `terminal` or `command prompt` (the program `Anaconda Command Prompt` on Windows).

(1) Download and install `Anaconda` from https://www.anaconda.com/download/success

(2) Open the software program `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac).

(3) Navigate to `C:\Users\<YourUsername>\` (on Windows) `/Users/<YourUsername>/`(on Mac) `/home/<YourUsername>/` (On Linux)
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

### 3. Clone this repository
You can clone the repository using GitHub Desktop or by running the following command in your terminal:
```
git clone https://git.arts.ac.uk/24004556/AI-4-Media-Project-Zimin-Yue
cd AI-4-Media-Project-Zimin-Yue
```

### 4. Install the Required Libraries
Run the following commands in `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac):

```
pip install -r requirements.txt
```


### 5. Place the `.streamlit` folder and `streamlitImage` folder in your home directory


Place the `.streamlit` folder and `streamlitImage` folder in
`C:\Users\<YourUsername>\`(on Windows)


`/Users/<YourUsername>/`(on Mac)


`/home/<YourUsername>/`(on Linux)


### 6. Run DeepASL Plus on your computer!
(1) Open the software program `Anaconda Command Prompt` (on Windows) or `terminal` (on Mac).

(2) Run the following commands, one after the other:
```
cd /path/to/AI-4-Media-Project-Zimin-Yue
```
(replace </path/to/> with the actual path where you downloaded the project)
```
conda activate aim
```
```
streamlit run StreamlitAPP/Welcome_to_DeepASL_Plus.py
```

