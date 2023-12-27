# DataLeakage Detection README

## Table of Contents

1. [Preliminaries](#preliminaries)
2. [Modules/Architecture](#modules-architecture)
4. [Execution Instructions](#execution-instructions)
5. [Notes](#notes)

---
## Preliminaries:
1. Install unzip `sudo apt-get install unzip`
2. Unzip the file using `unzip {filename}`
3. Install python `sudo apt install python3.10`
4. Install pip on your machine with `sudo apt install python3-pip -y`
5. Install the requirements listed in [requirements.txt](requirements.txt) with the command: `pip install -r requirements.txt`

This project was tested on **linux** with **python 3.10** 

---

# Modules/Architecture
The Data Leakage Detection Method use the following modules:
- Algorithm Overview:
  - This algorithm is using the recurring patterns in the data to find anomalous activity.
  - The data is aggregated into groups of `(Time Widnow, Function Role)`, meaning for each time-window (by default it is *1 minute*) we have aggregated data for each function, of all the operations made by it in this time-window.
  - Each dot on the plotted graphs is essentially the entire activity of a specific function in a specefic time-window.
- Module Descriptions:
  - `DatasetLoader`: This module handles the data from the CSV files and holds the DataLoader class.
  - `Experiment`: This module handles the information and outputs gained from the experiments and holds the Experiment class.
  - `preprocessing`: This module handles the data from the Dataloader and process it, contains only functions.
  - `Autoencoder`: This module handles the encoding data from the CSV files and creating tensor representation, holds the Autoencoder class.
  - `evaluation`: This module handles the evaluation, metrics and graphics for the model, contains only functions.
  - `main`: This is the run module, you can change the parameters from here.


# Execution Instructions
1. Place your CSV data files in the data/ directory. 
2. Modify the csv_files list in the script to include your CSV files. 
3. Configure the script parameters (train_ae, window_size, etc.) as needed. 
4. Run the script using a Python interpreter.

alternatively, just skip to step 4 for testing or change the enabled files in the script

# Notes
**It's important to notice that the graph may not be available to show during the run, but in the experiment folder You will be able to find everything**