# DataLeakage Detection README

## Table of Contents

1. [Preliminaries](#preliminaries)
2. [Modules/Architecture](#modules-architecture)
3. [Execution Instructions](#execution-instructions)
4. [Module Details](#module-details)
5. [Notes](#notes)

---
## Preliminaries:
1. Install python `sudo apt install python3.10`
2. Install pip on your machine with `sudo apt install python3-pip -y`
3. Install the requirements listed in [requirements.txt](requirements.txt) with the command: `pip install -r requirements.txt`

This project was tested on **linux** with **python 3.10** 

---

# Modules/Architecture
The Data Leakage Detection Method use the following modules:

![Docs\Modules.JPG](Docs\Modules.JPG)

- Algorithm Overview:
  - This algorithm is using the recurring patterns in the data to find anomalous activity.
  - The data is aggregated into groups of `(Time Window, Function Role)`,
  - meaning for each time-window (by default it is *1 minute*) we have aggregated data for each function, of all the operations made by it in this time-window.
  - Each dot on the plotted graphs is essentially the entire activity of a specific function in a specefic time-window.
- Module Descriptions:
  - `DatasetLoader`: This module handles the data from the CSV files and holds the DataLoader class.
  - `preprocessing`: This module handles the data from the Dataloader and process it, contains only functions.
  - `Autoencoder`: This module handles the encoding data from the CSV files and creating tensor representation, holds the Autoencoder class.
  - `evaluation`: This module handles the evaluation, metrics and graphics for the model, contains only functions.
  - `main`: This is the run module, you can change the parameters from here.

# Module Details
  In depth explanation for each module:
  ## DatasetLoader
In this module we have the basic attributes `data` and `dataset name` and basic handling of the data,
the only thing that may be need to be changed is the `base_path='data/'` part of the init of the class,
it could be done by adding the argument in `main.py line 22`.
  ## preprocessing
The preprocessing module handles the CSV files based on the 
format we get from the importer we used in the other projects,
data for example is provided in the `data` folder.

In this module we clean the data from the CSV and create dataframes of the relevant columns,
we create features based on the processed data and aggregate the data according to
the two keys `time window #` and `source` (function role name).
  ## Autoencoder
The Autoencoder class is used in order to create a higher dimensions representation for the data.
We take the `(Time Window, Function Role)` and encode each row in the aggregated dataframe, 
when the encoding is done, we have a tensors of the data.

******The generic idea of this algorithm is that only one model is actually needed as the features are contained inside a closed range or linear in nature.******

the architecture could be changed in the `encoder` and `decoder` parts of the init, as well as the base embedding size that you can enter as a parameter in `main line 45` 
  ## evaluation
In the evaluation module there are a few algorithms that could be tweaked with experimentation, they are: `UMAP`, `K-means` and `LOF`.
In these algorithms the parameters to change are the number of neighbors and contamination rate.

The other part of this module are the confusion matrix and graphs.
For the `confusion matrix` it's simply the change of the `predicted_label_col` parameter,
the 3rd parameters you give the function as seen in `main line 65`,
the column name needs to be in the dataframe (`df`) you also provide to the function

For the `graphs` we have two functions for two type of graphs, and you can change the column you visualise by:

simple graph `evaluation line 227` you can change the `hue` parameter

interactive graph `evaluation line 255` you can change the `color` parameter



# Execution Instructions
1. Place your CSV data files in the data/ directory. 
2. Modify the csv_files list in the script to include your CSV files. 
3. Configure the script parameters (train_ae, window_size, etc.) as needed. 
4. Run the script using a Python interpreter.

alternatively, just skip to step 4 for testing or change the enabled files in the script

### main
The changes that affect this code runs are the files in the `config.json`

the names in `csv_files` are being used in the current run,
and those in `files_not_in_use` will be ignored, you can mix and match but
only the last file will be evaluated, so make sure the file you want is last.

# Notes
**It's important to notice that the graph may not be available to show during the run, but in the experiment folder You will be able to find everything**