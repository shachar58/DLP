import csv
import glob
import json
import os

from Experiment import Experiment
from DatasetLoader import DatasetLoader as dl
import preprocessing as pre
import Autoencoder as AE
import evaluation as eva

# Create experiments file for future data retention if not already exist
try:
    with open("Experiments.csv", "x+", newline='') as f:
        # store data in variable
        data = csv.writer(f)
        # create file headers
        headers = ["Dataset", "Window size", "Retrain", "Auto-encoder model", "Minmax model"]
        # add data to csv file
        data.writerow(headers)
    f.close()
except:
    print("File Exists")


# Specify the directory containing the configuration files
config_directory = 'config'
# List all JSON files in the directory
config_files = glob.glob(os.path.join(config_directory, '*.json'))
# Loop over the configuration files
for config_file in config_files:
    # Load the current configuration file
    with open(config_file, 'r') as file:
        config = json.load(file)
    # Get configuration parameters
    csv_files = config["csv_files"]
    if "True" == config["train_ae"]:
        train_ae = True
    elif "False" == config["train_ae"]:
        train_ae = False
    else:
        print("train_ae Not Valid")
    minmax_model_to_load = config["minmax_model_to_load"]
    ae_model_to_load = config["ae_model_to_load"]
    window_size = config["window_size"]

    # Creating an instance of DatasetLoader for each CSV file
    loaders = {}
    for file in csv_files:
        # Create a unique name for each loader
        loader_name = file.replace('.csv', '').replace('_', ' ')
        # Create a DatasetLoader instance
        loaders[loader_name] = dl(file)
        loaders[loader_name].print_configuration_info()
        loaders[loader_name].load_dataset()
    # End of Data Loading Stage

    # Creating an array of experiments
    experiments = []

    # Preprocessing each DatasetLoader dataset
    for current_loader in loaders:
        current_loader = loaders[current_loader]
        exp = Experiment(f"/{current_loader.dataset_name}_{train_ae}")  # need to change for a more accurate representation
        experiments.append(exp)

        data, mapping = pre.pre_to_feature(current_loader)
        agg_df = pre.aggregate_data(data)

        print(current_loader.dataset_name)
        print(len(data))
        print(data.anomaly.sum())

        tensor_data, selected_data = pre.create_tensor_matrix(agg_df, train_ae, current_loader.dataset_name, minmax_model_to_load)
        if train_ae:
            ae = AE.train(tensor_data.shape[1], tensor_data, current_loader.dataset_name)
        else:
            ae = AE.load(tensor_data.shape[1], ae_model_to_load)

    # Creating DataFrames for evaluation
    df_vec, vectors_array = AE.create_vec_df(tensor_data, ae)

    # Mapping the time windows over 2D space
    umap_model, data_2d = eva.map_umap(vectors_array)

    # Clustering the results over K-means
    kmeans, kmeans_clusters = eva.cluster_kmeans(vectors_array)
    clusters, centroids_2d, df = eva.add_cluster_data(kmeans_clusters, kmeans,  selected_data, df_vec, data_2d, umap_model)

    # Using LOF anomaly detection
    eva.detect_LOF(df, df_vec)

    # Evaluating the results
    eva.roc_graph(df, exp)
    eva.apply_state(df, current_loader.dataset_name, exp, pred_column='lof_prediction')
    eva.evaluate_model_performance(df, 'anomaly', 'lof_prediction', current_loader.dataset_name, exp)

    # Visualising the results with graphs
    eva.visualise_clusters(df, centroids_2d, current_loader.dataset_name, window_size, exp, hue='state')
    eva.interactive_graph(df, current_loader.dataset_name, selected_data, window_size, train_ae, exp, color='state')

    with open("Experiments.csv", "a", newline='') as f:
        # Store data in variable
        data = csv.writer(f)
        row = (current_loader.dataset_name, config["window_size"], config["train_ae"],
               config["ae_model_to_load"], config["minmax_model_to_load"])
        # Add data to csv file
        data.writerow(row)
    f.close()
