from Experiment import Experiment
from DatasetLoader import DatasetLoader as dl
import preprocessing as pre
import Autoencoder as AE
import evaluation as eva

"""Base Configurations"""
#  All CSV datasets files
csv_files = [
    # 'small_benign_file.csv',
    # 'small_anomalies_file.csv',
    'airlines_july_benign.csv',
    'airlines_july_anomaly.csv',
    # 'vod_oct_benign.csv',
    # 'vod_oct_anomaly.csv',
    # 'combined-dataset.csv',
    # 'combined-anomaly.csv',
    # 'vod_storage_API_calls.csv'
    ]

data_dir = "data/"

ae_model_to_load = 'models/small benign file.mdl'
minmax_model_to_load = 'models/small benign file_min_max_scaler.joblib'

train_ae = False

window_size = 1
"""End of Configurations"""

################################
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
eva.interactive_graph(df, current_loader.dataset_name, selected_data, window_size, train_ae, exp)
