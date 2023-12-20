from Experiment import Experiment
from DatasetLoader import DatasetLoader as dl
from Parser import Parser
import preprocessing as pre
import Autoencoder as AE
import evaluation as eva

"""Base Configurations Table """
#  All CSV datasets files
csv_files = [
    # 'small_benign_file.csv',
    # 'small_anomalies_file.csv',
    'airlines_july_benign.csv',
    'airlines_july_anomaly.csv',
    # 'vod_oct_benign.csv',
    # 'vod_oct_anomaly.csv',
    # 'combined-dataset.csv',
    # 'combined-anomaly.csv'
]

data_dir = "data/"

minmax_model_to_load = '/model/small benign file_min_max_scaler-5.joblib'
ae_model_to_load = '/model/small benign file-5.mdl'

train_ae = True

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

experiments = []
for current_loader in loaders:
    current_loader = loaders[current_loader]
    exp1 = Experiment(f"/{current_loader.dataset_name}_{train_ae}")  # need to change for a more accurate representation

    # p = Parser(current_loader)
    # p.parse()

    data, mapping = pre.pre_to_feature(current_loader)
    agg_df = pre.aggregate_data(data)

    print(current_loader.dataset_name)
    print(len(current_loader.data))
    print(current_loader.data.anomaly.sum())

    tensor_data, selected_data = pre.create_tensor_matrix(agg_df, train_ae, current_loader.dataset_name, minmax_model_to_load)
    if train_ae:
        ae = AE.train(tensor_data.shape[1], tensor_data, current_loader.dataset_name, minmax_model_to_load)
    else:
        ae = AE.load(tensor_data.shape[1], ae_model_to_load)

df_vec, vectors_array = AE.create_vec_df(tensor_data, ae)

umap_model, data_2d = eva.map_umap(vectors_array)

kmeans, kmeans_clusters = eva.cluster_kmeans(vectors_array)
clusters, centroids_2d, df = eva.add_cluster_data(kmeans_clusters, kmeans,  selected_data, df_vec, data_2d, umap_model)

eva.detect_LOF(df, df_vec)
eva.create_confusion_matrix(df)
eva.roc_graph(df)
eva.apply_state(df, current_loader.dataset_name, pred_column='lof_prediction')

eva.visualise_clusters(df, centroids_2d, current_loader.dataset_name, window_size, hue='state')
eva.interactive_graph(df, current_loader.dataset_name, selected_data, window_size, train_ae)
