from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import LocalOutlierFactor
import umap.umap_ as umap
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# import plotly as plt
import plotly.io as pio
import plotly.express as px


def map_umap(vectors_array):
    """
     Performs dimensionality reduction using UMAP (Uniform Manifold Approximation and Projection).
     Parameters:
         vectors_array (array): An array of vectors to be reduced in dimensionality.
     Returns:
         tuple: A tuple containing the trained UMAP model and the 2D representation of the data.
     """
    print('start UMAP')
    umap_model = umap.UMAP(n_components=2, n_neighbors=20, min_dist=1, random_state=None)
    # Fit and transform the data
    data_2d = umap_model.fit_transform(vectors_array)
    print('finish UMAP')
    return umap_model, data_2d


def cluster_kmeans(vectors_array, n_cluster=8):
    """
    Applies KMeans clustering to the provided data.
    Parameters:
        vectors_array (array): An array of vectors to be clustered.
        n_cluster (int): Number of clusters to form.
    Returns:
        tuple: A tuple containing the KMeans model and the cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(vectors_array)
    return kmeans, clusters


def add_cluster_data(clusters, kmeans, selected_data, df_vec, data_2d, umap_model):
    """
    Adds cluster information to the original data and calculates the centroids.
    Parameters:
        clusters (array): Cluster labels for each data point.
        kmeans (KMeans): KMeans clustering model.
        selected_data (DataFrame): Original DataFrame.
        df_vec (DataFrame): DataFrame of vectors.
        data_2d (array): 2D representation of the data.
        umap_model (UMAP): Trained UMAP model.
    Returns:
        tuple: A tuple containing cluster labels, 2D centroids, and the updated DataFrame with cluster information.
    """
    df = selected_data.copy()
    df_2d = pd.DataFrame(data_2d, columns=["x", 'y'])
    df = pd.concat([df, df_2d, df_vec], axis=1)
    # Add cluster information to the original data
    df['cluster'] = clusters  # need to add centroids

    # Calculate centroids of each cluster
    centroids = kmeans.cluster_centers_
    # Transform centroids to the same 2D space as the data
    centroids_2d = umap_model.transform(centroids)
    print('finished clustering')
    return clusters, centroids_2d, df


def detect_LOF(df, df_vec, n_neighbors=20, contamination=0.2):
    """
    Applies the Local Outlier Factor (LOF) method for anomaly detection.
    Parameters:
        df (DataFrame): The original DataFrame.
        df_vec (DataFrame): DataFrame containing vectors for anomaly detection.
        n_neighbors (int): Number of neighbors to use for LOF.
        contamination (float): The proportion of outliers in the data set.
    Returns:
        None: Modifies the DataFrame in-place by adding LOF predictions and scores.
    """
    # Initialize and fit LOF
    for source in df['source'].unique():
        # Get the indices of rows for the current source
        indices = df[df['source'] == source].index
        # Extract the corresponding vectors from df_vec
        vectors = df_vec.loc[indices]
        # Convert vectors to a numpy array if it's not already
        source_vectors_array = vectors.to_numpy()
        # Apply LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        print(source, source_vectors_array.shape)
        df.loc[indices, 'lof_prediction'] = lof.fit_predict(source_vectors_array)
        df.loc[indices, 'lof_score'] = -lof.negative_outlier_factor_
        threshold = np.percentile(-lof.negative_outlier_factor_, 95)
        print(threshold)
        df.loc[indices, 'lof_threshold'] = threshold

    df['lof_prediction'] = df['lof_prediction'].map({1: 0, -1: 1})

    # df['our_pred'] = df['lof_score'] != 1
    # df['our_pred'] = df['lof_score'] >  1.000652   # ROC based
    df['our_pred'] = df['lof_score'] > df['lof_threshold']  # source threshold based


def create_confusion_matrix(label_column, exp, pred_column):
    """
    Creates and displays a confusion matrix for the given labels and predictions.
    Parameters:
        label_column (array): Array of true labels.
        exp (Experiment): Experiment object with directory information.
        pred_column (array): Array of predicted labels.
    Returns:
        None: Displays and saves the confusion matrix plot.
    """
    cm = confusion_matrix(y_true=label_column, y_pred=pred_column, labels=[0, 1])
    # Display confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Anomaly']).plot(values_format='d')
    plt.show()
    # Save the plot to a file
    plt.savefig(exp.plots_dir + '/confusion_matrix.png', bbox_inches='tight')


def roc_graph(df, exp):
    """
       Generates and displays a ROC curve for the given DataFrame.
       Parameters:
           df (DataFrame): The DataFrame containing true labels and scores.
           exp (Experiment): Experiment object with directory information.
       Returns:
           None: Displays and saves the ROC curve plot.
       """
    # Generate ROC curve data
    fpr, tpr, thresholds = roc_curve(df['anomaly'], df['lof_score'])
    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # Printing the optimal threshold
    print("Optimal Threshold:", optimal_threshold)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(exp.plots_dir + '/roc_curve.png', bbox_inches='tight')
    plt.show()


# Define a function to categorize each row
def categorize_row(row, prediction):
    """
       Categorizes a row as True Positive, False Positive, True Negative, or False Negative.
       Parameters:
           row (Series): A row from the DataFrame.
           prediction (str): Column name of the predicted labels.
       Returns:
           str: Category of the prediction ('TP', 'FP', 'TN', 'FN').
       """
    if row['anomaly'] == 1 and row[prediction] == 1:
        return 'TP'  # True Positive
    elif row['anomaly'] == 0 and row[prediction] == 1:
        return 'FP'  # False Positive
    elif row['anomaly'] == 0 and row[prediction] == 0:
        return 'TN'  # True Negative
    elif row['anomaly'] == 1 and row[prediction] == 0:
        return 'FN'  # False Negative


def apply_state(df, dataset_name, exp, pred_column):
    """
     Applies the categorization to each row and creates a state breakdown.
     Parameters:
         df (DataFrame): The DataFrame to categorize.
         dataset_name (str): Name of the dataset.
         exp (Experiment): Experiment object containing directory information for saving results.
         pred_column (str): Column name of the predicted labels.
     Returns:
         None: Saves the state breakdown to a CSV file.
     """
    # Apply the function to each row
    df['state'] = df.apply(lambda row: categorize_row(row, pred_column), axis=1)
    df_state_breakdown = df.groupby(['state', 'source']).count().query('time_window>0')['time_window']
    print(df_state_breakdown)
    df.to_csv(exp.results + f"/{dataset_name} state breakdown.csv")


def scores_to_indices(lof):
    """
    Converts LOF scores to outlier indices.
    Parameters:
        lof (LocalOutlierFactor): LOF model.
    Returns:
        array-like: Indices of outliers.
    """
    scores = -lof.negative_outlier_factor_
    threshold = np.percentile(scores, 95)
    # Identifying outliers
    is_outlier = scores > threshold
    df_scores = pd.DataFrame({'score': scores})
    # indexes_a = df_scores.query('score != 1.0').index
    return is_outlier


def get_unique_clusters(kmeans):
    """
    Returns the number of unique clusters identified by KMeans.
    Parameters:
        kmeans (KMeans): KMeans clustering model.
    Returns:
        int: Number of unique clusters.
    """
    # unique_clusters
    unique_clusters = len(np.unique(kmeans.labels_))
    print(unique_clusters)
    return unique_clusters


def visualise_clusters(df, centroids_2d, dataset_name, window_size, exp, hue='source'):
    """
    Visualizes the clusters with a scatter plot and saves the plot to a file.
    Parameters:
        df (DataFrame): DataFrame containing the data and cluster information.
        centroids_2d (array-like): 2D coordinates of cluster centroids.
        dataset_name (str): Name of the dataset.
        window_size (int): Time window size used for analysis.
        exp (Experiment): Experiment object containing directory information for saving plots.
        hue (str): Column name to determine the color of points in the scatter plot.
    Returns:
        None: Displays and saves the scatter plot.
    """
    # Visualize the clusters for each time window
    plt.figure(figsize=(15, 7))
    sns.scatterplot(data=df, x='x', y='y', hue=hue, palette='dark')

    # Add centroid labels
    for i, c in enumerate(centroids_2d):
        plt.text(c[0], c[1], f"{i}", color='black', ha='center', va='center', fontweight='bold', fontsize=10)

    plt.title(f'{dataset_name} - Cluster of Function Behaviour over Time Windows of {int(window_size)} minutes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(exp.plots_dir + f"/{hue}_{dataset_name}.jpg")


def interactive_graph(df, dataset_name, selected_data, window_size, train_ae, exp, color='anomaly'):
    """
    Creates an interactive scatter plot using Plotly and saves it as an HTML file.
    Parameters:
        color (str): Name of the column to visualise.
        df (DataFrame): DataFrame containing the data and additional information for hover.
        dataset_name (str): Name of the dataset.
        selected_data (DataFrame): The original data used for creating scatter plot hover information.
        window_size (int): Time window size used for analysis.
        train_ae (bool): Indicates whether an autoencoder was trained.
        exp (Experiment): Experiment object containing directory information for saving plots.
    Returns:
        None: Displays an interactive scatter plot and saves it as an HTML file.
    """
    # Create a scatter plot with hover data
    fig = px.scatter(df, x='x', y='y', color=color, symbol='source',
                     title=f'{dataset_name}- Function Behaviour over TimeWindows of {int(window_size)} minutes, {dataset_name}- train AE={train_ae}',
                     hover_data=list(selected_data.columns) +  # Add more columns as needed
                                ['lof_score', 'lof_prediction', 'anomaly', 'our_pred', 'state', 'cluster'])

    # Update layout for better readability and to fit the markers
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")), selector=dict(mode='markers'))
    fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0.5, xanchor='center', yanchor='bottom'))
    fig.update_layout(legend=dict(font=dict(size=10), y=-0.6, ))
    fig.update_layout(autosize=False, width=1000, height=800)

    # Show the plot
    fig.show()
    pio.write_html(fig, exp.plots_dir + f"/{dataset_name} - train with AE {train_ae}.html")


def evaluate_model_performance(data, true_label_col, predicted_label_col, dataset_name, exp):
    """
    Evaluates the model performance using various metrics and creates a confusion matrix.
    Parameters:
        data (DataFrame): DataFrame containing true and predicted labels.
        true_label_col (str): Column name for true labels.
        predicted_label_col (str): Column name for predicted labels.
        dataset_name (str): Name of the dataset.
        exp (Experiment): Experiment object containing directory information for saving results.
    Returns:
        DataFrame: A DataFrame containing the calculated metrics.
    """
    # Compute metrics
    accuracy = accuracy_score(data[true_label_col], data[predicted_label_col])
    precision = precision_score(data[true_label_col], data[predicted_label_col])
    recall = recall_score(data[true_label_col], data[predicted_label_col])
    f1 = f1_score(data[true_label_col], data[predicted_label_col])
    # Confusion Matrix
    cm = create_confusion_matrix(data[true_label_col], exp, pred_column=data[predicted_label_col])
    # Creating a DataFrame to hold the results
    results_df = pd.DataFrame({
        'Dataset': dataset_name,
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'],
        'Value': [accuracy, precision, recall, f1, cm]
    })
    return results_df


def apply_labels(df, dataset_name, exp, pred_column='lof_prediction'):
    """
    Applies labels to the DataFrame based on the LOF predictions and evaluates the model.
    Parameters:
        df (DataFrame): The DataFrame to label and evaluate.
        dataset_name (str): Name of the dataset.
        exp (Experiment): Experiment object containing directory information for saving results.
        pred_column (str): Column name for the predictions.
    Returns:
        None: Saves the evaluation results to a CSV file.
    """
    # Replace these with the actual column names in your dataset
    true_label_col = 'anomaly'
    # Evaluate the model performance
    evaluation_results = evaluate_model_performance(df, true_label_col, pred_column, dataset_name, exp)
    print(evaluation_results)
    evaluation_results.to_csv(exp.results + f"{dataset_name} - {pred_column} - metrics scores.csv")
