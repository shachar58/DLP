from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,  auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import umap
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import plotly as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px


def map_umap(vectors_array):
    print('start UMAP')
    umap_model = umap.UMAP(n_components=2, n_neighbors=20, min_dist=1, random_state=None)
    # Fit and transform the data
    data_2d = umap_model.fit_transform(vectors_array)
    print('finish UMAP')
    return data_2d


def cluster_kmeans(vectors_array, selected_data, df_vec, data_2d, umap_model, n_cluster=8):
    print('start clustering')
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(vectors_array)

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
    return centroids_2d


def detect_LOF(df, df_vec, n_neighbors=20, contamination=0.2):
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
    # df['our_pred'] = df['lof_score'] > df['lof_threshold'] # source threshold based


def confusion_matrix(df, pred_column='lof_prediction'):
    cm = confusion_matrix(y_true=df['anomaly'], y_pred=df[pred_column], labels=[0, 1])
    # Display confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Anomaly']).plot(values_format='d')


def roc_graph(df):
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
    plt.show()


# Define a function to categorize each row
def categorize_row(row, prediction):
    if row['anomaly'] == 1 and row[prediction] == 1:
        return 'TP'  # True Positive
    elif row['anomaly'] == 0 and row[prediction] == 1:
        return 'FP'  # False Positive
    elif row['anomaly'] == 0 and row[prediction] == 0:
        return 'TN'  # True Negative
    elif row['anomaly'] == 1 and row[prediction] == 0:
        return 'FN'  # False Negative


def apply_state(df, dataset_name, pred_column):
    # Apply the function to each row
    df['state'] = df.apply(lambda row: categorize_row(row, pred_column), axis=1)
    df_state_breakdown = df.groupby(['state', 'source']).count().query('time_window>0')['time_window']
    print(df_state_breakdown)
    df.to_csv(f"{dataset_name} state breakdown.csv")


def scores_to_indices(lof):
    scores = -lof.negative_outlier_factor_
    threshold = np.percentile(scores, 95)
    # Identifying outliers
    is_outlier = scores > threshold
    df_scores = pd.DataFrame({'score': scores})
    # indexes_a = df_scores.query('score != 1.0').index
    return is_outlier


def get_unique_clusters(kmeans):
    # unique_clusters
    unique_clusters = len(np.unique(kmeans.labels_))
    print(unique_clusters)
    return unique_clusters


def visualise_clusters(df, centroids_2d, dataset_name, window_size):
    # Visualize the clusters for each time window
    plt.figure(figsize=(15, 7))
    sns.scatterplot(data=df, x='x', y='y', hue='anomaly', palette='dark')

    # Add centroid labels
    for i, c in enumerate(centroids_2d):
        plt.text(c[0], c[1], f"{i}", color='black', ha='center', va='center', fontweight='bold', fontsize=10)

    plt.title(f'{dataset_name} - Cluster of Function Behaviour over Time Windows of {int(window_size)} minutes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def interactive_graph(df, dataset_name, selected_data, window_size, train_ae):
    # Create a scatter plot with hover data
    fig = px.scatter(df, x='x', y='y', color='anomaly', symbol='source',
                     title=f'{dataset_name}- Function Behaviour over TimeWindows of {int(window_size)} minutes, {dataset_name}- train AE={train_ae}',
                     hover_data=list(selected_data.columns) + ['lof_score', 'lof_prediction', 'anomaly', 'our_pred',
                                                               'state', 'cluster'])  # Add more features as needed

    # Update layout for better readability and to fit the markers
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color="DarkSlateGrey")), selector=dict(mode='markers'))
    fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0.5, xanchor='center', yanchor='bottom'))
    fig.update_layout(legend=dict(font=dict(size=10), y=-0.6,))
    fig.update_layout(autosize=False, width=1000, height=800)

    # clicked_data = None
    # Function to handle click events
    # def handle_click(trace, points, state):
    #     if points.point_inds:
    #         ind = points.point_inds[0]
    #         clicked_data = df.iloc[ind]
    #         print(f"Clicked on data point: {clicked_data}")

    # Linking click event to the handler function
    # scatter = fig.data[0]
    # scatter.on_click(handle_click)

    # Show the plot
    fig.show()
    pio.write_html(fig, f"{dataset_name} - train with AE {train_ae}.html")


def evaluate_model_performance(data, true_label_col, predicted_label_col, dataset_name):
    # Compute metrics
    accuracy = accuracy_score(data[true_label_col], data[predicted_label_col])
    precision = precision_score(data[true_label_col], data[predicted_label_col])
    recall = recall_score(data[true_label_col], data[predicted_label_col])
    f1 = f1_score(data[true_label_col], data[predicted_label_col])
    # ROC AUC can be included if predicted_label_col contains scores/probabilities

    # Confusion Matrix
    conf_matrix = confusion_matrix(data[true_label_col], data[predicted_label_col])

    # Creating a DataFrame to hold the results
    results_df = pd.DataFrame({
        'Dataset': dataset_name,
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'],
        'Value': [accuracy, precision, recall, f1, conf_matrix]
    })

    return results_df


def apply_labels(df, pred_column, dataset_name):
    # Replace these with the actual column names in your dataset
    true_label_col = 'anomaly'
    # Evaluate the model performance
    evaluation_results = evaluate_model_performance(df, true_label_col, pred_column, dataset_name)
    print(evaluation_results)
    evaluation_results.to_csv(f"{dataset_name} - {pred_column} - metrics scores.csv")
