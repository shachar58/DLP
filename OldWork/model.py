import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM

#  Globals
ANOMALY_RATIO = 0.01
MODEL_NAME = ''

def evaluation(is_anomaly, scores):
    print(classification_report(outlier['Anomaly'], is_anomaly))

    conf_matrix = confusion_matrix(outlier['Anomaly'], is_anomaly)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('', fontsize=18)
    plt.ylabel('Class', fontsize=18)
    plt.title('Prediction', fontsize=18)
    plt.show()

    # # Create a scatter plot of the numerical features
    plt.scatter(is_anomaly, scores, c=outlier['Anomaly'])
    plt.xlabel('anomaly class')
    plt.ylabel('prediction score')
    plt.title('Predictions')
    plt.colorbar()
    plt.savefig('Experiments/plot',)
    plt.show()


def isolation_forest():
    model = IsolationForest(contamination=0.0095, random_state=42)
    model.fit(normal[categorical_features])

    predictions = model.fit_predict(outlier[categorical_features])
    scores = model.decision_function(outlier[categorical_features])

    threshold = np.percentile(scores, 100 * ANOMALY_RATIO)
    is_anomaly = scores <= threshold

    predictions[predictions == 1] = 0
    predictions[predictions == -1] = 1

    outlier['Predictions'] = predictions
    outlier['Scores'] = scores

    return is_anomaly


def svm():
    model = OneClassSVM(gamma='auto', nu=0.2)
    model.fit(normal[categorical_features])

    predictions = model.predict(outlier[categorical_features])
    scores = model.decision_function(outlier[categorical_features])

    threshold = np.percentile(scores, 100 * ANOMALY_RATIO)
    is_anomaly = scores <= threshold

    predictions[predictions == 1] = 0
    predictions[predictions == -1] = 1

    outlier['Predictions'] = predictions
    outlier['Scores'] = scores

    return is_anomaly


def encode_data(encoder, dataset, column_names):
    for title in column_names:
        dataset[title] = encoder.fit_transform(dataset[title].astype(str))
    return dataset


if __name__ == '__main__':

    normal = pd.read_csv('relations_airline_july_benign.csv')

    # normal = pd.read_csv('relations_VOD_oct_benign.csv')
    # frames = [normal, vod]
    # normal = pd.concat(frames)
    # normal = sklearn.utils.shuffle(normal)

    normal.drop('SampleTime', inplace=True, axis=1)

    # outlier = pd.read_csv('relations_airline_july_anomaly.csv')
    outlier = pd.read_csv('relations_airline_sep_anomaly.csv')
    # outlier = pd.read_csv('relations_vod_dec_anomaly.csv')
    outlier.drop('SampleTime', inplace=True, axis=1)

    categorical_features = ['RoleName', 'TargetName', 'TargetType', 'OperationUsed', 'ReadOnly',
                            'change_s3', 'change_dynamoDB', 'remove_policy']
    #  When removing role name, less false were made.
    # categorical_features = ['TargetType', 'OperationUsed', 'ReadOnly',
    #                         'change_s3', 'change_dynamoDB', 'remove_policy']

    label = LabelEncoder()
    label_encoder_dict = {}

    encode_data(label, outlier, categorical_features)
    encode_data(label, normal, categorical_features)

    is_anomaly = svm()
    # is_anomaly = isolation_forest()

    evaluation(is_anomaly, outlier['Scores'])

    # outlier.copy().to_csv("for_gpt.csv", index=False)
