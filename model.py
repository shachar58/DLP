from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns


def encode_data(encoder, dataset, column_names, label_dict):
    for title in column_names:
        # Fit the label encoder with the updated dataset
        encoder.fit(dataset[title].astype(str))
        # Map unknown values to "<unknown>" and transform the labels
        label_dict[title] = dataset[title].map(lambda s: -1 if s not in dataset[title].values else s)
        dataset[title] = dataset[title].transform(lambda s: label_dict[title].get(s, -1))
    return dataset


if __name__ == '__main__':

    data = pd.read_csv('relations_al_b1.csv')
    new_data = pd.read_csv('relations_al_a1.csv')
    reshaped_new_data = new_data.reindex(data.index)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(data, reshaped_new_data, test_size=0.2, stratify=reshaped_new_data["Anomaly"], random_state=42)

    # Check the number of records
    print('The number of records in the training dataset is', X_train.shape[0])
    print('The number of records in the test dataset is', X_test.shape[0])
    print(f"The training dataset has {sorted(Counter(y_train).items())[0][1]} records for the majority class and {sorted(Counter(y_train).items())[1][1]} records for the minority class.")

    # categorical_features = ['RoleName', 'PolicyName', 'Resource', 'Operations']
    # small_features = ['TargetName', 'TargetType', 'Effect', 'OperationUsed', 'ReadOnly', 'SampleTime']

    categorical_features = ['RoleName', 'TargetName', 'TargetType', 'OperationUsed', 'ReadOnly', 'change_s3', 'change_dynamoDB', 'remove_policy', 'Anomaly']

    headers3 = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Millisecond', 'RoleName', 'TargetName', 'TargetType',
                'OperationUsed', 'ReadOnly', 'change_s3', 'change_dynamoDB', 'remove_policy', 'Anomaly']

    label = LabelEncoder()
    label_encoder_dict = {}

    encode_data(label, X_train, categorical_features, label_encoder_dict)

    encode_data(label, X_test, categorical_features, label_encoder_dict)

    model = IsolationForest(contamination=0.003, random_state=42).fit(X_train)  # Contamination defines the expected outlier percentage
    preds = model.predict(X_test)
    preds = [1 if i == -1 else 0 for i in preds]
    print(classification_report(y_test['Anomaly'], preds))
    # role_names = new_data['RoleName']

    encode_data(label, data, categorical_features, label_encoder_dict)

    encode_data(label, reshaped_new_data, categorical_features, label_encoder_dict)

    predictions = model.decision_function(reshaped_new_data)
    results = pd.concat([reshaped_new_data, pd.Series(predictions).rename('Pred')], axis=1)
    # results = pd.concat([results, role_names.rename('Role Names')], axis=1)

    # # Create a scatter plot of the numerical features
    plt.scatter(results['Anomaly'], results['Pred'], c=predictions)
    plt.xlabel('anomaly')
    plt.ylabel('prediction')
    plt.title('Isolation Forest Predictions')
    plt.colorbar()
    plt.show()
