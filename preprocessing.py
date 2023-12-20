import joblib
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def nans(df):
    return df[df.isnull().any(axis=1)]


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table['Length'] = len(df)
    cols_to_rename = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    cols_to_rename = cols_to_rename[['Missing Values', 'Length', '% of Total Values']]
    cols_to_rename = cols_to_rename[
        cols_to_rename.loc[:, '% of Total Values'] != 0].sort_values('% of Total Values', ascending=False).round(2)
    print("Your selected data frame has " + str(df.shape[1]) + " columns.\n" "There are " + str(
        cols_to_rename.shape[0]) + " columns that have missing values.")
    return cols_to_rename


def set_target(row):
    if row['eventsource'] == 's3.amazonaws.com':
        return row['requestparameters_bucketname']
    elif row['eventsource'] == 'dynamodb.amazonaws.com':
        return row['requestparameters_tablename']
    else:
        raise Exception('No Target Type Avialable')


def clean_and_separate_data(dataset_name, data, window_size=1):
    print(dataset_name)
    try:
        data = data[~data['sourceipaddress'].isin(['config.amazonaws.com', 'appsync.amazonaws.com'])]
        data = data[~data['useridentity_type'].isin(['AWSAccount'])]
        data = data[~data['useridentity_sessioncontext_sessionissuer_username'].isin(['Project-Overwatch'])]
    except:
        print("doesnt have source IP")
    # Convert 'time' to datetime and sort the data by time
    try:
        data['time'] = pd.to_datetime(data['time'])
    except:
        data['time'] = pd.to_datetime(data['eventtime'])  # json source
    data.sort_values('time', inplace=True)

    # Define your time window in minutes
    for N in [window_size]:
        tw_id_col_name = f'tw_id_{str(N).zfill(4)}'
        tw_start_col_name = f'tw_start_{str(N).zfill(4)}'
        # Calculate the start of the window
        data[tw_start_col_name] = data['time'].dt.floor(f'{N}T')
        # Add the time window column to the DataFrame
        data[tw_id_col_name] = data.groupby(tw_start_col_name).ngroup()

        data['time_window'] = data[tw_id_col_name]
        data['time_window_start'] = data[tw_start_col_name]
        data['time_window_size'] = N  # do not run multiple time window

    if any(substring in dataset_name for substring in ['airline', 'vod']):
        print('for Airline and VOD only!!')
        data.rename(columns={'useridentity_sessioncontext_sessionissuer_username': 'source'}, inplace=True)
        data.rename(columns={'eventname': 'operation'}, inplace=True)
        data.rename(columns={'requestparameters_key': 'key'}, inplace=True)
        data['key'].fillna('No File', inplace=True)

        # Apply the function
        data['target'] = data.apply(set_target, axis=1)
        data['bucket_access'] = 'private'
        data.loc[data['target'].str.contains("public"), 'bucket_access'] = 'public'

        data['anomaly'] = data.get('anomaly', 0)

    if any(substring in dataset_name for substring in ['airline', 'vod', 'small scale benign']):
        data['anomaly'] = (data['key'].str.contains("leak")).astype(int)

    # adapter for all data sources
    data = data[['time_window', 'time_window_start', 'time_window_size', 'time', 'operation', 'readonly',
                 'source', 'target', 'key', 'bucket_access', 'anomaly']]
    return data


def combine_datasets(loaders, load_key_array=('small benign file', 'airlines july benign', 'vod oct anomaly')):
    for k in load_key_array:
        loaders[k].load_dataset()
    all_datasets = {'small benign file': None,
                    'airlines july benign': None,
                    'vod_oct benign': None,
                    'vod oct anomaly': None
                    }
    for k in load_key_array:
        print(loaders[k].dataset_name, len(loaders[k].data))
    for k in load_key_array:
        all_datasets[k] = clean_and_separate_data(loaders[k].dataset_name, loaders[k].data)
    for k in load_key_array:
        print(all_datasets[k].columns)

    keys = list(all_datasets.keys())
    df_combined = pd.concat([all_datasets[keys[0]], all_datasets[keys[1]], all_datasets[keys[3]]], axis=0)
    print(df_combined.shape)
    print(len(df_combined))
    df_combined = df_combined.dropna(subset=['source'])
    df_combined = df_combined[df_combined['source'] != 'VODApplication-ErrorHandlerRole-YYQFY381BRQK']
    df_combined.reset_index(drop=True, inplace=True)
    print(len(df_combined))
    print(nans(df_combined))
    missing_values_table(df_combined)
    print(df_combined.anomaly.sum())
    return df_combined


def category_mapping(data, columns_to_encode=('operation', 'readonly', 'source', 'target', 'bucket_access')):
    category_mappings = {}
    # Loop through each column, convert to categorical, and store the category-code mapping
    for col in columns_to_encode:
        data[col] = data[col].astype('category')  # Ensure the column is converted to a categorical type
        categories = data[col].cat.categories
        codes = data[col].cat.codes.unique()
        mapping_df = pd.DataFrame({col: categories, f'{col}_code': codes})
        mapping_df.reset_index(drop=True, inplace=True)
        category_mappings[col] = mapping_df
    # For a more structured view, especially if columns have a different number of categories
    category_mappings_df = pd.concat(category_mappings, axis=1)
    return category_mappings_df


def remove_minmax_window(data):
    # remove first and last time windows due to natural data anomalies (partial event count in time window)
    min_tw = data.time_window.min()
    max_tw = data.time_window.max()
    print(f"first window:{min_tw}, last window:{max_tw}")
    data = data.query(f'(time_window>{min_tw}) and (time_window<{max_tw})')
    min_tw = data.time_window.min()
    max_tw = data.time_window.max()
    print(f"first window:{min_tw}, last window:{max_tw}")
    return data


def preprocess(dataloader):
    data = dataloader.data.copy()
    dataset_name = dataloader.dataset_name
    data = clean_and_separate_data(dataset_name, data)
    mapping = category_mapping(data)
    data = remove_minmax_window(data)
    return data, mapping


def feature_extraction(data):
    # Calculate the percentage of 'readonly == False' for each group and add it as a new column
    data['f_write_access_percentage'] = data.groupby(['time_window', 'source'], observed=True)['readonly'].transform(
        lambda x: (x == False).mean() * 100)
    # data['f_read_readwrite_ratio'] = data.groupby(['time_window', 'source'])['readonly'].transform(lambda x: (x == False).mean() * 100)
    # data['f_count_readwrite'] = data.groupby(['time_window', 'source']).readonly.transform(lambda x: (x == 'False').count())
    # data['f_count_read'] = data.groupby(['time_window', 'source']).readonly.transform(lambda x: (x == 'True').count())
    # data['f_read_readwrite_ratio'] = data.apply(lambda row: 0 if row['f_count_readwrite'] == 0 else row['f_count_read'] / (row['f_count_readwrite']),axis=1)

    # Count 'private' and 'public' bucket_access for each time_window and source
    data['f_count_private'] = data.groupby(['time_window', 'source'], observed=True).bucket_access.transform(
        lambda x: (x == 'private').count())
    data['f_count_public'] = data.groupby(['time_window', 'source'], observed=True).bucket_access.transform(
        lambda x: (x == 'public').count())

    # Set ratio to 0 when both counts are 0, otherwise calculate the normal ratio
    data['f_private_public_ratio'] = data.apply(
        lambda row: 0 if row['f_count_public'] == 0 else row['f_count_private'] / (row['f_count_public']), axis=1)

    # Calculate the percentage of 'readonly == False' for each group and add it as a new column
    data['f_public_access_percentage'] = data.groupby(['time_window', 'source'], observed=True)['bucket_access'].transform(
        lambda x: (x == 'public').mean() * 100)

    # Binary features for the presence of 'public' and 'private' accesses
    data['f_has_private_access'] = (data['f_count_private'] > 0).astype(int)
    data['f_has_public_access'] = (data['f_count_public'] > 0).astype(int)
    return data


def pre_to_feature(dataloader):
    data, mapping = preprocess(dataloader)
    data = feature_extraction(data)
    return data, mapping


def aggregate_data(data):
    data.sort_values(['time_window', 'source'], inplace=True)
    additional_max_columns = ['time_window_size', 'anomaly']
    first_val_columns = ['time_window_start', 'time']

    # Group by 'time_window' and 'source', then take the max
    grouped = data.groupby(['time_window', 'source'], observed=True)
    grouped_data = grouped.max()

    # Filter columns that start with 'f_' or are in the list of additional columns
    agg_data = grouped_data.filter(items=[col for col in grouped_data.columns if col.startswith('f_') or col in additional_max_columns])

    # Add the first value of specific columns from the grouped DataFrame
    for col in first_val_columns:
        agg_data[col] = grouped[col].first()

    # Reset the index to turn the multi-index into columns
    agg_df = agg_data.reset_index()
    print(f"Anomalies in aggregated data: {agg_df['anomaly'].sum()}")
    return agg_df


def create_tensor_matrix(agg_df, train_ae, dataset_name, minmax_model_to_load):
    exclude_cols = ['f_count_public', 'f_count_private']
    AE_cols = [col for col in agg_df.columns if (col.startswith('f_') and col not in exclude_cols)]
    selected_data = agg_df.dropna(how='all', subset=AE_cols).reset_index(drop=True)
    if train_ae:
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(selected_data[AE_cols].values)
        model_file_name = f'/content/drive/MyDrive/model/{dataset_name}_min_max_scaler.joblib'
        # model_file_name=f'/content/drive/MyDrive/model/combined benign_max_scaler-5.joblib'
        joblib.dump(min_max_scaler, model_file_name)
        print(f"Saving MinMax: {model_file_name}")
    else:
        min_max_scaler = joblib.load(minmax_model_to_load)
        print("Loading MinMax" + " " + minmax_model_to_load)

    print(len(AE_cols))
    normalized_selected_data = min_max_scaler.transform(selected_data[AE_cols].values)

    tensor_data = torch.tensor(normalized_selected_data, dtype=torch.float32)
    return tensor_data, normalized_selected_data
