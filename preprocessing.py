import pandas as pd


def set_target(row):
    if row['eventsource'] == 's3.amazonaws.com':
        return row['requestparameters_bucketname']
    elif row['eventsource'] == 'dynamodb.amazonaws.com':
        return row['requestparameters_tablename']
    else:
        raise Exception('No Target Type Avialable')


def preprocess(dataset_name, data, window_size=1):
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
        print('#################################for Airline and VOD only!!')
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


def feature_extraction(data):
    # Calculate the percentage of 'readonly == False' for each group and add it as a new column
    data['f_write_access_percentage'] = data.groupby(['time_window', 'source'])['readonly'].transform(
        lambda x: (x == False).mean() * 100)
    # data['f_read_readwrite_ratio'] = data.groupby(['time_window', 'source'])['readonly'].transform(lambda x: (x == False).mean() * 100)
    # data['f_count_readwrite'] = data.groupby(['time_window', 'source']).readonly.transform(lambda x: (x == 'False').count())
    # data['f_count_read'] = data.groupby(['time_window', 'source']).readonly.transform(lambda x: (x == 'True').count())
    # data['f_read_readwrite_ratio'] = data.apply(lambda row: 0 if row['f_count_readwrite'] == 0 else row['f_count_read'] / (row['f_count_readwrite']),axis=1)

    # Count 'private' and 'public' bucket_access for each time_window and source
    data['f_count_private'] = data.groupby(['time_window', 'source']).bucket_access.transform(
        lambda x: (x == 'private').count())
    data['f_count_public'] = data.groupby(['time_window', 'source']).bucket_access.transform(
        lambda x: (x == 'public').count())

    # Set ratio to 0 when both counts are 0, otherwise calculate the normal ratio
    data['f_private_public_ratio'] = data.apply(
        lambda row: 0 if row['f_count_public'] == 0 else row['f_count_private'] / (row['f_count_public']), axis=1)

    # Calculate the percentage of 'readonly == False' for each group and add it as a new column
    data['f_public_access_percentage'] = data.groupby(['time_window', 'source'])['bucket_access'].transform(
        lambda x: (x == 'public').mean() * 100)

    # Binary features for the presence of 'public' and 'private' accesses
    data['f_has_private_access'] = (data['f_count_private'] > 0).astype(int)
    data['f_has_public_access'] = (data['f_count_public'] > 0).astype(int)
    return data


def combine_datasets(load_key_array, loaders ):
    load_key_array = ['small benign file', 'airlines july benign', 'vod oct anomaly'] # 'vod oct benign'
    for k in load_key_array:
        loaders[k].load_dataset()
    all_datasets =  { 'small benign file' : None,
                      'airlines july benign' : None,
                      'vod_oct benign' : None,
                      'vod oct anomaly' : None }

    for k in load_key_array:
        print(loaders[k].dataset_name, len(loaders[k].data))

    for k in load_key_array:
        all_datasets[k] = preprocess(loaders[k].dataset_name, loaders[k].data )

    for k in load_key_array:
        print(all_datasets[k].columns)

    keys = list(all_datasets.keys())
    df_combined = pd.concat([all_datasets[keys[0]],all_datasets[keys[1]],all_datasets[keys[3]]],axis=0)
    print(df_combined.shape)

    print(len(df_combined))
    df_combined = df_combined.dropna(subset=['source'])
    df_combined = df_combined[df_combined['source'] != 'VODApplication-ErrorHandlerRole-YYQFY381BRQK']
    df_combined.reset_index(drop=True, inplace=True)
    print(len(df_combined))
    display(nans(df_combined))
    missing_values_table(df_combined)

    print(df_combined.anomaly.sum())

    return df_combined

# print(len(df_combined))
# df_combined = combine_datasets()
# data = df_combined
# # dataset_name = 'combined SBABVB'
# dataset_name = 'combined SBABVA'

data = current_loader.data
dataset_name = current_loader.dataset_name


print(dataset_name)
print(len(data))
print(data.anomaly.sum())