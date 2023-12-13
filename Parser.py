import json
import pandas as pd
#
# # Load the provided CSV file
# file_path = 'C:/Users/shach/Downloads/b7991125-d19e-44a2-b08d-caca77447afd.csv'
# data = pd.read_csv(file_path)
#
#
# # Function to parse the JSON-like strings in the dataframe
# def parse_json_string(json_string):
#     # Removing curly braces and splitting the string by commas
#     key_value_pairs = json_string.strip('{}').split(', ')
#     # Splitting each pair by '=' and forming a dictionary
#     data_dict = {pair.split('=')[0]: pair.split('=')[1] for pair in key_value_pairs}
#     try:
#         return data_dict
#     except json.JSONDecodeError:
#         return {}
#
#
# def parse_json_string_other(json_string):
#     # Removing curly braces and splitting the string by commas
#     key_value_pairs = json_string.strip('{}').split(',')
#     key_value_pairs = [s.replace('"', '') for s in key_value_pairs]
#     # Splitting each pair by '=' and forming a dictionary
#     data_dict = {pair.split(':')[0]: pair.split(':')[1] for pair in key_value_pairs}
#     try:
#         return data_dict
#     except json.JSONDecodeError:
#         return {}
#
#
# # Filtering events from S3 only
# s3_events = data[data['eventsource'] == 's3.amazonaws.com']
#
# # Extracting required fields
# s3_events['time'] = s3_events['eventtime']
# s3_events['operation'] = s3_events['eventname']
# s3_events['readonly'] = s3_events['readonly']
#
# # Extracting 'source' from 'useridentity'
# user_dict = s3_events['useridentity'].apply(parse_json_string)
# s3_events['source'] = user_dict.apply(lambda d: d.get("username", ''))
#
# # Extracting 'target' and 'key' from 'requestparameters'
# bucket_dict = s3_events['requestparameters'].apply(parse_json_string_other)
# s3_events['target'] = bucket_dict.apply(lambda d: d.get('bucketName', ''))
# s3_events['key'] = bucket_dict.apply(lambda d: d.get("key", 'None'))
#
# # Defining buckets and their access types
# bucket_access_types = {
#     'public': ["test-data-area", "amplify-public-bucket"],
#     'private': [
#         "asafs-athena", "sagemaker-studio-l7tewy0ax0r", "amplify-booking-reports",
#         "pdf-files-bucket-1", "amplify-trivia-dev-161621-deployment", "omer-test-logs",
#         "amplify-trivia-dev-162817-deployment", "aws-cloudtrail-logs-696714140038-8effbb5f"
#     ]
# }
#
# # Function to determine bucket access type
# def get_bucket_access(bucket_name):
#     for access_type, buckets in bucket_access_types.items():
#         if any(bucket in bucket_name for bucket in buckets):
#             return access_type
#     return 'unknown'
#
#
# s3_events['bucket_access'] = s3_events['target'].apply(get_bucket_access)
# # Selecting the required columns
# final_df = s3_events[['time', 'operation', 'readonly', 'source', 'target', 'key', 'bucket_access']]
# # Saving the dataframe as a CSV file
# output_csv_path = 'parsed_logs2011.csv'
# final_df.to_csv(output_csv_path, index=False)




