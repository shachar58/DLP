import json


class Parser:

    def __init__(self, dataset_loader, bucket_access=''):
        self.data = dataset_loader.data.copy()
        if bucket_access == '':
            self.bucket_access = {'public': ["test-data-area", "amplify-public-bucket"],
                                  'private': ["asafs-athena", "sagemaker-studio-l7tewy0ax0r", "amplify-booking-reports",
                                  "pdf-files-bucket-1", "amplify-trivia-dev-161621-deployment", "omer-test-logs",
                                  "amplify-trivia-dev-162817-deployment", "aws-cloudtrail-logs-696714140038-8effbb5f"]}
        else:
            self.bucket_access = bucket_access

    # Functions to parse the JSON-like strings in the dataframe
    def parse_useridentity(json_string):
        # Removing curly braces and splitting the string by commas
        key_value_pairs = json_string.strip('{}').split(', ')
        # Splitting each pair by '=' and forming a dictionary
        data_dict = {pair.split('=')[0]: pair.split('=')[1] for pair in key_value_pairs}
        try:
            return data_dict
        except json.JSONDecodeError:
            return {}

    def parse_requestparameters(json_string):
        # Removing curly braces and splitting the string by commas
        key_value_pairs = json_string.strip('{}').split(',')
        key_value_pairs = [s.replace('"', '') for s in key_value_pairs]
        # Splitting each pair by '=' and forming a dictionary
        data_dict = {pair.split(':')[0]: pair.split(':')[1] for pair in key_value_pairs}
        try:
            return data_dict
        except json.JSONDecodeError:
            return {}

    def rename_s3_values(self):
        # Filtering events from S3 only
        s3_events = self.data[self.data['eventsource'] == 's3.amazonaws.com']
        # Extracting required fields
        s3_events['time'] = s3_events['eventtime']
        s3_events['operation'] = s3_events['eventname']
        s3_events['readonly'] = s3_events['readonly']
        # Extracting 'source' from 'useridentity'
        user_dict = s3_events['useridentity'].apply(self.parse_useridentity())
        s3_events['source'] = user_dict.apply(lambda d: d.get("username", ''))
        # Extracting 'target' and 'key' from 'requestparameters'
        bucket_dict = s3_events['requestparameters'].apply(self.parse_requestparameters)
        s3_events['target'] = bucket_dict.apply(lambda d: d.get('bucketName', ''))
        s3_events['key'] = bucket_dict.apply(lambda d: d.get("key", 'None'))
        return s3_events

# Function to determine bucket access type
    def get_bucket_access(self, bucket_name):
        for access_type, buckets in self.bucket_access.items():
            if any(bucket in bucket_name for bucket in buckets):
                return access_type
            else:
                print("Labeled Unknown!")
                return 'unknown'

    def parse(self):
        s3_events = self.rename_s3_values()
        s3_events['bucket_access'] = s3_events['target'].apply(self.get_bucket_access)
        # Selecting the required columns
        final_df = s3_events[['time', 'operation', 'readonly', 'source', 'target', 'key', 'bucket_access']]
        # Saving the dataframe as a CSV file
        output_csv_path = 'parsed_logs2011.csv'
        final_df.to_csv(output_csv_path, index=False)
        return final_df
