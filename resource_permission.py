import boto3
import pandas as pd
import csv


def response_handle(response):
    responses_list = []
    response_dict = {}
    for line in response['PolicyDocument']['Statement']:
        if not isinstance(line, dict):
            responses_list.append(response['PolicyDocument']['Statement'])
            return responses_list
        response_dict['Resource'] = line['Resource']
        response_dict['Action'] = line['Action']
        response_dict['Effect'] = line['Effect']
        responses_list.append(response_dict)
    return responses_list


def get_policy_func(arg, file_row):
    global role_dict
    global data
    global policy_dict

    def create_role_changes():
        try:
            row_role = file_row['userIdentity_sessionContext_sessionIssuer_userName']
        except:
            row_role = file_row['requestParameters_bucketName']

        time = file_row['eventTime']
        TargetName_parameter = file_row['requestParameters_tableName'] if file_row['eventSource'] == 'dynamodb.amazonaws.com' else file_row['requestParameters_bucketName']
        TargetType_parameter = file_row['eventSource']
        Operation_parameter = file_row['eventName']
        readOnly_parameter = file_row['readOnly']

        return time, TargetName_parameter, TargetType_parameter, Operation_parameter, readOnly_parameter, row_role

    def get_permission_changes():

        anomaly = 0
        if file_row['requestParameters_key'] != 0 and 'confirm_leak' in file_row['requestParameters_key']:
            anomaly = 1

        filtered_df = permission_file[(permission_file['requestParameters_roleName'] == role) & (permission_file['eventTime'] < time)]
        if filtered_df.empty:
            data.append([file_row['eventTime'], role, TargetName, TargetType, Operation, readOnly, 0, 0, 0, anomaly])
        else:
            nearest_event = filtered_df.iloc[-1]
            if nearest_event['eventName'] == 'DeleteRolePolicy':
                data.append([file_row['eventTime'], role, TargetName, TargetType, Operation, readOnly, 0, 0, 1, anomaly])
            elif nearest_event['eventName'] == 'PutRolePolicy' and 'S3' in nearest_event['requestParameters_policyDocument']:
                data.append([file_row['eventTime'], role, TargetName, TargetType, Operation, readOnly, 1, 0, 0, anomaly])
            elif nearest_event['eventName'] == 'PutRolePolicy' and 'dynamodb' in nearest_event['requestParameters_policyDocument']:
                data.append([file_row['eventTime'], role, TargetName, TargetType, Operation, readOnly, 0, 1, 0, anomaly])

    def create_policy_list():
        try:
            row_role = file_row['userIdentity_sessionContext_sessionIssuer_userName']
        except:
            row_role = file_row['requestParameters_bucketName']

        TargetName_parameter = file_row['requestParameters_tableName'] if file_row['eventSource'] == 'dynamodb.amazonaws.com' else file_row['requestParameters_bucketName']
        TargetType_parameter = file_row['eventSource']
        Operation_parameter = file_row['eventName']
        readOnly_parameter = file_row['readOnly']

        if row_role in role_dict:
            existing_policy_list = role_dict[row_role]
        else:
            existing_policy_list = iam.list_role_policies(RoleName=row_role)
            # Remove metadata
            existing_policy_list = existing_policy_list['PolicyNames']
            role_dict[row_role] = existing_policy_list
        return TargetName_parameter, TargetType_parameter, Operation_parameter, readOnly_parameter, \
            existing_policy_list, row_role

    def update_policy(policy):
        x = permission_file.head(n=1)
        if file_row['eventTime'] > permission_file.head(n=1)['eventTime'] and policy in policy_dict:
            if x['eventName'] == 'DeleteRolePolicy':
                policy_dict[x['requestParameters_policyName']] = {}
            elif x['eventName'] == 'PutRolePolicy':
                x['requestParameters_policyDocument'] = x['requestParameters_policyDocument']['Statement']

    def get_policy_permissions():
        for policy in policy_list:
            update_policy(policy)
            if policy in policy_dict:
                policy_response = policy_dict[policy]
            else:
                policy_response = iam.get_role_policy(RoleName=role, PolicyName=policy)
            # Handle different formats of responses
            response_list = policy_dict[policy] = response_handle(policy_response)

            # Retrieve each line from the response(some policies have multiple configurations)
            for response_data in response_list:
                data.append([role, TargetName, TargetType, policy, response_data['Resource'], response_data['Action'],
                             response_data['Effect'], Operation, readOnly, file_row['eventTime']])

    if arg == 1:
        TargetName, TargetType, Operation, readOnly, policy_list, role = create_policy_list()
        get_policy_permissions()
    elif arg == 0:
        time, TargetName, TargetType, Operation, readOnly, role = create_role_changes()
        get_permission_changes()
        return


if __name__ == '__main__':

    # Create Boto3 clients
    iam = boto3.client('iam')
    s3_client = boto3.client('s3')

    # Files as objects
    # Benign
    # permission_file = pd.read_csv('D:/#Projects/Overwatch/data/output/LogStreamParser/AL/Dec/benign-og/permission_changes_logs.csv')
    # api_file = pd.read_csv('D:/#Projects/Overwatch/data/output/LogStreamParser/AL/Dec/benign-og/storage_API_calls.csv')
    # Anomaly
    permission_file = pd.read_csv('D:/Projects/Overwatch/data/output/LogStreamParser/AL/Dec/anomaly-og/permission_changes_logs.csv')
    api_file = pd.read_csv('D:/Projects/Overwatch/data/output/LogStreamParser/AL/Dec/anomaly-og/storage_API_calls.csv')

    # # Prepare files columns
    # api_headers = ['eventTime', 'eventSource', 'eventName', 'readOnly', 'userIdentity_sessionContext_sessionIssuer_userName',
    #                'requestParameters_bucketName', 'requestParameters_tableName']
    # change_permission_headers = ['eventTime', 'eventName', 'requestParameters_roleName', 'requestParameters_policyName',
    #                              'requestParameters_policyDocument']

    """---------------------------------------definitions-------------------------------------------------"""

    # Arrange files
    api_file = api_file.sort_values('eventTime')
    permission_file = permission_file.sort_values('eventTime')

    api_file['eventTime'] = pd.to_datetime(api_file['eventTime'])
    permission_file['eventTime'] = pd.to_datetime(permission_file['eventTime'])
    api_file['requestParameters_key'] = api_file['requestParameters_key'].fillna(0)
    # Holds the entire data for the new dataframe/CSV file
    data = []
    # Holds the policies of roles that had been already seen
    role_dict = {}
    # Holds the permissions of policies that had been already seen
    policy_dict = {}

    api_file.apply(lambda file_row: get_policy_func(0, file_row), axis=1)

    # Dataframe columns
    headers = ['RoleName', 'TargetName', 'TargetType', 'PolicyName', 'Resource', 'Operations', 'Effect', 'OperationUsed',
               'ReadOnly', 'SampleTime', 'change_s3', 'change_dynamoDB', 'remove_policy']

    headers2 = ['SampleTime', 'RoleName', 'TargetName', 'TargetType', 'OperationUsed', 'ReadOnly',
                'change_s3', 'change_dynamoDB', 'remove_policy', 'Anomaly']

    # create a dataframe from the data pf the response gathered
    df_roles = pd.DataFrame(data=data, columns=headers2)

    # Convert SampleTime to time categorical columns
    df_roles['Year'] = df_roles['SampleTime'].dt.year
    df_roles['Month'] = df_roles['SampleTime'].dt.month
    df_roles['Day'] = df_roles['SampleTime'].dt.day
    df_roles['Hour'] = df_roles['SampleTime'].dt.hour
    df_roles['Minute'] = df_roles['SampleTime'].dt.minute
    df_roles['Second'] = df_roles['SampleTime'].dt.second
    df_roles['Millisecond'] = df_roles['SampleTime'].dt.microsecond // 1000

    # Define the new header with the updated columns
    headers3 = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Millisecond', 'RoleName', 'TargetName',
                'TargetType',
                'OperationUsed', 'ReadOnly', 'change_s3', 'change_dynamoDB', 'remove_policy', 'Anomaly']

    # Save the DataFrame to a CSV file
    # df_roles[headers3].copy().to_csv('relations_al_b1.csv', index=False)
    df_roles[headers3].copy().to_csv('relations_al_a3.csv', index=False)
    # df_roles.to_csv('relations_al_b1.csv', index=False, header=headers3)

    # with open('relations_al_b1.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     # writer.writerow(headers2)
    #     writer.writerow(headers3)
    #     for index, row in df_roles.iterrows():
    #         writer.writerow(row)
    #     file.close()
