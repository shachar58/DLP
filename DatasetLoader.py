import os
import pandas as pd


class DatasetLoader:

    def __init__(self, dataset_csv_name, base_drive_path='/content/drive/MyDrive/', user_path=''):
        self.base_drive_path = base_drive_path
        self.user_path = user_path
        self.data = pd.DataFrame()
        self.dataset_name = ''
        self.dataset_csv_name = ''
        self._init_dataset(dataset_csv_name)

    def _init_dataset(self, dataset_csv_name):
        self.dataset_csv_name = dataset_csv_name
        self.dataset_name = self._extract_dataset_name(dataset_csv_name)

    def load_dataset(self):
        full_path = self.get_full_path(self.dataset_csv_name)
        self.data = pd.read_csv(full_path, low_memory=False)
        self.data.columns = [col.lower() for col in self.data.columns]
        print(f'{self.dataset_name} loaded.')
        return self

    def _extract_dataset_name(self, csv_name):
        base_name = os.path.basename(csv_name)
        name, _ = os.path.splitext(base_name)
        return name.replace('_', ' ')

    def get_full_path(self, relative_path=''):
        return os.path.join(self.base_drive_path, self.user_path, relative_path)

    def get_base_path(self):
        return self.base_drive_path

    def get_user_path(self):
        return self.user_path

    def get_is_data_loaded(self):
        if not self.data.empty:
            return True
        return False

    def print_configuration_info(self):
        print(f'{self.dataset_name}')
        print(f'Data loaded: {self.get_is_data_loaded()}')
