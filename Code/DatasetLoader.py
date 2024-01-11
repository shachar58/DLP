import os
import pandas as pd


def _extract_dataset_name(csv_name):
    """
    Internal method to extract the dataset name from the CSV file name.

    Parameters:
        csv_name (str): The CSV file name.

    Returns:
        str: The extracted dataset name.
    """
    base_name = os.path.basename(csv_name)
    name, _ = os.path.splitext(base_name)
    return name.replace('_', ' ')


class DatasetLoader:
    """
    DatasetLoader class for loading datasets from CSV files.

    Attributes:
        base_drive_path (str): Base path for the dataset storage.
        user_path (str): Additional path for user-specific datasets.
        data (DataFrame): Dataframe to hold the dataset.
        dataset_name (str): Name of the dataset.
        dataset_csv_name (str): Name of the CSV file for the dataset.
    """

    def __init__(self, dataset_csv_name, base_path='data/', user_path=''):
        """
        Initializes the DatasetLoader with the specified CSV file name and paths.

        Parameters:
            dataset_csv_name (str): The name of the CSV file to load.
            base_path (str): The base directory path where the datasets are stored.
            user_path (str): Optional additional path for user-specific data.
        """
        self.base_drive_path = base_path
        self.user_path = user_path
        self.data = pd.DataFrame()
        self.dataset_name = ''
        self.dataset_csv_name = ''
        self._init_dataset(dataset_csv_name)

    def _init_dataset(self, dataset_csv_name):
        """
        Internal method to initialize the dataset.

        Parameters:
            dataset_csv_name (str): The name of the CSV file to load.
        """
        self.dataset_csv_name = dataset_csv_name
        self.dataset_name = _extract_dataset_name(dataset_csv_name)

    def load_dataset(self):
        """
        Loads the dataset from the CSV file into a DataFrame.

        Returns:
            self: Returns the instance itself for method chaining.
        """
        full_path = self.get_full_path(self.dataset_csv_name)
        self.data = pd.read_csv(full_path, low_memory=False)
        self.data.columns = [col.lower() for col in self.data.columns]
        print(f'{self.dataset_name} loaded.')
        return self

    def get_full_path(self, relative_path=''):
        """
        Constructs the full path for a given relative path.

        Parameters:
            relative_path (str): A relative path to append to the base path.

        Returns:
            str: The full path.
        """
        return os.path.join(self.base_drive_path, self.user_path, relative_path)

    def get_base_path(self):
        """
        Returns the base path of the dataset storage.

        Returns:
            str: The base path.
        """
        return self.base_drive_path

    def get_user_path(self):
        """
        Returns the user-specific path for the datasets.

        Returns:
            str: The user path.
        """
        return self.user_path

    def get_is_data_loaded(self):
        """
        Checks if the data has been loaded into the DataFrame.

        Returns:
            bool: True if data is loaded, False otherwise.
        """
        return not self.data.empty

    def print_configuration_info(self):
        """
        Prints the configuration information of the dataset loader.
        """
        print(f'{self.dataset_name}, Data loaded: {self.get_is_data_loaded()}')
