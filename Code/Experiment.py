import os
import time


class Experiment:
    """
    Experiment class to manage the directory structure for a given experiment.

    This class creates and manages directories for storing plots, logs, input data,
    results, and models related to a specific experimental run. The directory
    structure is based on the experiment name and an optional description for the
    current run.

    Attributes:
        exp_dir (str): Base directory for the experiment.
        current_exp_run_dir (str): Directory for the current run of the experiment.
        plots_dir (str): Directory for storing plots.
        logs_dir (str): Directory for storing logs.
        input_data (str): Directory for storing input data.
        results (str): Directory for storing results.
        models_dir (str): Directory for storing models.
    """

    def __init__(self, exp_dir, current_run_description=''):
        """
        Initializes the Experiment object with the experiment directory and
        an optional description for the current run.

        Parameters:
            exp_dir (str): Name of the experiment directory.
            current_run_description (str, optional): Description for the current run.
                Defaults to a timestamp if not provided.
        """
        self.exp_dir = 'experiments' + '/' + exp_dir

        # Set current experiment run directory, use timestamp if no description provided
        if current_run_description == '':
            print('NO DESCRIPTION FOR CURRENT RUN DIR')
            self.current_exp_run_dir = self.exp_dir + '/' + str(time.time()).split('.')[0]
        else:
            self.current_exp_run_dir = self.exp_dir + '/' + current_run_description

        # Initialize directories for various components of the experiment
        self.plots_dir = self.current_exp_run_dir + '/plots/'
        self.logs_dir = self.current_exp_run_dir + '/logs/'
        self.input_data = self.current_exp_run_dir + '/input_data/'
        self.results = self.current_exp_run_dir + '/results/'
        self.models_dir = self.current_exp_run_dir + '/models/'

        # Create the directories if they do not exist
        print('exp dir: ', self.current_exp_run_dir)
        if not os.path.exists(self.current_exp_run_dir):
            dirs_ls = [self.plots_dir, self.models_dir, self.logs_dir, self.input_data, self.results]
            for dir in dirs_ls:
                os.makedirs(dir)

