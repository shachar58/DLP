import os
import shutil
import time


class exp:
    """This class is representing the paths of the directories in the experiment object type"""
    def __init__(self, exp_dir, scripts_ls, current_run_description=''):
        self.exp_dir = 'experiments' + '/' + exp_dir

        if current_run_description == '':
            print('''NO DESCRIPTION FOR CURRENT RUN DIR''')
            self.current_exp_run_dir = self.exp_dir+'/'+str(time.time()).split('.')[0]
        else:
            self.current_exp_run_dir = self.exp_dir+'/'+current_run_description

        """Experiment dirs"""
        self.plots_dir = self.current_exp_run_dir+'/'+'plots/'
        self.logs_dir = self.current_exp_run_dir + '/' + 'logs/'
        self.scripts_dir = self.current_exp_run_dir + '/' + 'scripts/'
        self.input_data = self.current_exp_run_dir + '/' + 'input_data/'
        self.results = self.current_exp_run_dir + '/' + 'results/'
        self.wgt_dir = self.current_exp_run_dir + '/' + 'wgt/'
        self.files_ls = scripts_ls

        """Create dirs"""
        print('exp dir: ', self.current_exp_run_dir)
        if not os.path.exists(self.current_exp_run_dir):
            dirs_ls = [self.plots_dir, self.wgt_dir, self.logs_dir, self.scripts_dir, self.input_data, self.results]
            for dir in dirs_ls:
                os.makedirs(dir)

            """save scripts and config to scripts dir"""
            print('save scripts')
            for script in scripts_ls:
                warn = os.popen('cp '+script + ' ' + self.scripts_dir + script)
                print('warn', warn)
                if '_wrap_close' in str(warn):
                    print('Execute "save scripts" on window')
                    shutil.copyfile(script, self.scripts_dir + script)
