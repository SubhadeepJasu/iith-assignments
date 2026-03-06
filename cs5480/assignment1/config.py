"""
Configuration for wandb
"""
# Author: Subhadeep Jasu

class Config:
    """
    Configuration for FNN run or sweep.
    """
    def __init__(self, config:dict=None):
        if config is None:
            self.cv_method = 'bayes'
            self.cv_metric = 'accuracy'
            self.cv_goal = 'maximize'
            self.optimizers = ['sgd']
            self.epochs = [5]
            self.learning_rates = [1e-3]
            self.batch_sizes = [16]
            self.fc_layer_sizes = [8]
            self.fc_layer_depths = [1]
            self.weight_decays = [0]
            self.activations = ['sigmoid']
            self.weight_inits = ['uniform']
        else:
            if 'parameters' in config:
                self.cv_method = config['method']
                self.cv_metric = config['metric']['name']
                self.cv_goal = config['metric']['goal']
                self.optimizers = config['parameters']['values']['optimizer']
                self.epochs = config['parameters']['values']['epochs']
                self.learning_rates = config['parameters']['values']['learning_rate']
                self.batch_sizes = config['parameters']['values']['batch_size']
                self.fc_layer_sizes = config['parameters']['values']['fc_layer_size']
                self.fc_layer_depths = config['parameters']['values']['fc_layer_depth']
                self.weight_decays = config['parameters']['values']['weight_decay']
                self.activations = config['parameters']['values']['activation']
                self.weight_inits = config['parameters']['values']['weight_init']
            else:
                self.cv_method = 'bayes'
                self.cv_metric = 'accuracy'
                self.cv_goal = 'maximize'
                self.optimizers = config['optimizer']
                self.epochs = config['epochs']
                self.learning_rates = config['learning_rate']
                self.batch_sizes = config['batch_size']
                self.fc_layer_sizes = config['fc_layer_size']
                self.fc_layer_depths = config['fc_layer_depth']
                self.weight_decays = config['weight_decay']
                self.activations = config['activation']
                self.weight_inits = config['weight_init']

    def get_for_run(self):
        """
        Get the config as a dictionary for use with wandb run.
        """
        return {
            'optimizer': self.optimizers[0],
            'epochs': self.epochs[0],
            'learning_rate': self.learning_rates[0],
            'batch_size': self.batch_sizes[0],
            'fc_layer_size': self.fc_layer_sizes[0],
            'fc_layer_depth': self.fc_layer_depths[0],
            'weight_decay': self.weight_decays[0],
            'activation': self.activations[0],
            'weight_init': self.weight_inits[0],
        }


    def get_for_sweep(self):
        """
        Get the config as a dictionary for use with wandb sweep.
        """
        return {
            'method': self.cv_method,
            'metric': {
                'name': self.cv_method,
                'goal': self.cv_goal
            },
            'parameters': {
                'optimizer': { 'values': self.optimizers },
                'epochs': { 'values': self.epochs },
                'learning_rate': { 'values': self.learning_rates },
                'batch_size': { 'values': self.batch_sizes },
                'fc_layer_size': { 'values': self.fc_layer_sizes },
                'fc_layer_depth': { 'values': self.fc_layer_depths },
                'weight_decay': { 'values': self.weight_decays },
                'activation': { 'values': self.activations },
                'weight_init': { 'values': self.weight_inits },
            }
        }
