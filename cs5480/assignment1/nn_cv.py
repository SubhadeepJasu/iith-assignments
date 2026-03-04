"""
Configurable Feed-Forward Network
"""
# Author: Subhadeep Jasu

from typing import Any
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.regularizers import l2
import wandb
from config import Config

class NeuralNetCV():
    """
    A configurable feed-forward network with wandb logging capabilities
    and built-in cross validation via wandb sweeps.
    """

    instance = None
    _EPSILON = 0.01

    def __init__(self, config:Config|None=None, project='ocr'):
        self.wandb_config = config if config is not None else Config()
        self.training_set = {}
        self.testing_set = {}
        self.validation_set = tuple()
        self.wandb_params = {
            'entity': '',
            'project': project,
            'run': '',
            'sweep': '',
            'bestRun': {},
            'metricTrend': []
        }
        self.n_classes = 10
        self.sweeping = False


    def set_data(
            self,
            training_set:tuple[Any, Any],
            testing_set:tuple[Any, Any],
            validation_set:tuple[Any, Any],
            num_classes = 10
        ):
        """
        Set the dataset for training with cross validation.
        """
        self.training_set = {
            'x': training_set[0],
            'y': training_set[1]
        }
        self.testing_set = {
            'x': testing_set[0],
            'y': testing_set[1]
        }
        self.validation_set = validation_set
        self.n_classes = num_classes


    def _reset_best_run(self):
        self.wandb_params['bestRun'] = {
            'accuracy': -99999999 if self.wandb_config.cv_goal == 'maximize' else 99999999,
            'loss': -99999999 if self.wandb_config.cv_goal == 'maximize' else 99999999,
            'config': None,
            'url': None,
            'id': None,
            'model': None,
            'history': None
        }


    @staticmethod
    def _train(config=None):
        with wandb.init(
            project=NeuralNetCV.instance.wandb_params['project'],
            config=config,
            allow_val_change=NeuralNetCV.instance.sweeping,
            settings=wandb.Settings(console="off")
        ) as run:
            if NeuralNetCV.instance.sweeping:
                config = wandb.config.as_dict()
            else:
                print(config)

            if config['activation'] == 'relu' and config['weight_init'] == 'glorot_uniform':
                run.finish()
                return run

            model = Sequential()
            model.add(Input(shape=(784,)))
            for _ in range(config['fc_layer_depth']):
                model.add(
                    Dense(
                        config['fc_layer_size'],
                        activation=config['activation'],
                        kernel_initializer=config['weight_init'],
                        kernel_regularizer=l2(config['weight_decay'])
                    )
                )

            model.add(
                Dense(
                    NeuralNetCV.instance.n_classes,
                    activation='softmax',
                    kernel_initializer=config['weight_init']
                )
            )

            if config['optimizer'] == 'sgd':
                _opt = SGD(learning_rate=config['learning_rate'])
            elif config['optimizer'] == 'adam':
                _opt = Adam(learning_rate=config['learning_rate'])
            elif config['optimizer'] == 'momentum':
                _opt = SGD(learning_rate=config['learning_rate'], momentum=0.9)
            elif config['optimizer'] == 'nesterov':
                _opt = SGD(learning_rate=config['learning_rate'], nesterov=True)
            elif config['optimizer'] == 'rmsprop':
                _opt = RMSprop(learning_rate=config['learning_rate'])
            else:
                _opt = Nadam(learning_rate=config['learning_rate'])

            model.compile(
                optimizer=_opt,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            history = model.fit(
                NeuralNetCV.instance.training_set['x'],
                NeuralNetCV.instance.training_set['y'],
                epochs=config['epochs'],
                validation_data=NeuralNetCV.instance.validation_set,
                batch_size=config['batch_size'],
                verbose=0
            )
            loss, acc = model.evaluate(
                NeuralNetCV.instance.testing_set['x'],
                NeuralNetCV.instance.testing_set['y'],
                verbose=0
            )
            NeuralNetCV.instance.wandb_params['metricTrend'].append(acc)
            wandb.log({'summary': acc, 'accuracy': acc, 'run': str(datetime.now()), 'loss': loss})
            run.finish()
            __path = run.path.split('/')
            if NeuralNetCV.instance.wandb_params['entity'] == '':
                NeuralNetCV.instance.wandb_params['entity'] = __path[0]

            if NeuralNetCV.instance.wandb_params['bestRun'] is None or \
                    (NeuralNetCV.instance.wandb_config.cv_goal == 'maximize' and
                        acc > NeuralNetCV.instance.wandb_params['bestRun']['accuracy']) or \
                    (NeuralNetCV.instance.wandb_config.cv_goal == 'minimize' and
                        acc < NeuralNetCV.instance.wandb_params['bestRun']['accuracy']):
                NeuralNetCV.instance.wandb_params['bestRun'] = {
                    'accuracy': acc,
                    'loss': loss,
                    'config': config,
                    'url': run.get_url(),
                    'id': run.id,
                    'model': model,
                    'history': history.history
                }

            return run


    def train_once(self, config:Config|None=None):
        """
        Train the model once.
        """
        if config is None:
            __config = self.wandb_config.get_for_run()
        else:
            __config = config.get_for_run()

        NeuralNetCV.instance = self
        self._reset_best_run()
        self.sweeping = False
        run = NeuralNetCV._train(__config)
        run_id = self.wandb_params['run']
        return self.wandb_params['bestRun']['model'], run.get_url(), run_id


    def train_cv(self, config:Config|None=None, num_iter=10):
        """
        Train the model in multiple iterations to find the best model to fit the data
        well enough.
        """
        if config is None:
            __config = self.wandb_config.get_for_sweep()
        else:
            __config = config.get_for_sweep()

        project = self.wandb_params['project']
        self.wandb_params['sweep'] = wandb.sweep(__config, project=project)

        sweep_id = self.wandb_params['sweep']
        NeuralNetCV.instance = self
        self._reset_best_run()
        self.sweeping = True
        wandb.agent(sweep_id, NeuralNetCV._train, count=num_iter)
        wandb.finish()
        self.sweeping = False
        entity = self.wandb_params['entity']
        wandb_api = wandb.Api()
        sweep = wandb_api.sweep(f"{entity}/{project}/{sweep_id}")
        best_run = self.wandb_params['bestRun']
        return best_run['model'], best_run['id'], sweep.url, sweep_id


    def evaluate(self, x=None, y=None, batch_size=None):
        """
        Get the best prediction results of the last or best run after `train_cv()` call.
        """
        loss, accuracy = self.wandb_params['bestRun']['model'].evaluate(x, y, batch_size, verbose=0)
        return {
            'loss': loss,
            'accuracy': accuracy
        }


    def get_best_config(self):
        """
        Get the best config of the last or best run after `train_cv()` call.
        """
        return self.wandb_params['bestRun']['config']


    def plot_metric_trend(self):
        """
        Plot the trend of the metric across all runs after `train_cv()` call.
        """
        metric_trend = self.wandb_params['metricTrend']
        if not metric_trend:
            print("No metric trend data found.")
            return

        best_metric_at_run = []
        for i in range(len(metric_trend)):
            if self.wandb_config.cv_goal == 'maximize':
                best_metric_at_run.append(max(metric_trend[:i+1]))
            else:
                best_metric_at_run.append(min(metric_trend[:i+1]))
        plt.figure(figsize=(10, 5))
        plt.plot(metric_trend, marker='o', label='Actual Metric')
        plt.plot(best_metric_at_run, marker='s', label='Best Metric So Far')
        plt.title(f"{self.wandb_config.cv_metric} Trend Across Runs")
        plt.xlabel('Run')
        plt.ylabel(self.wandb_config.cv_metric)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_history(self):
        """
        Plot the training history of the best run after `train_cv()` call.
        """
        history = self.wandb_params['bestRun']['history']
        if history is None:
            print("No history found for the best run.")
            return

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
