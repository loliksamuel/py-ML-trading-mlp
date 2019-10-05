import os
import optuna
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
from os import path
from typing import Dict

from click._compat import raw_input
from sklearn.neural_network import MLPClassifier
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

from lib.env.reward import Reward_Strategy_BASE, RewardIncremental, RewardWeightedUnrealized, RewardPnL
from lib.data.providers.dates import ProviderDateFormat
from lib.data.providers import BaseDataProvider,  StaticDataProvider, ExchangeDataProvider
from lib.util.logger import init_logger




class Optuna:

    study_name    = None

    def __init__(self
                 , model_actor    : BaseRLModel = PPO2
                 , policy         : BasePolicy  = MlpLnLstmPolicy  #(ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
                 , reward_strategy: Reward_Strategy_BASE = RewardPnL  #IncrementalProfit
                 , exchange_args  : Dict = {}
                 , **kwargs):
        self.model_actor :BaseRLModel= model_actor
        self.policy                 = policy
        self.Reward_Strategy        = reward_strategy
        self.exchange_args          = exchange_args
        self.logger                 = kwargs.get('logger', init_logger(__name__, show_debug=kwargs.get('show_debug', True)))
        self.db_path                = kwargs.get('db_path', 'sqlite:///data/params.db')
        self.date_format            = kwargs.get('date_format', ProviderDateFormat.DATETIME_HOUR_24)
        self.data_path              = kwargs.get('data_path', 'data/input/coinbase-1h-btc-usd.csv')
        self.data_train_split_pct   = kwargs.get('train_split_percentage', 0.8)
        self.data_provider          = kwargs.get('data_provider', 'static')
        #self.columns_map            = kwargs.get('columns_map', {})
        self.n_envs                 = kwargs.get('n_envs'       , os.cpu_count())
        self.n_minibatches          = kwargs.get('n_minibatches', self.n_envs)
        self.model_logs_tb          = kwargs.get('tensorboard_path', os.path.join('data', 'logs_tb'))
        self.model_verbose          = kwargs.get('model_verbose', 1)
        self.do_load_raw_data: bool = kwargs.get('do_load_raw_data', True)
        self.features_to_add   : str  = kwargs.get('features_to_add', 'none')
        self.initialize_data(self.do_load_raw_data, self.features_to_add)
        self.initialize_db_optuna()#optimization for hyper param search

        self.logger.info(f'sucsessfully Initialize RLTrader study name {self.study_name} , open terminal, tensorboard --logdir={self.model_logs_tb}, click to http://localhost:6006/')

    def initialize_data(self, load_raw_data, features_to_add):
        self.logger.debug('Initializing data:')
        if self.data_provider == 'static':
            if not os.path.isfile(self.data_path):
                class_dir = os.path.dirname(__file__)
                self.data_path = os.path.realpath(os.path.join(class_dir, "../{}".format(self.data_path)))

            # data_columns = {'Date': 'Date',
            #                 'Open': 'Open'     , 'High': 'High'    ,    'Low': 'Low'  , 'Close': 'Close'     , 'Volume': 'Volume'
            #                ,'OpenVIX':'OpenVIX','HighVIX':'HighVIX', 'LowVIX':'LowVIX', 'CloseVIX':'CloseVIX', 'SKEW':'SKEW'# for *v2.csv files
            #                 }#VolumeFrom

            if load_raw_data:
                self.logger.info(f'Loading from disc raw data :{self.data_path} ')
                df = None
            else:
                d = self.data_path.replace('.csv', '_with_features.csv')
                self.logger.info(f'Loading from disc prepared data :{d} ')
                df = pd.read_csv(d)

            self.data_provider = StaticDataProvider(date_format  =self.date_format
                                                    , csv_data_path=self.data_path
                                                    , df =df
                                                    , do_prepare_data= load_raw_data
                                                    , features_to_add = features_to_add
                                                    )
        elif self.data_provider == 'exchange':
            self.data_provider = ExchangeDataProvider(**self.exchange_args)

        self.logger.debug(f'Successfully Initialized data , \nFeature list={self.data_provider.columns}')






    def initialize_db_optuna(self):
        self.logger.debug('Initializing Optuna and get best model from db')
        try:
            mlp = MLPClassifier         (random_state=5, hidden_layer_sizes= (250,150,100,50,20,10,5,), shuffle=False, activation='relu', solver='adam', batch_size=100, max_iter=200, learning_rate_init=0.001)#50.86 %  #activation=('identity', 'logistic', 'tanh', 'relu'),  solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'
            #                              , nminibatches=1)
            self.study_name = f'{mlp.__class__.__name__}__{mlp.act_model.__class__.__name__}'

        except:
            self.study_name = f'ErrorModel__ErrorPolicy__ErrorStrategy {mlp}'
        #run  we can load up the study from the sqlite database we told Optuna to create.
        self.optuna_study = optuna.create_study( study_name     =self.study_name
                                               , storage        =self.db_path
                                               , direction      ='minimize'
                                               , load_if_exists =True)

        self.logger.debug(f'Successfully Initialized Optuna , study_name={self.study_name}')

        try:
            self.logger.debug(f'found in db  {len(self.optuna_study.trials)} trials , Best value (minimum)={self.optuna_study.best_value} , params={self.optuna_study.best_params.items()}')# or {self.optuna_study.best_trial.params.items()}')

        except:
            self.logger.debug('Error: No trials have been finished yet.')


    def optuna_get_model_params(self):#get_model_params that found in optuna
        params = self.optuna_study.best_trial.params
        return {
             'batch_size'   : int(params['batch_size'   ]) #241.999  (int) The number of steps to run for each environment per update. aka ExperienceHorizon. must be > n_minibatch
            ,'hidden_size'  :     params['hidden_size'  ]  #0.9     (float) aka future_reward_importance or decay or discount rate, determines the importance of future rewards.If=0 then agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.
            ,'learning_rate':     params['learning_rate']  #0.9    (float or callable) The learning rate, it can be a function
            ,'epoch'        :     params['epoch'        ]  #11.999 (float) Entropy coefficient for the loss calculation. the higher the value the more explore
            ,'dropout'      :     params['dropout'      ]  #0.2    (float or callable) Clip factor for limiting the change in each policy update step. parameter specific to the OpenAI implementation. If None is passed (default), then `dropout` (that is used for the policy) will be used. reduce volatility of Advantage KL
            }


    def optimize_agent_params(self, trial):

        return {#Defining Parameter Spaces
            'batch_size'    : int(trial.suggest_loguniform    ('batch_size'   , 8  , 512)),#float between 16–2048 in a logarithmic manner (16, 32, 64, …, 1024, 2048)
            'hidden_size'   :     trial.suggest_loguniform    ('hidden_size'  , 20 , 1000),#float   Discount Factor hidden_size Range 0.8 to 0.9997 default 0.99
            'learning_rate' :     trial.suggest_loguniform    ('learning_rate', 0.00001, 0.01),
            'epoch'         :     trial.suggest_loguniform    ('epoch'        , 10, 600),
            'dropout'       :     trial.suggest_uniform       ('dropout'      , 0.0 , 0.4),#floats in a simple, additive manner (0.0, 0.1, 0.2, 0.3, 0.4)
        }
    '''
      activations = [ 'sigmoid', 'softmax']#, 'softplus', 'softsign', 'sigmoid',  'tanh', 'hard_sigmoid', 'linear', 'relu']#best 'softmax', 'softplus', 'softsign'
      inits       = ['glorot_uniform']#, 'zero', 'uniform', 'normal', 'lecun_uniform',  'glorot_uniform',  'he_uniform', 'he_normal']#all same except he_normal worse
      optimizers  = ['RMSprop', 'SGD']#, 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] # same for all
      losses =      ['categorical_crossentropy']#, 'categorical_crossentropy']#['mse', 'mae']  #better=binary_crossentropy
      epochs =      [300, 800]  # , 100, 150] # default epochs=1,  better=100
      batch_size = [12,128]#],150,200]  # , 10, 20]   #  default = none best=32
      size_hiddens = [ 200, 600]  # 5, 10, 20] best = 100 0.524993 Best score: 0.525712 using params {'batch_size': 128, 'dropout': 0.2, 'epochs': 100, 'loss': 'binary_crossentropy', 'size_hidden': 100}
      lrs     =      [0.01, 0.001, 0.00001]#,0.03, 0.05, 0.07]#,0.001,0.0001,1,0.1,0.00001]#best 0.01 0.001 0.0001
      dropout =     [0.2]#, 0.2, 0.3, 0.4]  # , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    '''
    def optimize(self, n_trials: int = 10):
        self.logger.info(f'start optimizing  {n_trials} trials')

        try:
            self.optuna_study.optimize(  self.optimize_params#it is callable function where the learn process accurs!!
                                         , n_trials=n_trials#if `None`, there is no limitation on the number of trials.
                                         #, timeout=100100100 #seconds to run
                                         , n_jobs=1)#if n_jobs=1 it will not run in parralel
        except KeyboardInterrupt:
            pass

        self.logger.info(f'Finished optimizing. trials# in db : {len(self.optuna_study.trials)}')
        #self.logger.info(f'Best trial: {self.optuna_study.best_trial.value}')
        self.logger.info('Params: ')

        for key, value in self.optuna_study.best_trial.params.items():
            self.logger.info(f'    {key}: {value}')
        #self.optuna_study._storage.set_trial_state(trial_id, structs.TrialState.COMPLETE)
        #optuna.visualization.plot_intermediate_values(self.optuna_study)
        df = self.optuna_study.trials_dataframe()
        return df


    def optimize_params(self, trial
                        , n_epochs: int = 2  #  for optimization process 2 is ok,  for train need 5 milion
                        , n_tests_per_eval: int = 1):
        #we must not  hypertune model for the 20% of the test. so we split the train to 80 20
        x_train, x_test  = self.data_provider.split_data_train_test(self.data_train_split_pct)#0.8
        x_train, x_valid =            x_train.split_data_train_test(self.data_train_split_pct)#0.64
        del x_test


        model_params = self.optimize_agent_params(trial)
        mlp = MLPClassifier         (random_state=5, hidden_layer_sizes= (250,150,100,50,20,10,5,), shuffle=False, activation='relu', solver='adam', batch_size=100, max_iter=200, learning_rate_init=0.001)#50.86 %  #activation=('identity', 'logistic', 'tanh', 'relu'),  solver : {'lbfgs', 'sgd', 'adam'}, default 'adam'

                            #nminibatches     =1,


        error_last = -np.finfo(np.float16).max
        n_samples = len(x_train.df)
        steps = int(n_samples / n_epochs)
        attempt = 0
        for epoch in range(1, n_epochs+1): #(1, n_epochs+1):
            self.logger.info(f'{epoch}/{n_epochs} epochs. Training on small sample size {steps}  (time steps)')
            try:#learn
                mlp.fit          (self.x_train, self.y_train)
            except AssertionError:
                raise


            if trades_s     < (steps * 0.05):
                self.logger.info(f'Setting status of trial#{epoch} as TrialState.PRUNED due to small amount of shorts ({trades_s}). ')
                raise optuna.structs.TrialPruned()




            #predict

            while n_episodes < n_tests_per_eval:
                error = mlp.predict(x_test)

                errorsum += error[0]

            lll = len(error)
            error_last = np.mean(error)
            attempt +=1
            self.logger.info(f'Found a setup. mean of {lll} rewards= {-1 * error_last}$. inserting to optuna db this attempt# {attempt}')#mean reward 5.39998984336853$
            #optuna.trial.Trial
            trial.report(value=-1 * error_last
                       , step =epoch)#If step =None, the value is stored as a final value of the trial. Otherwise, it is saved as an intermediate value.

            if trial.should_prune(epoch):#Pruning Unpromising Trials
                raise optuna.structs.TrialPruned()


        return -1 * error_last# we muliply reawrd by -1 cause  Optuna interprets lower return value as better trials.

