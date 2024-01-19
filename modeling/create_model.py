import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ipywidgets import interact
import imageio as iio
import plotly.graph_objects as go
import shap
import logging
import mlflow
from datetime import date
import pickle
import os

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from xgboost.sklearn import XGBClassifier


from data_preprocessing.sql_functions import SqlFunctions


logger = logging.getLogger(f"{os.path.basename(__file__)}")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class Modeling:

    RANDOM_STATE = 12
    FOLDS = StratifiedKFold(n_splits = 5)
    SCORER = make_scorer(f1_score, average='macro')
    DATA_NAME = 'churn_data'
    PROJECT_NAME = 'churn_detection'

    def __init__(self, start_date, end_date):

        data = SqlFunctions()
        self.train_data = data.read_data(str(start_date), str(end_date))
        self.training_date = str(datetime.now().date())


        X_train, X_test, Y_train, Y_test = train_test_split(
                                                    self.train_data.drop(['MSISDN', 'CLASS'],
                                                                          axis=1),
                                                    self.train_data['CLASS'], test_size=0.25,
                                                    random_state=self.RANDOM_STATE,
                                                    stratify=self.train_data['CLASS'])
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        

    def tuning(self, model_name, model, hyper_params):
    
        model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params,
                        scoring=self.SCORER, 
                        cv = self.FOLDS)
        
        model_cv.fit(self.X_train, self.Y_train)
    
        return model_cv.best_estimator_, model_cv.best_params_
    
    
    def get_and_print_metrics(self, model_name, Y_true, Y_pred, model):

    
        f1 = round(f1_score(Y_pred, Y_true, average='macro'), 2)
        prec = round(precision_score(Y_true, Y_pred, average='macro'), 2)
        recall = round(recall_score(Y_true, Y_pred, average='macro'), 2)
        
        metrics = {'f1_score': f1, 'precision': prec, 'recall': recall}
        
        print(model_name + ' metrics')

        print(f'Train f1_score: - {f1_score(model.predict(self.X_train), self.Y_train, average="macro")}')
        print(f'Test f1_score: - {f1}')
        print('Confusion matrix')
        print(confusion_matrix(Y_true, Y_pred))
        print('Classification report')
        print(classification_report(Y_true, Y_pred))

        return metrics
    

    def create_experiment(self, experiment_name, run_metrics, model,
                           X_test, run_params, voting=False):
    
        if not os.path.exists("modeling/models"):
            os.makedirs("modeling/models")

        if not os.path.exists("modeling/images"):
            os.makedirs("modeling/images")
                    
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            
            for param in run_params:
                mlflow.log_param(param, run_params[param])
                
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
                
            pred_proba = model.predict_proba(X_test)
            plt.hist(pred_proba[:,1])
            fig_name = f"modeling/images/{experiment_name}_proba_hist.png"
            plt.savefig(fig_name)
            plt.clf()

            mlflow.log_artifact(fig_name) 
            
            mlflow.sklearn.log_model(model, experiment_name + '_' +"model")
            
            if voting:
                print(f"Saving model {experiment_name}...")

                pickle.dump(model, open(
                    f"modeling/models/Churn_{experiment_name}" +  ".pkl", "wb"))


    def training(self):
        # self.models = { "XGB":  (XGBClassifier(random_state=self.RANDOM_STATE),
        #                         {'n_estimators':np.arange(500, 1500, 100),
        #                         'learning_rate':[0.01, 0.03, 0.05, 0.1, 0.3],
        #                         'max_depth': np.arange(5, 15)}),
        #                 "RF":  (RandomForestClassifier(random_state=self.RANDOM_STATE),
        #                        {"criterion": ["gini", "entropy"],
        #                         'n_estimators':np.arange(500, 1500, 100),
        #                         'max_depth': np.arange(5, 15),
        #                         'min_samples_split': np.arange(5, 10)}),
        #                 "CatBoost":(CatBoostClassifier(random_state=self.RANDOM_STATE,
        #                                                early_stopping_rounds=20, verbose=False),
        #                            {'iterations':np.arange(500, 1500, 100),
        #                             'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
        #                             'depth':np.arange(5, 15),
        #                             'l2_leaf_reg': np.arange(1, 3)})
        #                 }
        
        self.models = { "XGB":  (XGBClassifier(random_state=self.RANDOM_STATE),
                                {'max_depth': np.arange(5, 7)}),
                        "RF":  (RandomForestClassifier(random_state=self.RANDOM_STATE),
                               {'max_depth': np.arange(5, 7)}),
                        "CatBoost":(CatBoostClassifier(random_state=self.RANDOM_STATE,
                                                       early_stopping_rounds=20, verbose=False),
                                   {'depth':np.arange(5, 7)})
                        }
        
        
        self.models_list = []
        for model_name, param in self.models.items():
            best_estimator, best_params = self.tuning(model_name, param[0], param[1])

            model_pred = best_estimator.predict(self.X_test)

            metrics = self.get_and_print_metrics(model_name, self.Y_test, model_pred, best_estimator)
            
            self.create_experiment(experiment_name=model_name + "_" + str(self.training_date),
                                    run_metrics=metrics,
                                    model=best_estimator,
                                    X_test=self.X_test,
                                    run_params=best_params)
            
            self.models_list.append((model_name, best_estimator))
        return self.models_list


    def modeling(self):

        voting = VotingClassifier(estimators=self.training() ,voting='soft')

        voting.fit(self.X_train, self.Y_train)
        voting_pred = voting.predict(self.X_test)
        voting_model_metrics = self.get_and_print_metrics('Voting_RF_XGB_CatBoost', self.Y_test,
                                                           voting_pred, voting)
        
        voting_params = {'voting': 'soft', 'estimators': 3}

        self.create_experiment(experiment_name="Voting_RF_XGB_CatBoost_" + str(self.training_date),
                                    run_metrics=voting_model_metrics,
                                    model=voting,
                                    X_test=self.X_test,
                                    run_params=voting_params,
                                    voting=True)
        


        

        
            


