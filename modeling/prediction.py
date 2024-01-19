import os
import pickle
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from data_preprocessing.sql_functions import SqlFunctions

from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score


class Prediction:

    def __init__(self, start_date:datetime, end_date:datetime)->None:
        
        # Initialize the loaded_model attribute by calling load_latest_model
        
        self.connection = SqlFunctions()
        self.end_date = end_date
        self.prediction_data = self.connection.read_data(str(start_date), str(end_date),
                                                     train=False)
        self.loaded_model = self.load_latest_model()
        #TODO -> self.training_date = str(datetime.now().date())


    def load_latest_model(self)-> bool:


        models_folder = "modeling/models/"
        
        # List all files in the models folder
        model_files = [f for f in os.listdir(models_folder) if f.endswith(".pkl")]

        if not model_files:
            print("No model files found.")
            return None

        try:
            # Find the latest model file based on the modification time
            latest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(models_folder, f)))

            # Construct the full path to the latest model file
            file_path = os.path.join(models_folder, latest_model_file)

            # Load the model
            with open(file_path, 'rb') as file:
                loaded_model = pickle.load(file)

            print("Successfully loaded the latest model.")
            return loaded_model

        except Exception as e:
            print(f"Failed to load the latest model. Error: {e}")
            return None
        
    
    def predict(self):
        
        if self.loaded_model is not None:

            insert_cols = ['MSISDN', 'CURRENT_TP', 'MonthlyFee']
            project_name = 'churn'

            #TODO delete 
            insert_cols = ['MSISDN']
            insert_data = self.prediction_data[insert_cols]
            insert_data['CURRENT_TP'] = 'CURRENT_TP'
            insert_data['MonthlyFee'] = 123
            #TODO delete

            # insert_data = self.prediction_data[insert_cols]
            insert_data["PROJECT_NAME"] = project_name
            insert_data["MODEL_NAME"] = self.loaded_model.__class__.__name__
            insert_data['DATE_ID'] =  self.end_date + relativedelta(months=1)
            insert_data['ACTION_GROUP'] = 'target'

            pred_proba = self.loaded_model.predict_proba(
                                        self.prediction_data.drop(insert_cols, axis=1)
                                        )[:, 1].round(3)
            
            insert_data["PROBA"] = pred_proba
            insert_data["PROBA_GROUP"] = pred_proba.round(1)

            self.connection.insert_data(insert_data)

            # return insert_data
        else:
            print("No model loaded. Cannot make predictions.")
            return None



