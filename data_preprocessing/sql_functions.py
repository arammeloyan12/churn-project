import pandas as pd
import numpy as np
import logging
import os
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import  create_engine, text, MetaData, Table, update, and_, delete
from config import SQLConfig


logger = logging.getLogger(f"{os.path.basename(__file__)}")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class SqlFunctions:
    """ Consists of SQL functions """
    
    def __init__(self) -> None:
        
        self.config = SQLConfig()
        self.engine = create_engine( url=self.config.database_url_psycopg, echo=True)
        # Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        # self.session = Session()

    def read_data(self, start_date, end_date, train=True) -> pd.DataFrame:
        """  
        Returns
        -------
        data : pd.DataFrame : Queried data in dataframe

        """
        
        if train:
            with open('data_preprocessing/sql_query_for_training.sql', 'r') as file:
                loaded_sql_query = file.read()
        else:
            with open('data_preprocessing/sql_query_for_prediction.sql', 'r') as file:
                loaded_sql_query = file.read()

        with self.engine.connect() as connection:
            result = connection.execute(text(loaded_sql_query),
                                        start_date=start_date,
                                        end_date=end_date)
            results = result.fetchall()
            columns = result.keys()

        data = pd.DataFrame(results, columns=columns)
        
        logger.info(f"Data's shape equal {data.shape}")

        return data


    def insert_data(self, data):
        data.to_sql('ml_inference', self.engine, index=False, if_exists='replace')
        #TODO check if_exists




#!tests    
# from datetime import date
# from dateutil.relativedelta import relativedelta

# start_date = date(2023, 1, 1)

# # Add one month to the start_date
# new_date = start_date + relativedelta(months=1)
# print(new_date)

# from datetime import date
# start_date = date(2023, 1, 1)
# end_date = date(2023, 3, 1)
# test = SqlFunctions()
# df = test.read_data(str(start_date), str(end_date), train=False) 
# print(df)
# print('OK')

# df['PROJECT_NAME'].replace('Churn', 'bzbz', inplace=True)
# df['CURRENT_TP'].replace('New Youth Z', 'zzz', inplace=True)
# test = DataManipulation()
# test.to_sql_update(df, ['CURRENT_TP', 'PROJECT_NAME'], 'ml_inference', 'public')

# delete_conditions = {'CURRENT_TP': 'zzz', 'PROJECT_NAME':'bzbz'}
# test = DataManipulation()
# test.delete_sql_rows('ml_inference', delete_conditions, 'public')
            
# test = DataManipulation()


# Read the SQL query from the text file

