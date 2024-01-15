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
        Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = Session()

    def from_sql(self, start_date=None, end_date=None) -> pd.DataFrame:
        """  
        Returns
        -------
        data : pd.DataFrame : Queried data in dataframe

        """

        with open('data_preprocessing/sql_query_for_training.sql', 'r') as file:
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


#!tests    

test = SqlFunctions()
df = test.from_sql() 
print(df)


# df['PROJECT_NAME'].replace('Churn', 'bzbz', inplace=True)
# df['CURRENT_TP'].replace('New Youth Z', 'zzz', inplace=True)
# test = DataManipulation()
# test.to_sql_update(df, ['CURRENT_TP', 'PROJECT_NAME'], 'ml_inference', 'public')

# delete_conditions = {'CURRENT_TP': 'zzz', 'PROJECT_NAME':'bzbz'}
# test = DataManipulation()
# test.delete_sql_rows('ml_inference', delete_conditions, 'public')
            
# test = DataManipulation()


# Read the SQL query from the text file

