import numpy as np
import pandas as pd
import yaml 
import os
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formater)
console_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        test_size = yaml.safe_load(open(params_path,'r'))['ingestion']['test_size']
        logger.debug('test size retrieved from yaml file')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug('data retrieved from url')
        return df
    except FileNotFoundError:
        logger.error('file not found')
        raise
    except Exception as e:
        logger.error('unknown error occured')
        raise

def preprocess_data(df : pd.DataFrame,test_size : float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        df.drop(columns=['tweet_id'],inplace=True)

        final_df = df[df['sentiment'].isin(['happiness','sadness'])]

        final_df['sentiment'].replace({'happiness':1,'sadness':0},inplace=True)

        train_data,test_data = train_test_split(final_df,test_size=test_size,shuffle=True,random_state=23)

        logger.debug('data processed')
        return train_data,test_data
    
    except Exception as e:
        logger.error('unexpected error')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug('data got saved in raw folder')
    except Exception as e:
        logger.error('error while saving the data in raw folder')
        raise


def main() -> None:
    try:
        test_size = load_params('params.yaml')
        df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        train_data,test_data = preprocess_data(df,test_size)
        save_data(train_data,test_data,'./data')
        logger.debug('error while running ingestion main function')
    except Exception as e:
        logger.error('some error occured in main function')
        raise


if __name__ == '__main__':
    main()