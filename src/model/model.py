import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formater)
logger.addHandler(console_handler)

file_handler.setFormatter(formater)
logger.addHandler(file_handler)

def load_params(params_path: str):
    try:
        params = yaml.safe_load(open(params_path,'r'))['model']
        logger.debug('params loaded from yaml file')
        return params
    except FileNotFoundError:
        logger.error('file not found in load params')
        raise

def model_building(train_path: str,params) -> BaseEstimator:
    try:
        train_data = pd.read_csv(train_path)

        xtrain = train_data.iloc[:,0:-1]
        ytrain = train_data.iloc[:,-1]

        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
        clf.fit(xtrain,ytrain)
        logger.debug('model build successfully')
        return clf
    except FileNotFoundError:
        logger.error('file not found in model_building')
        raise
    except Exception as e:
        logger.error('some other error occured in model_building',e)

def save_model(model_path : str,clf : BaseEstimator) -> None:
    try:
        pickle.dump(clf,open(model_path,'wb'))
        logger.info('model saved as a binary file')
    except Exception as e:
        logger.error('error in save model ')
        raise

def main() -> None:
    try:
        params = load_params('params.yaml')
        clf = model_building('./data/processed/train_tfidf.csv',params)
        save_model('models/model.pkl',clf)
        logger.debug('main file executed')
    except:
        logger.error('some error occured in main file')

if __name__ == '__main__':
    main()