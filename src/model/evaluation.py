import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
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

def load_model(model_path: str) -> BaseEstimator:
    try:
        clf = pickle.load(open(model_path,'rb'))
        logger.debug('model loaded successfully')
        return clf
    except FileExistsError:
        logger.error('model not found')
    except Exception as e:
        logger.error('some error in load model')
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(data_path)
        xtest = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        logger.debug('data loaded successfully')
        return xtest,y_test
    except Exception as e:
        logger.error('error occured in load_data')
        raise

def test(clf : BaseEstimator,xtest: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    try:
        y_pred = clf.predict(xtest)
        y_pred_proba = clf.predict_proba(xtest)[:, 1]
        logger.debug('model tested')
        return y_pred,y_pred_proba
    except Exception as e:
        logger.error('error occured in test')
        raise

# Calculate evaluation metrics
def evaluation(y_pred: np.ndarray,y_test: np.ndarray,y_pred_proba:np.ndarray) -> dict:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        logger.debug('evaluating model')
        return metrics_dict
    except Exception as e:
        logger.error('error occured during evaluating')
        raise

def saving_metrics(path : str,metrics_dict: dict) -> None:
    try:
        with open(path,'w') as file:
            json.dump(metrics_dict,file,indent=4)
        logger.info('metrics saved')
    except Exception as e:
        logger.error('error occured with saving the metrics .json')
        raise

def main() -> None:
    try:
        clf = load_model('models/model.pkl')
        xtest,ytest = load_data('./data/processed/test_tfidf.csv')
        ypred,ypred_proba = test(clf,xtest)
        metrics = evaluation(ypred,ytest,ypred_proba) 
        saving_metrics('metrics.json',metrics)
        logger.debug('main file executed')
    except Exception as e:
        logger.error('error occured in main')
        raise

if __name__ == '__main__':
    main()