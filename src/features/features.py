import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging

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

def load_params(params_path: str) -> int:
    try:
        no_of_features = yaml.safe_load(open(params_path,'r'))['features']['max_features']
        logger.debug('fetched load_params')
        return no_of_features
    except:
        logger.error('failed to load params from yaml file')
        raise

def load_data(train_path: str,test_path: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('fetched data')
        return train_data,test_data
    except:
        logger.error('failed to load data')

def missing_value_treatment(train_data: pd.DataFrame,test_data: pd.DataFrame) -> None:
    try:
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        logger.debug('treated missing value')
    except:
        logger.error('failed to treat missing value')

def train_test_split(train_data: pd.DataFrame,test_data: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    try:
        xtrain = train_data['content'].values
        ytrain = train_data['sentiment'].values

        xtest = test_data['content'].values
        ytest = test_data['sentiment'].values
        logger.debug('train test splited ')
        return xtrain,ytrain,xtest,ytest
    except:
        logger.error('failed in train test split')

def vectorizer(no_of_features: int,xtrain: np.ndarray,ytrain: np.ndarray,xtest: np.ndarray,ytest: np.ndarray) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        vectorizer = TfidfVectorizer(max_features=no_of_features)

        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(xtrain)

        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(xtest)

        train_df = pd.DataFrame(X_train_bow.toarray())

        train_df['label'] = ytrain

        test_df = pd.DataFrame(X_test_bow.toarray())

        test_df['label'] = ytest

        logger.debug('text got vectorized')

        return train_df,test_df
    except:
        logger.error('failed to vectorize')

def save_file(train_df: pd.DataFrame,test_df: pd.DataFrame) -> None:
    try:
        data_path = os.path.join('./data','processed')

        os.makedirs(data_path)

        train_df.to_csv(os.path.join(data_path,'train_tfidf.csv'))
        test_df.to_csv(os.path.join(data_path,'test_tfidf.csv'))
        logger.debug('files saved successfully')
    except:
        logger.error('error in saving file')

def main() -> None:
    try:
        no_of_features = load_params('params.yaml')
        train_data,test_data = load_data('./data/interim/train_processed_data.csv','./data/interim/test_processed_data.csv')
        missing_value_treatment(train_data,test_data)
        xtrain,ytrain,xtest,ytest = train_test_split(train_data,test_data)
        train_df,test_df = vectorizer(no_of_features,xtrain,ytrain,xtest,ytest)
        save_file(train_df,test_df)
        logger.debug('main file executed')
    except:
        logger.error('some error in main file')

if __name__ == '__main__':
    main()