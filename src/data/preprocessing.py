import pandas as pd
import numpy as np
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_path: str,test_path: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.debug('data loaded successfully in preprocessing file')
        return train_data,test_data
    except FileNotFoundError:
        logger.error('file not found in load_data')
        raise

def missing_value_treatment(train_data: pd.DataFrame,test_data: pd.DataFrame) -> None:
    try:
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        logger.debug('treated null values')
    except Exception as e:
        print('unexcepted error occured in missing value treamtment')
        logger.error('error in missing value treatment')
        raise

def utility_download() -> None:
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logger.debug('downloaded all nltk utility req.')
    except Exception as e:
        logger.error('error in downloading untility in utility_download')
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer= WordNetLemmatizer()

        text = text.split()

        text=[lemmatizer.lemmatize(y) for y in text]

        return " " .join(text)
    except Exception as e:
        logger.error('error in lemmatization')
        raise

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logger.error('error in remove_stop_words')
        raise

def removing_numbers(text: str) -> str:
    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error('erroor in removing_numbers')
        raise

def lower_case(text: str) -> str:

    try:
        text = text.split()

        text=[y.lower() for y in text]

        return " " .join(text)
    except Exception as e:
        logger.error('error in lower_case')
        raise

def removing_punctuations(text: str) -> str:

    try:
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        # logger.debug('removed punctuations')
        return text.strip()
    
    except Exception as e:
        logger.error('error in remoing punctuations')
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # logger.debug('removed urls')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error('error in removing urls')
        raise

def remove_small_sentences(df: str) -> str:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        # logger.debug('removed all small sentences')

    except Exception as e:
        logger.error('error in remove small sentences')
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.debug('normalize text run successfully')
        return df
    except Exception as e:
        logger.error('error in normalize text')
        raise

def save_data(train_processed_data: pd.DataFrame,test_processed_data: pd.DataFrame) -> None:
    try:
        data_path = os.path.join('./data','interim')

        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed_data.csv"))
        test_processed_data.to_csv(os.path.join(data_path,"test_processed_data.csv"))
        logger.info('saved preprocessed data')
    except Exception as e:
        logger.error('error in save_data')
        raise


def main() -> None:
    try:
        train_data,test_data = load_data('./data/raw/train.csv','./data/raw/test.csv')
        missing_value_treatment(train_data,test_data)
        utility_download()
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        save_data(train_processed_data,test_processed_data)
        logger.debug('main file run successfully')
    except Exception as e:
        logger.error('error in main function')
        raise

if __name__ == '__main__':
    main()