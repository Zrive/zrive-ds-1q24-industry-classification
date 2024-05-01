import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
import spacy
from typing import Tuple


class DataPreparation:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.df = self.load_data()
        self.synthetic_df = self.load_synthetic()
        self.text_col = "BUSINESS_DESCRIPTION"

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        logging.info("Coverwallet data loaded")
        return df

    def set_text_col(self, text_col: str) -> None:
        self.text_col = text_col

    def preprocess_data(self) -> None:
        nlp = spacy.load("en_core_web_sm")

        def preprocess(text: str) -> str:
            doc = nlp(text)
            word_list = [
                word.lemma_ for word in doc if not (word.is_stop or word.is_punct)
            ]
            return " ".join(word_list)

        logging.info("Start text preprocessing")
        self.df[self.text_col] = self.df[self.text_col].apply(preprocess)
        logging.info("Text preprocessing ended")

    def save_df(self, path="./model.parquet") -> None:
        self.df.to_parquet(path)
        logging.info(f"Dataframe saved at {path}")

    def train_test_split(self, code: str) -> Tuple[pd.DataFrame]:
        X = self.df[self.text_col]
        y = self.df[code]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        return X_train, X_test, y_train, y_test


class TfidfBinaryModel:
    def __init__(self, target: str, text_col: str = "BUSINESS_DESCRIPTION") -> None:
        self.model = Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]
        )
        self.features = text_col
        self.target = target

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def save_model(self, path: str = "./model.joblib") -> None:
        joblib.dump(self.model, path)
