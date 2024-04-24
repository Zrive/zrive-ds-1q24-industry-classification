import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
import spacy
from typing import Tuple


class DataPreparation:
    def __init__(self, data_path: str, synthetic_path: str) -> None:
        self.data_path = data_path
        self.synthetic_path = synthetic_path
        self.df = self.load_data()
        self.synthetic_df = self.load_synthetic()
        self.text_col = "BUSINESS_DESCRIPTION"

    def load_data(self) -> pd.DataFrame:
        # data expected in excel format
        df = pd.read_excel(self.data_path)
        logging.info("Coverwallet data loaded")
        return df

    def load_synthetic(self) -> pd.DataFrame:
        # data expected in parquet format
        synthetic_df = pd.read_parquet(self.synthetic_path)
        synthetic_df.drop(columns=[self.text_col], inplace=True)
        synthetic_df.rename(
            columns={"naics_2": "NAICS_2", "PREPROCESSED_DESCRIPTION": self.text_col},
            inplace=True,
        )

        logging.info("Synthetic data loaded")
        return synthetic_df

    def mix_data(self) -> pd.DataFrame:
        # mix the two dataframes
        df = pd.concat([self.df, self.synthetic_df])
        return df

    def set_text_col(self, text_col: str) -> None:
        self.text_col = text_col

    def preprocess_data(self) -> None:
        # applies stemming and removes stopwords and punctuation
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

    def train_test_split(self) -> Tuple[pd.DataFrame]:
        X = self.df[self.text_col]
        y = self.df["NAICS_2"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        X_train = pd.concat([X_train, self.synthetic_df[self.text_col]])
        y_train = pd.concat([y_train, self.synthetic_df["NAICS_2"]])

        return X_train, X_test, y_train, y_test

    def get_join_synthetic(self):
        return pd.concat(self.df, self.synthetic_df)


class TfidfBaselineModel:
    def __init__(
        self, text_col: str = "BUSINESS_DESCRIPTION", target: str = "NAICS_2"
    ) -> None:
        self.c = 1
        self.kernel = "linear"
        self.model = Pipeline(
            [("tfidf", TfidfVectorizer()), ("svm", SVC(C=self.c, kernel=self.kernel))]
        )
        self.features = text_col
        self.target = target

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def set_params(self, c: int, kernel: str) -> None:
        self.c = c
        self.kernel = kernel
        self.model = Pipeline(
            [("tfidf", TfidfVectorizer()), ("svm", SVC(C=self.c, kernel=self.kernel))]
        )

    def get_params(self):
        return self.model.get_params()

    def save_model(self, path: str = "./model.joblib") -> None:
        joblib.dump(self.model, path)
