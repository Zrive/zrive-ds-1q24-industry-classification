import pandas as pd
import openpyxl 
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, route_certified_business: str, route_sba_dataset: str) -> None:
        self.route_certified_business: str = route_certified_business
        self.route_sba_dataset: str = route_sba_dataset

    def load_and_process(self) -> Optional[pd.DataFrame]:
        try:
            df_certified_business: pd.DataFrame = pd.read_csv(self.route_certified_business)
            df_sba_dataset: pd.DataFrame = pd.read_csv(self.route_sba_dataset)
        except Exception as e:
            logging.error("Error loading csv: %s", e)
            return None

        df_certified_business: pd.DataFrame = self.process_certified_business(df_certified_business)
        df_sba_dataset: pd.DataFrame = self.process_sba(df_sba_dataset)

        df_naics_business: pd.DataFrame = pd.concat([df_certified_business, df_sba_dataset], ignore_index=True)
        df_naics_business_unified: pd.DataFrame = self._finalize_dataframe(df_naics_business)
        return df_naics_business_unified

    def process_certified_business(self, df: pd.DataFrame) -> pd.DataFrame:
        df_certified_business_processed: pd.DataFrame = df.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
        df_certified_business_processed.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_certified_business_processed['NAICS'] = df_certified_business_processed['NAICS'].astype(str)
        return df_certified_business_processed

    def process_sba(self, df: pd.DataFrame) -> pd.DataFrame:
        df_certified_sba_dataset: pd.DataFrame = df[df['NAICS'].apply(lambda x: ',' not in str(x) and ' ' not in str(x))][['NAICS', 'DESCRIPTION_OF_OPERATIONS']]
        df_certified_sba_dataset.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_certified_sba_dataset['NAICS'] = df_certified_sba_dataset['NAICS'].astype(str)
        return df_certified_sba_dataset

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_final_naics: pd.DataFrame = df.dropna(subset=['NAICS', 'BUSINESS_DESCRIPTION'])
        df_final_naic = df_final_naic[['NAICS', 'BUSINESS_DESCRIPTION']]
        df_final_naic.dropna()
        return df_final_naic

