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

        jointNAICSBusinessDatasetDF: pd.DataFrame = pd.concat([df_certified_business, df_sba_dataset], ignore_index=True)
        unifiedNAICSDescriptionsDF: pd.DataFrame = self._finalize_dataframe(jointNAICSBusinessDatasetDF)
        return unifiedNAICSDescriptionsDF

    def process_certified_business(self, df: pd.DataFrame) -> pd.DataFrame:
        filteredNAICSBusinessDF_certified_business: pd.DataFrame = df.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
        filteredNAICSBusinessDF_certified_business.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        filteredNAICSBusinessDF_certified_business['NAICS'] = filteredNAICSBusinessDF_certified_business['NAICS'].astype(str)
        return filteredNAICSBusinessDF_certified_business

    def process_sba(self, df: pd.DataFrame) -> pd.DataFrame:
        filteredNAICSBusinessDF_sba_dataset: pd.DataFrame = df[df['NAICS'].apply(lambda x: ',' not in str(x) and ' ' not in str(x))][['NAICS', 'DESCRIPTION_OF_OPERATIONS']]
        filteredNAICSBusinessDF_sba_dataset.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        filteredNAICSBusinessDF_sba_dataset['NAICS'] = filteredNAICSBusinessDF_sba_dataset['NAICS'].astype(str)
        return filteredNAICSBusinessDF_sba_dataset

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        finalMergedNAICSDataFrame: pd.DataFrame = df.dropna(subset=['NAICS', 'BUSINESS_DESCRIPTION'])
        finalMergedNAICSDataFrame = finalMergedNAICSDataFrame[['NAICS', 'BUSINESS_DESCRIPTION']]
        finalMergedNAICSDataFrame.dropna()
        return finalMergedNAICSDataFrame

