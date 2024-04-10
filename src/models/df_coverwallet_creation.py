import pandas as pd
import openpyxl 
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, route_certified_business, route_sba_dataset):
        self.route_certified_business = route_certified_business
        self.route_sba_dataset = route_sba_dataset

    def load_and_process(self):
        try:
            df_certified_business = pd.read_csv(self.route_certified_business)
            df_sba_dataset = pd.read_csv(self.route_sba_dataset)
        except Exception as e:
            logging.error("Error loading csv: %s", e)
            return None

        df_certified_business_filtered = self._process_df1(df_certified_business)
        df_sba_dataset = self._process_df2(df_sba_dataset)

        df_concatenated = pd.concat([df_certified_business_filtered, df_sba_dataset], ignore_index=True)
        final_df = self._finalize_dataframe(df_concatenated)

        return final_df

    def _process_df1(self, df):
        df_filtered = df.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
        df_filtered.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_filtered['NAICS'] = df_filtered['NAICS'].astype(str)
        return df_filtered

    def _process_df2(self, df):
        df_filtered = df[df['NAICS'].apply(lambda x: ',' not in str(x) and ' ' not in str(x))][['NAICS', 'DESCRIPTION_OF_OPERATIONS']]
        df_filtered.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_filtered['NAICS'] = df_filtered['NAICS'].astype(str)
        return df_filtered

    def _finalize_dataframe(self, df):
        final_df = df.dropna(subset=['NAICS', 'BUSINESS_DESCRIPTION'])
        final_df = final_df[['NAICS', 'BUSINESS_DESCRIPTION']]
        final_df.dropna()
        return final_df
