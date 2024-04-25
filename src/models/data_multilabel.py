import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self):
        self.ROUTE_SBA = 'src/data/sba_dataset.csv'
        self.ROUTE_SBS = 'src/data/SBS_Certified_Business_List.csv'
    
    def load_data(self) -> None:
        self.df_sba = pd.read_csv(self.ROUTE_SBA)[['NAICS', 'DESCRIPTION_OF_OPERATIONS']]
        df_sbs = pd.read_csv(self.ROUTE_SBS)
        df_certified = df_sbs.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
        df_certified.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_certified['NAICS'] = df_certified['NAICS'].astype(str)
        df_certified['NAICS'] = df_certified['NAICS'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])

        df_clean_sba = self.df_sba.dropna()
        df_clean_sba['NAICS'] = df_clean_sba['NAICS'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        df_clean_sba.rename(columns={'DESCRIPTION_OF_OPERATIONS': 'BUSINESS_DESCRIPTION'}, inplace=True)
        self.df_naics = pd.concat([df_certified, df_clean_sba], ignore_index=True)

    def truncate_naics_codes(self, digits: int) -> None:
        if 'NAICS' in self.df_naics.columns:
            self.df_naics['NAICS'] = self.df_naics['NAICS'].apply(
                lambda naics_list: [str(code).strip()[:digits] for code in naics_list]
            )

    def count_unique_naics(self) -> pd.Series:
        if 'NAICS' not in self.df_naics.columns:
            logging.error("La columna 'NAICS' no está presente en el DataFrame.")
            return pd.Series()
        
        flattened_naics = self.df_naics['NAICS'].explode()
        return flattened_naics.value_counts()

    def count_distinct_naics(self) -> int:
        if 'NAICS' not in self.df_naics.columns:
            logging.error("La columna 'NAICS' no está presente en el DataFrame.")
            return 0
        
        flattened_naics = self.df_naics['NAICS'].explode()
        return flattened_naics.nunique()

    def plot_naics_distribution(self) -> None:
        expanded_naics = self.df_naics['NAICS'].explode()
        naics_counts = expanded_naics.value_counts()
        naics_counts_df = pd.DataFrame({'NAICS': naics_counts.index, 'Count': naics_counts.values})

        plt.figure(figsize=(12, 8))
        plt.bar(naics_counts_df['NAICS'], naics_counts_df['Count'], color='blue')
        plt.xlabel('NAICS Code')
        plt.ylabel('Count')
        plt.title('Distribution of NAICS Codes in df_naics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def one_hot_encode_naics(self) -> pd.DataFrame:
        expanded_naics = self.df_naics['NAICS'].explode().reset_index()
        one_hot_encoded = pd.get_dummies(expanded_naics, columns=['NAICS'])
        df_naics_encoded = one_hot_encoded.groupby('index').sum()
        df_naics_encoded.reset_index(drop=True, inplace=True)
        df_naics_final = pd.concat([self.df_naics.drop('NAICS', axis=1), df_naics_encoded], axis=1)
        df_naics_final.columns = [col.replace('NAICS_', '') if 'NAICS_' in col else col for col in df_naics_final.columns]
        return df_naics_final

    def run(self) -> None:
        self.load_data()
        self.truncate_naics_codes(2)
        unique_naics_count = self.count_unique_naics()
        distinct_naics = self.count_distinct_naics()
        logging.info(f'Total unique NAICS codes: {distinct_naics}')
        logging.info(f'\n{unique_naics_count}')
        self.plot_naics_distribution()
        df_naics_final = self.one_hot_encode_naics()
        logging.info(f'\n{df_naics_final.head()}')


