import pandas as pd
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self):
        self.ROUTE_SBA = '../src/data/sba_dataset.csv'
        self.ROUTE_SBS = '../src/data/SBS_Certified_Business_List.csv'
        self.df_naics = None

    def load_data(self):
        df_sba = pd.read_csv(self.ROUTE_SBA)
        df_sba = df_sba[['NAICS', 'DESCRIPTION_OF_OPERATIONS']]

        df_sbs = pd.read_csv(self.ROUTE_SBS)
        df_certified_business_processed = df_sbs.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
        df_certified_business_processed.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
        df_certified_business_processed['NAICS'] = df_certified_business_processed['NAICS'].astype(str)
        df_certified_business_processed['NAICS'] = df_certified_business_processed['NAICS'].apply(lambda x: x.split(',') if isinstance(x, str) else [x])

        df_clean_sba = df_sba.dropna()
        df_clean_sba['NAICS'] = df_clean_sba['NAICS'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        df_clean_sba.rename(columns={'DESCRIPTION_OF_OPERATIONS': 'BUSINESS_DESCRIPTION'}, inplace=True)
        self.df_naics = pd.concat([df_certified_business_processed, df_clean_sba], ignore_index=True)

    def truncate_naics_codes(self, digits: int):
        if 'NAICS' in self.df_naics.columns:
            self.df_naics['NAICS'] = self.df_naics['NAICS'].apply(
                lambda naics_list: [str(code).strip()[:digits] for code in naics_list]
            )

    def count_unique_naics(self):
        if 'NAICS' not in self.df_naics.columns:
            logging.info("La columna 'NAICS' no está presente en el DataFrame.")
            return
        
        flattened_naics = self.df_naics['NAICS'].explode()
        naics_counts = flattened_naics.value_counts()
        return naics_counts

    def plot_naics_distribution(self):
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

    def run_all(self):
        self.load_data()
        self.truncate_naics_codes(2)
        unique_naics_count = self.count_unique_naics()
        logging.info(f'\n{unique_naics_count.head()}')
        self.plot_naics_distribution()