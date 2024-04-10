import pandas as pd
import openpyxl 
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

route1 = '../data/SBS_Certified_Business_List.csv'   

route2 = '../data/sba_dataset.csv'

try:  
    df1 = pd.read_csv(route1)
    df2 = pd.read_csv(route2)
except Exception as e:
    logging.error("Error loading csv: %s", e)

#df1
df1_filtered = df1.dropna(subset=['Business_Description'])[['ID6_digit_NAICS_code', 'Business_Description']]
df1_filtered.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
df1_filtered['NAICS'] = df1_filtered['NAICS'].astype(str)

#df2
df4_filtered = df2[df2['NAICS'].apply(lambda x: ',' not in str(x) and ' ' not in str(x))][['NAICS', 'DESCRIPTION_OF_OPERATIONS']]
df4_filtered.columns = ['NAICS', 'BUSINESS_DESCRIPTION']
df4_filtered['NAICS'] = df4_filtered['NAICS'].astype(str)

#concatenation
df_concatenated = pd.concat([df1_filtered, df4_filtered], ignore_index=True)

final_df = df_concatenated.dropna(subset=['NAICS', 'BUSINESS_DESCRIPTION'])
final_df = final_df[['NAICS', 'BUSINESS_DESCRIPTION']]
final_df.dropna()

print(final_df.head())

