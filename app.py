import logging
from src.models.df_coverwallet_creation import DataProcessor
from src.models.data_aumentation_2_digits import NAICSRebalancer
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    '''
    ROUTE_CERTIFIED_BUSSINESS = 'src/data/SBS_Certified_Business_List.csv'   
    ROUTE_SBA_DATASET = 'src/data/sba_dataset.csv'

    processor = DataProcessor(ROUTE_CERTIFIED_BUSSINESS, ROUTE_SBA_DATASET)
    df_final = processor.load_and_process()

    if df_final is not None:
        logging.info("\n%s", df_final.head())
    '''    
    
    ROUTE_COVERWALLET= 'src/data/coverwallet.xlsx'
    
    df_coverwallet_aumentation = pd.read_excel(ROUTE_COVERWALLET)
    rebalancer = NAICSRebalancer()
    df_naics_rebalanced = rebalancer.rebalance_df_naics(df_coverwallet_aumentation)
    logging.info(f"Number of rows before EDA: {len(df_coverwallet_aumentation)}")
    logging.info(f"Number of rows after EDA: {len(df_naics_rebalanced)}")

    if df_naics_rebalanced is not None: 
        logging.info("\n%s", df_naics_rebalanced.head())   
        

if __name__ == '__main__':
    main()




