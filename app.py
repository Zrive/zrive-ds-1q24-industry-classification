import logging
from src.models.df_coverwallet_creation import DataProcessor
from src.models.classification_synthetic_no_cv import DataCreation,NaicsRebalancer

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
    ROUTE_SYNTHETIC= 'src/data/synthetic_preprocessed.parquet' 
    ROUTE_COVERWALLET= 'src/data/coverwallet.xlsx'
    data_creator = DataCreation(ROUTE_SYNTHETIC, ROUTE_COVERWALLET)

    result_df = data_creator.process_data()

    print("Resultado del procesamiento de datos:")
    print(result_df.head())
if __name__ == '__main__':
    main()
