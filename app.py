import logging
from src.models.df_coverwallet_creation import DataProcessor
from src.models.classification_synthetic_no_cv import SummaryClassificationSyntheticNoCV
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    
    ROUTE_SYNTHETIC= 'src/data/synthetic_preprocessed.parquet' 
    ROUTE_COVERWALLET= 'src/data/coverwallet.xlsx'
    
    summary_classifier = SummaryClassificationSyntheticNoCV(ROUTE_SYNTHETIC, ROUTE_COVERWALLET)
    summary_classifier.run()
    
if __name__ == '__main__':
    main()
