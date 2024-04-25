import logging
from src.models.df_coverwallet_creation import DataProcessor

from src.models.data_augmentation import NAICSRebalancer
import pandas as pd

from src.models.classification_synthetic_no_cv import SummaryClassificationSyntheticNoCV
import matplotlib.pyplot as plt
from src.models.data_multilabel import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(): 
    
    processor = DataProcessor()
    processor.run()
    
if __name__ == '__main__':
    main()




