import logging
import os
from src.models.df_coverwallet_creation import DataProcessor
from src.models.tfidf_baseline import TfidfBaselineModel, DataPreparation
from sklearn.metrics import classification_report
from src.models.noise_conclutions import plot_class_data
import pandas as pd
from src.models.final_models import BinaryClassifier
from src.models.final_models import train_classifier_for_targets,get_df_naics,LogisticRegressionModel
import pandas as pd
import ast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def exec_tf_idf_baseline():
    DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))
    NAICS_DATA_PATH = os.path.join(DATA_DIR, "processed/coverwallet.xlsx")
    SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "processed/total_naics_synthetic.csv")

    data_prep = DataPreparation(NAICS_DATA_PATH, SYNTHETIC_DATA_PATH)
    data_prep.preprocess_data()
    data_prep.save_df(os.path.join(DATA_DIR, "processed/preprocessed_naics"))

    X_train, X_test, y_train, y_test = data_prep.train_test_split()

    tf_idf_model = TfidfBaselineModel(text_col="BUSINESS_DESCRIPTION", target="NAICS_2")
    tf_idf_model.fit(X_train, y_train)
    y_pred = tf_idf_model.predict(X_test)

    logging.info(f"{'#'*10} RESULTS FOR TF-IDF BASELINE MODEL {'#'*10}")
    print(classification_report(y_test, y_pred))
    logging.error("Error loading and executing TF-IDF model")


def main():
    ROUTE_CERTIFIED_BUSSINESS = "src/data/SBS_Certified_Business_List.csv"
    ROUTE_SBA_DATASET = "src/data/sba_dataset.csv"

    processor = DataProcessor(ROUTE_CERTIFIED_BUSSINESS, ROUTE_SBA_DATASET)
    df_final = processor.load_and_process()

    try:
        exec_tf_idf_baseline()
    except Exception as e:
        logging.error(e)
    
   plot_class_data()
   df_naics = get_df_naics()
   logistic_model = LogisticRegressionModel(df_naics, '31-33')
   logistic_model.prepare_data()
   logistic_model.train_model()
   logistic_model.evaluate_model()
   
   metrics = train_classifier_for_targets(df_naics)
   
  
if __name__ == '__main__':
    main()




