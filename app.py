import logging
import os
from src.models.df_coverwallet_creation import DataProcessor
from src.models.tfidf_baseline import TfidfBaselineModel, DataPreparation
from sklearn.metrics import classification_report

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

    if df_final is not None:
        logging.info("\n%s", df_final.head())

    try:
        exec_tf_idf_baseline()
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()
