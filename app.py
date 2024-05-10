from src.models.final_models import BinaryClassifier
from src.models.final_models import train_classifier_for_targets,get_df_naics,LogisticRegressionModel
import pandas as pd
import ast
def main(): 
   
   df_naics = get_df_naics()
   logistic_model = LogisticRegressionModel(df_naics, '31-33')
   logistic_model.prepare_data()
   logistic_model.train_model()
   logistic_model.evaluate_model()
   
   metrics = train_classifier_for_targets(df_naics)
   
  
if __name__ == '__main__':
    main()




