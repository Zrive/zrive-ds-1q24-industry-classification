import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, f1_score,accuracy_score,log_loss
from sklearn.preprocessing import label_binarize
from itertools import cycle
import logging
import pandas as pd
from typing import Tuple 
from torch import nn
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from nltk import download
from nltk.corpus import wordnet as wn
import random
from typing import List, Set
from torch import cuda
from transformers import DistilBertTokenizer, DistilBertModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = 'cuda' if cuda.is_available() else 'cpu'

class DataCreation:
    def __init__(self, route_synthetic: str, route_coverwallet: str):
        self.route_synthetic = route_synthetic
        self.route_coverwallet = route_coverwallet
        self.df_synthetic = self.load_and_prepare_synthetic_data()
        self.df_coverwallet_aumentation = self.load_data_coverwallet()
        self.rebalancer = NaicsRebalancer()
        self.df_naics_rebalanced = self.rebalancer.rebalance_df_naics(self.df_coverwallet_aumentation)
        self.df_coverwallet_aumented = self.concat_and_process()

    def load_and_prepare_synthetic_data(self) -> pd.DataFrame:
        df_synthetic = pd.read_parquet(self.route_synthetic)
        df_synthetic = df_synthetic[['BUSINESS_DESCRIPTION', 'NAICS_2']]
        df_synthetic.rename(columns={'NAICS_2': 'NAICS'}, inplace=True)
        logging.info(f"Synthetic Data Loaded and Prepared: \n{df_synthetic.head()}")
        return df_synthetic

    def load_data_coverwallet(self) -> pd.DataFrame:
        df_coverwallet = pd.read_excel(self.route_coverwallet)
        logging.info("Coverwallet Data Loaded")
        return df_coverwallet

    def concat_and_process(self) -> pd.DataFrame:
        df_combined = pd.concat([self.df_naics_rebalanced, self.df_synthetic], ignore_index=True)
        df_combined['NAICS'] = df_combined['NAICS'].astype(str)
        df_combined.sort_values(by='NAICS', inplace=True)
        logging.info(f"Data Combined and Sorted: Number of rows after EDA: {len(df_combined)}")
        return df_combined

    def plot_naics_counts(self):
        naics_counts = self.df_coverwallet_aumented['NAICS'].value_counts()
        logging.info(f"NAICS Counts: \n{naics_counts}")
        naics_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
        plt.xlabel('NAICS Code')
        plt.ylabel('Count of Rows')
        plt.title('Number of Rows per NAICS Code in df_coverwallet_aumented')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def print_data_head(self):
        logging.info(f"Data Head: \n{self.df_coverwallet_aumented.head()}")


class NaicsRebalancer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        download('wordnet')

    def get_synonyms(self, word: str) -> List[str]:
        synonyms: Set[str] = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                synonyms.add(synonym)
        synonyms.discard(word)
        return list(synonyms)

    def random_swap(self, sentence: str, n: int) -> str:
        words = sentence.split()
        n = min(n, len(words) // 2)
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def augment_description(self, description: str) -> str:
        words = description.split()
        if len(words) < 2:
            return description
        new_description = description
        num_swaps = random.randint(1, max(2, len(words) // 2))
        swapped_words = set()
        for _ in range(num_swaps):
            swap_word = random.choice(words)
            if swap_word in swapped_words:
                continue
            synonyms = self.get_synonyms(swap_word)
            if synonyms:
                new_word = random.choice(synonyms)
                new_description = new_description.replace(swap_word, new_word, 1)
                swapped_words.add(swap_word)
        return self.random_swap(new_description, n=num_swaps)

    def rebalance_df_naics(self, df: pd.DataFrame) -> pd.DataFrame:
        df_2_digits, dataset_train_2_digits, dataset_final_val_2_digits = self.truncate_naics_and_prepare_data(df, 'NAICS', 2)
        naics_counts = df_2_digits['NAICS'].value_counts()
        underrepresented_naics = naics_counts[naics_counts < 10].index
        overrepresented_naics = naics_counts[naics_counts > 1500].index
        new_rows = []
        for naics_code in underrepresented_naics:
            needed_rows = 10 - naics_counts[naics_code]
            original_rows = df_2_digits[df_2_digits['NAICS'] == naics_code]
            while needed_rows > 0:
                for _, row in original_rows.iterrows():
                    if needed_rows <= 0:
                        break
                    new_description = self.augment_description(row['BUSINESS_DESCRIPTION'])
                    new_rows.append({'NAICS': naics_code, 'BUSINESS_DESCRIPTION': new_description})
                    needed_rows -= 1
        for naics_code in overrepresented_naics:
            rows_to_keep = df_2_digits[df_2_digits['NAICS'] == naics_code].sample(n=1500, random_state=42)
            df_2_digits = pd.concat([df_2_digits[df_2_digits['NAICS'] != naics_code], rows_to_keep], ignore_index=True)
        new_rows_df = pd.DataFrame(new_rows)
        df_2_digits = pd.concat([df_2_digits, new_rows_df], ignore_index=True)
        return df_2_digits

    def truncate_naics_and_prepare_data(self, df: pd.DataFrame, column_name: str, num_digits: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.dropna()
        if not isinstance(num_digits, int) or num_digits <= 0:
            logging.error("Number of digits must be a positive integer")
            raise ValueError("Number of digits must be a positive integer")
        df_coverwallet_copy = df.copy()
        def truncate_code(code):
            try:
                return str(code)[:num_digits]
            except Exception as e:
                logging.exception(f"Error truncating code: {code}")
                return code
        df_coverwallet_copy[column_name] = df_coverwallet_copy[column_name].apply(truncate_code)
        df_coverwallet_copy[column_name] = df_coverwallet_copy[column_name].astype(str)
        df_coverwallet_copy_train, df_coverwallet_copy_val = train_test_split(df_coverwallet_copy, test_size=0.15, shuffle=True, random_state=42)
        return df_coverwallet_copy, df_coverwallet_copy_train, df_coverwallet_copy_val
    
class CustomDataset(Dataset):
    
    def __init__(self, dataframe, max_len):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.texts = np.array(dataframe['BUSINESS_DESCRIPTION'].astype(str))
        self.targets = np.array(dataframe['NAICS'].astype(int))
        self.max_len = max_len
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.model.config.dim, len(np.unique(self.targets)))
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :] 
        logits = self.classifier(embeddings)
        label = torch.tensor(target, dtype=torch.long)
        return {
            'embeddings': embeddings,
            'label': label,
        } 
        
class DatasetProcessor:
    def __init__(self, dataframe, max_len=128):
        self.dataframe = dataframe
        self.max_len = max_len
        self.dataset = CustomDataset(self.dataframe, self.max_len)
        self.df_embeddings = self.create_dataframe()

    def create_dataframe(self):
        data = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            embeddings = item['embeddings'].squeeze().detach().numpy()
            label = item['label'].item()
            data.append({
                'label': label,
                'embeddings': embeddings
            })
        df_embeddings = pd.DataFrame(data)
        logging.info(f"DataFrame created: \n{df_embeddings.head()}")
        return df_embeddings

    def get_dataframe(self):
        return self.df_embeddings 
    
class LogisticRegressionClassifier:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def prepare_data(self):
        df_training, df_val = train_test_split(self.dataframe, test_size=0.2, random_state=42, stratify=self.dataframe['label'])
        self.X_train = np.stack(df_training['embeddings'].apply(lambda x: np.array(x).flatten()))
        self.y_train = df_training['label'].to_numpy()
        self.X_val = np.stack(df_val['embeddings'].apply(lambda x: np.array(x).flatten()))
        self.y_val = df_val['label'].to_numpy()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_val_pred = self.model.predict(self.X_val)
        y_val_proba = self.model.predict_proba(self.X_val)
        conf_matrix = confusion_matrix(self.y_val, y_val_pred)
        report = classification_report(self.y_val, y_val_pred)
        f1 = f1_score(self.y_val, y_val_pred, average='weighted')
        logging.info("Classification Report:\n" + report)
        logging.info("F1 Score: %f", f1)
        self.plot_precision_recall_curve(y_val_pred, y_val_proba)

    def plot_precision_recall_curve(self, y_val_pred, y_val_proba):
        unique_classes = np.unique(self.y_train)
        y_val_binarized = label_binarize(self.y_val, classes=unique_classes)
        precision = dict()
        recall = dict()
        average_precision = dict()
        f1_scores = []
        colors = cycle(plt.cm.viridis(np.linspace(0, 1, len(unique_classes))))
        plt.figure(figsize=(10, 8))
        for i, color in zip(range(len(unique_classes)), colors):
            precision[i], recall[i], _ = precision_recall_curve(y_val_binarized[:, i], y_val_proba[:, i])
            average_precision[i] = average_precision_score(y_val_binarized[:, i], y_val_proba[:, i])
            f1 = (2 * precision[i] * recall[i]) / (np.nan_to_num(precision[i]) + np.nan_to_num(recall[i]))
            f1_scores.append(np.nanmax(f1))
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label='Class {0} (AP={1:0.2f})'.format(unique_classes[i], average_precision[i])) 
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_val_binarized.ravel(), y_val_proba.ravel())
        average_precision["micro"] = average_precision_score(y_val_binarized, y_val_proba, average="micro")
        plt.plot(recall["micro"], precision["micro"], color='gold', linestyle=':', linewidth=4,
                 label='Micro-average Precision-Recall (AP={0:0.2f})'.format(average_precision["micro"]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Logistic Regression)')
        plt.legend(loc="best")
        plt.show()
        f1_average = np.mean(f1_scores)
        logging.info("Average F1 Score Logistic Regression: %f", f1_average)