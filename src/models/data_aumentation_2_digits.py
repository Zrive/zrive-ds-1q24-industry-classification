import pandas as pd
import random
from typing import List, Set
from nltk.corpus import wordnet as wn
from nltk import download
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple

class NAICSRebalancer:
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
        df_2_digits, dataset_train_2_digits, dataset_final_val_2_digits = truncate_naics_and_prepare_data(df, 'NAICS', 2)
        naics_counts = df_2_digits['NAICS'].value_counts()
        underrepresented_naics = naics_counts[naics_counts < 600].index
        overrepresented_naics = naics_counts[naics_counts > 1500].index
        new_rows = []

        for naics_code in underrepresented_naics:
            needed_rows = 600 - naics_counts[naics_code]
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
    
def truncate_naics_and_prepare_data(df: pd.DataFrame, column_name: str, num_digits: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    df=df.dropna()
    
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

    df_coverwallet_copy [column_name] = df_coverwallet_copy [column_name].apply(truncate_code)

    df_coverwallet_copy [column_name] = df_coverwallet_copy [column_name].astype(str)

    df_coverwallet_copy_train, df_coverwallet_copy_val = train_test_split(df_coverwallet_copy , test_size=0.15, shuffle=True, random_state=42)

    return df_coverwallet_copy , df_coverwallet_copy_train, df_coverwallet_copy_val
