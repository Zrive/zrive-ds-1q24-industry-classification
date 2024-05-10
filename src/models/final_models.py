import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

def get_df_naics():
    
    ROUTE_BINARIZED_NAICS = 'src/data/BUENO_POR_FIN_2.csv'

   
    df_naics = pd.read_csv(ROUTE_BINARIZED_NAICS)

    df_naics['31-33'] = df_naics[['31', '32', '33']].max(axis=1)
    df_naics['44-45'] = df_naics[['44', '45']].max(axis=1)
    df_naics['48-49'] = df_naics[['48', '49']].max(axis=1)
    df_naics['embeddings'] = df_naics['embeddings'].apply(ast.literal_eval)
    df_naics.drop(['31', '32', '33', '44', '45', '48', '49'], axis=1, inplace=True)
    #df_naics['embeddings'] = df_naics['embeddings'].apply(ast.literal_eval)
    
    class_columns = df_naics.columns[2:]
    counts = df_naics[class_columns].apply(pd.value_counts).loc[1].sort_values()

    middle_class = counts.index[len(counts) // 2]  
    results = {col: {'unique_ones': 0, 'non_unique_ones': 0} for col in class_columns}

    for index, row in df_naics.iterrows():
     
        total_ones = row[class_columns].sum()
        for col in class_columns:
            if row[col] == 1:
                if total_ones == 1:
                    results[col]['unique_ones'] += 1
                else:
                    results[col]['non_unique_ones'] += 1

    #df_naics['embeddings'] = df_naics['embeddings'].apply(ast.literal_eval)
    return df_naics

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x



class LogisticRegressionModel:
    def __init__(self, data_frame, target_column):
        self.df = data_frame
        self.target_column = target_column
        self.model = LogisticRegression(class_weight='balanced', C=0.25, max_iter=1000)
    
    def prepare_data(self):
        X = np.array(self.df['embeddings'].tolist())
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        
        f1 = f1_score(self.y_test, y_pred)
        print(f'F1 Score: {f1}')
        
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print('Matriz de Confusión:')
        print(conf_matrix)
        
        fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(self.y_test, y_scores)
        auc_score = auc(recall, precision)
        print(f'AUC de la Curva de Precisión-Recall: {auc_score:.2f}')
        print(f'ROC: {roc_auc:.2f}')
        
        self.plot_precision_recall(recall, precision, auc_score)
        self.plot_roc_curve(fpr, tpr, roc_auc)
    
    def plot_precision_recall(self, recall, precision, auc_score):
        plt.figure()
        plt.plot(recall, precision, marker='.', label=f'AUC = {auc_score:.2f}')
        plt.title('Curva de Precisión-Recall')
        plt.xlabel('Recall (Sensibilidad)')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig('precision_recall.jpg')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc.jpg')
        plt.show()





def train_binary_classifier(X_train, y_train, X_val, y_val, input_size, learning_rate=0.001, epochs=20):
    model = BinaryClassifier(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=10, shuffle=False)

   
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'f1_score': [],
        'conf_matrix': [],
        'auc_pr': [],
        'roc_auc': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = outputs.squeeze().round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        metrics['train_loss'].append(total_loss / len(train_loader))
        metrics['train_acc'].append(correct / total)

       
        model.eval()
        total_loss, correct, total = 0, 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                predicted = outputs.squeeze().round()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                val_preds.extend(predicted.tolist())
                val_labels.extend(labels.tolist())

        metrics['val_loss'].append(total_loss / len(val_loader))
        metrics['val_acc'].append(correct / total)

        
        cm = confusion_matrix(val_labels, val_preds)
        fscore = f1_score(val_labels, val_preds)
        precision, recall, _ = precision_recall_curve(val_labels, val_preds)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(val_labels, val_preds)

        metrics['f1_score'].append(fscore)
        metrics['conf_matrix'].append(cm)
        metrics['auc_pr'].append(pr_auc)
        metrics['roc_auc'].append(roc_auc)

    return metrics
def train_classifier_for_targets(df_naics):
    
    classification_targets = [col for col in df_naics.columns if col != 'embeddings']
    
    all_metrics = {}
    
    for target in classification_targets:
        print(f"Training for {target}")
        X = torch.tensor(np.stack(df_naics['embeddings'].values), dtype=torch.float)
        y = torch.tensor(df_naics[target].values, dtype=torch.float)

        unique_targets = y.unique()
        print(f"Unique values in {target}: {unique_targets}")

        if not all((unique_targets == 0) | (unique_targets == 1)):
            print(f"Error: Non-binary values found in {target}")
            continue  

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics = train_binary_classifier(X_train, y_train, X_val, y_val, X_train.shape[1])

        # Output final epoch metrics for review
        print(f'Final Validation Metrics for {target}:')
        print(f"Validation Accuracy: {metrics['val_acc'][-1]:.4f}")
        print(f"F1 Score: {metrics['f1_score'][-1]:.4f}")
        print(f"Confusion Matrix:\n{metrics['conf_matrix'][-1]}")
        print(f"AUC Precision-Recall: {metrics['auc_pr'][-1]:.4f}")
        print(f"ROC AUC: {metrics['roc_auc'][-1]:.4f}")

        all_metrics[target] = metrics
    
    return all_metrics