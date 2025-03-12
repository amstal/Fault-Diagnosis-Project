import os
import zipfile
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Téléchargement et extraction du dataset
def download_and_extract(url, output_folder):
    zip_path = output_folder + ".zip"
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("../")
    os.remove(zip_path)

# Paramètres
data_url = "https://nextcloud.centralesupelec.fr/s/7AR6aamBZNXcRM8/download"
data_folder = "../dataset"
if not os.path.exists(data_folder):
    download_and_extract(data_url, data_folder)

# Chargement des données à partir de CSV
def load_data_from_csv(folder):
    features, labels = [], []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                features.append(df.iloc[:, :-1].values)  # Toutes les colonnes sauf la dernière
                labels.append(df.iloc[:, -1].values)  # Dernière colonne
    return np.vstack(features), np.concatenate(labels)


X_train, y_train = load_data_from_csv("../dataset/trainingDatasets/20241016/")
X_test, y_test = load_data_from_csv("../dataset/testDatasets/20241008")

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Conversion en tensors PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Définition du Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)  # Ajout de la dimension temporelle
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définition du modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.softmax(out)

input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = len(torch.unique(y_train))
model = LSTMClassifier(input_dim, hidden_dim, output_dim)

# Définition de l'optimiseur et de la fonction de perte
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Évaluation du modèle
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.tolist())
        y_true.extend(y_batch.tolist())

# Affichage des résultats
print(classification_report(y_true, y_pred))
