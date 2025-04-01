import os
import zipfile
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
def load_data_from_csv(root_folder):
    features, labels = [], []
    label_dict = {}  # Dictionnaire pour mapper les noms de dossiers à des indices numériques
    label_index = 0  # Index de départ pour les labels

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)  # Nom du dossier parent comme label

                # Si le dossier n'est pas encore dans le dictionnaire, on lui assigne un index
                if folder_name not in label_dict:
                    label_dict[folder_name] = label_index
                    label_index += 1

                # Chargement des données
                df = pd.read_csv(file_path)
                
                # Colonnes de commande (0-2) et réponse (3-5)
                commande = df.iloc[:, :3].values
                reponse = df.iloc[:, 3:6].values
                
                # Calcul de la différence entre commande et réponse
                diff = commande - reponse
                
                # Concaténation des colonnes de commande et de la différence
                combined_features = np.hstack((commande, diff))
                
                features.append(combined_features)  # Utiliser la commande et la différence comme features
                labels.append(np.full(len(df), label_dict[folder_name]))  # Tous les labels = dossier

    return np.vstack(features), np.concatenate(labels), label_dict

# Chargement des datasets
X_train, y_train, train_label_dict = load_data_from_csv("../dataset/trainingDatasets/")
X_test, y_test, test_label_dict = load_data_from_csv("../dataset/testDatasets/")

# Affichage des labels trouvés
print("Labels trouvés pour l'entraînement:", train_label_dict)
print("Labels trouvés pour le test:", test_label_dict)

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
        self.X = X.unsqueeze(1)  # Ajout de la dimension temporelle (préférable d'ajouter avant la dimension 'features')
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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()

        # Convolution pour extraire les patterns spatiaux
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM pour capturer les dépendances temporelles
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)  # Cette ligne n'est plus nécessaire car la dimension temporelle est déjà ajoutée dans Dataset

        # Appliquer la convolution
        x = self.conv1(x).relu()  # Conv1d attend un format [batch_size, channels, seq_length]
        x = self.conv2(x).relu()
        x = self.pool(x)  # Réduction de la dimension

        x = x.permute(0, 2, 1)  # Reshape pour LSTM (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Prendre le dernier état caché

        x = self.fc1(last_out)
        x = self.batch_norm(x).relu()
        x = self.dropout(x)
        return self.fc2(x)  # Pas de Softmax (géré par CrossEntropyLoss)


# Initialisation du modèle
input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = len(torch.unique(y_train))
model = LSTMClassifier(input_dim, hidden_dim, output_dim)

# Définition de l'optimiseur et de la fonction de perte
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Évite overfitting sur un seul label
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Réduction du LR si stagnation
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Entraînement du modèle
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Calcul de l'exactitude pour l'entraînement
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

    # Calculer l'exactitude sur le jeu de test et ajuster le taux d'apprentissage
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    scheduler.step(accuracy)

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

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.show()
