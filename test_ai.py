import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# from csv_to_mat_converter import process_directory

# Creating a test neural network model :

class TestNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer1 = nn.LSTM(hidden_size = 100, input_size = input_size, num_layers = 1, batch_first = True)
        self.layer2 = nn.Dropout(0.1)
        self.layer3 = nn.LSTM(hidden_size = 100, input_size = input_size, num_layers = 1, batch_first = True)
        self.layer4 = nn.Sequential(nn.Linear(input_size, num_classes), nn.BatchNorm1d(num_classes), nn.ReLU())
        self.layer5 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))   
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Initialize the model :

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # If GPU is cuda capable use it
model = TestNN(input_size = 100, num_classes = 9).to(device)

# Choosing the loss function : 

loss_function = nn.CrossEntropyLoss() # Remark : no loss_function is given in the thesis
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)




"""A partir de ce moment c'est la machine qui prend le dessus sur l'homme"""


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
                df = pd.read_csv(file_path, nrows=20)
                features.append(df.iloc[:, :-1].values)  # Toutes les colonnes sauf la dernière
                labels.append(np.full(len(df), label_dict[folder_name]))  # Tous les labels = dossier

    return np.vstack(features), np.concatenate(labels), label_dict

# Chargement des datasets
X_train, y_train, train_label_dict = load_data_from_csv("./dataset/trainingDatasets/")
X_test, y_test, test_label_dict = load_data_from_csv("./dataset/testDatasets/")

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

# Réduction du LR si stagnation
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)


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
        loss = loss_function(outputs, y_batch)
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