import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gc

print("Importation des bibliothèques réussie")

# Configuration pour optimiser l'utilisation de la mémoire MPS
# Utiliser une valeur valide pour le watermark ratio (entre 0.0 et 1.0)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Désactive la limite supérieure
# Définir également le low watermark si nécessaire
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

# Configuration du périphérique (CPU/GPU)
# Ajouter une vérification pour s'assurer que CUDA est correctement initialisé
if torch.cuda.is_available():
    try:
        # Test de création d'un petit tensor sur CUDA pour vérifier la configuration
        test_tensor = torch.ones(1).to('cuda')
        device = torch.device('cuda')
        print("CUDA correctement configuré et fonctionnel")
    except RuntimeError as e:
        print(f"CUDA disponible mais erreur lors de l'initialisation: {e}")
        print("Utilisation du CPU à la place")
        device = torch.device('cpu')
else:
    # Vérifier si MPS est disponible (pour MacOS avec Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS disponible et activé pour accélération sur Apple Silicon")
    else:
        device = torch.device('cpu')
        print("Ni CUDA ni MPS disponible, utilisation du CPU")

print(f"Utilisation du périphérique: {device}")

# Paramètre pour l'augmentation des données
DATA_AUGMENTATION_ENABLED = False  # Activer/désactiver l'augmentation de données
NOISE_LEVEL = 0.001  # 0.01% de bruit gaussien
NUM_AUGMENTATIONS = 0  # Nombre de versions augmentées à créer pour chaque échantillon

# Fonction pour augmenter les données avec un bruit gaussien
def apply_data_augmentation(X_train_consigne, X_train_reponse, y_train, noise_level=NOISE_LEVEL, num_augmentations=NUM_AUGMENTATIONS):
    """
    Fonction simplifiée qui retourne les données originales sans augmentation
    
    Args:
        X_train_consigne, X_train_reponse: Tenseurs des données d'entrée
        y_train: Tenseur des étiquettes
        noise_level: Non utilisé (gardé pour compatibilité)
        num_augmentations: Non utilisé (gardé pour compatibilité)
    
    Returns:
        X_train_consigne, X_train_reponse, y_train: Données originales sans augmentation
    """
    print("Data augmentation désactivée, retour des données originales")
    
    # Convertir en NumPy si nécessaire pour garantir la compatibilité
    if isinstance(X_train_consigne, torch.Tensor):
        X_train_consigne_np = X_train_consigne.cpu().numpy()
        X_train_reponse_np = X_train_reponse.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
    else:
        X_train_consigne_np = X_train_consigne
        X_train_reponse_np = X_train_reponse
        y_train_np = y_train
    
    print(f"Données d'entraînement: {len(y_train_np)} échantillons")
    
    return X_train_consigne_np, X_train_reponse_np, y_train_np

# Liste des catégories de défauts
categories = [
    'Healthy',
    'Motor_1_Stuck',
    'Motor_1_Steady_state_error',
    'Motor_2_Stuck',
    'Motor_2_Steady_state_error',
    'Motor_3_Stuck',
    'Motor_3_Steady_state_error',
    'Motor_4_Stuck',
    'Motor_4_Steady_state_error'
]

# Chemins des dossiers pour l'entraînement, la validation et le test
TRAIN_FOLDER_1 = 'dataset/trainingDatasets/20241016'
TRAIN_FOLDER_2 = 'dataset/trainingDatasets/20241017'
TEST_FOLDER = 'dataset/testDatasets/20241008'
TEST_FOLDER_2 = 'dataset/testDatasets/20241016'  # Ajout explicite du deuxième dossier de test
VALIDATION_SPLIT_RATIO = 0.2  # 20% des données pour validation

# Définir une fonction pour convertir les .mat en tensors PyTorch (comme dans thesis_ai.py)
def transfer_tensor(mat, X_name, Y_name, mean=None, std=None):
    # Extraire les tableaux imbriqués
    data_X = mat[X_name][0]  # Extraire le tableau d'objets imbriqués pour X
    data_Y = mat[Y_name][0]
    
    # Extraire X dans un tensor
    data_X_combined = np.array([data_X[i] for i in range(len(data_X))])
    data_tensor_X = torch.tensor(data_X_combined, dtype=torch.float32).to(device)  # Déplacer vers GPU/MPS
    
    # Créer un nouveau tensor pour stocker les données traitées
    data_tensor_X_with_residual = data_tensor_X.clone()  # Cloner pour préserver les données originales
    
    # Calculer les résidus et remplacer les trois dernières features par ces résidus
    residual = data_tensor_X[:, :, :3] - data_tensor_X[:, :, 3:6]  # Calculer les résidus
    data_tensor_X_with_residual[:, :, 3:6] = residual  # Remplacer les trois dernières features par les résidus
    
    data_tensor_X = data_tensor_X_with_residual
    
    # Normaliser les features
    if mean is None or std is None:
        # Si la moyenne et l'écart-type ne sont pas fournis, les calculer
        mean = data_tensor_X.mean(dim=(0, 1), keepdim=True)
        std = data_tensor_X.std(dim=(0, 1), keepdim=True)
    
    data_tensor_X = (data_tensor_X - mean) / std  # Normaliser les features
    
    # Extraire Y dans un tensor
    data_Y_combined = np.array([data_Y[i] for i in range(len(data_Y))])
    data_Y_combined = data_Y_combined.flatten()
    
    # Créer un dictionnaire associant chaque catégorie à l'index correspondant
    category_to_index = {category: index for index, category in enumerate(categories)}
    
    data_Y_numeric = np.array([category_to_index[category] for category in data_Y_combined])
    
    # Convertir le tableau NumPy en tensor PyTorch
    data_tensor_Y = torch.tensor(data_Y_numeric, dtype=torch.int64).to(device)  # Déplacer vers GPU/MPS
    
    return data_tensor_X, data_tensor_Y, mean, std  # Renvoyer moyenne et écart-type

# Fonction pour charger les données à partir des fichiers .mat (comme dans thesis_ai.py)
def load_data_from_mat():
    print("Chargement des données à partir des fichiers .mat...")
    
    # Charger les fichiers .mat en tant que dictionnaires
    try:
        mat_train = scipy.io.loadmat('models/my_dataset_train.mat')
        mat_test = scipy.io.loadmat('models/my_dataset_test.mat')
        
        # Convertir les données d'entraînement et de test en tensors
        data_tensor_X_train, data_tensor_Y_train, simulation_mean, simulation_std = transfer_tensor(
            mat_train, 'X_array', 'y_array')
        data_tensor_X_test, data_tensor_Y_test, _, _ = transfer_tensor(
            mat_test, 'X_test_array', 'y_test_array', mean=simulation_mean, std=simulation_std)
        
        print("Données chargées avec succès à partir des fichiers .mat")
        print(f"Forme des données d'entraînement X: {data_tensor_X_train.shape}")
        print(f"Forme des données d'entraînement Y: {data_tensor_Y_train.shape}")
        print(f"Forme des données de test X: {data_tensor_X_test.shape}")
        print(f"Forme des données de test Y: {data_tensor_Y_test.shape}")
        
        # Extraire les consignes et les réponses pour correspondre à notre modèle
        X_train_consigne = data_tensor_X_train[:, :, :3]
        X_train_reponse = data_tensor_X_train[:, :, 3:6] 
        X_test_consigne = data_tensor_X_test[:, :, :3]
        X_test_reponse = data_tensor_X_test[:, :, 3:6]
        
        # Créer un dictionnaire d'étiquettes pour correspondre à l'ancien format
        train_label_dict = {category: idx for idx, category in enumerate(categories)}
        test_label_dict = train_label_dict.copy()
        
        return (X_train_consigne, X_train_reponse), data_tensor_Y_train, train_label_dict, \
               (X_test_consigne, X_test_reponse), data_tensor_Y_test, test_label_dict
        
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers .mat: {e}")
        print("Utilisation de la méthode de chargement CSV alternative...")
        return None

import pickle
import os

def load_data_from_csv_cached(root_folder, cache_file=None):
    """Version optimisée avec mise en cache du chargement des données CSV"""
    if cache_file is None:
        cache_file = f"cache_{os.path.basename(root_folder)}.pkl"
        
    # Vérifier si les données mises en cache existent
    if os.path.exists(cache_file):
        print(f"Chargement des données depuis le cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement du cache: {e}")
            print("Rechargement à partir des fichiers CSV...")
    
    # Si pas de cache ou erreur, on charge depuis les fichiers CSV
    print(f"Chargement des données depuis {root_folder}...")
    
    # Utiliser la fonction de chargement originale
    data = load_data_from_csv(root_folder)
    
    # Enregistrer le résultat dans le cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Données mises en cache dans {cache_file}")
    except Exception as e:
        print(f"Erreur lors de la création du cache: {e}")
    
    return data

# Ancienne fonction de chargement des données CSV pour compatibilité
def load_data_from_csv(root_folder):
    features_consigne, features_reponse, labels = [], [], []
    label_dict = {}
    label_index = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)

                if folder_name not in label_dict:
                    label_dict[folder_name] = label_index
                    label_index += 1

                # Charger le CSV sans supposer d'en-têtes
                df = pd.read_csv(file_path, header=None)
                
                # Les 3 premières colonnes sont pour les consignes (indices 0, 1, 2)
                # Les 3 colonnes suivantes sont pour les réponses (indices 3, 4, 5)
                
                # Vérifier que nous avons au moins 6 colonnes (3 consignes + 3 réponses)
                if df.shape[1] < 6:
                    print(f"Attention: Le fichier {file_path} n'a que {df.shape[1]} colonnes. Au moins 6 sont attendues.")
                    # Ajouter des colonnes vides si nécessaire
                    for i in range(df.shape[1], 7):
                        df[i] = 0
                
                # Restructurer les données pour qu'elles ressemblent au format 3D
                # Les données doivent avoir la forme [batch, time_steps, features]
                sequence_length = 1000  # Comme dans thesis_ai.py
                
                # Si les données ont plus de 1000 pas de temps, nous les divisons en séquences
                if len(df) > sequence_length:
                    num_sequences = len(df) // sequence_length
                    for i in range(num_sequences):
                        start_idx = i * sequence_length
                        end_idx = start_idx + sequence_length
                        
                        # Extraire les séquences et les ajouter à nos listes
                        consigne_seq = df.iloc[start_idx:end_idx, 0:3].values
                        reponse_seq = df.iloc[start_idx:end_idx, 3:6].values
                        features_consigne.append(consigne_seq)
                        features_reponse.append(reponse_seq)
                        labels.append(label_dict[folder_name])
                else:
                    # Padding si la séquence est trop courte
                    consigne_data = df.iloc[:, 0:3].values
                    reponse_data = df.iloc[:, 3:6].values
                    
                    # Padding pour atteindre sequence_length
                    pad_length = sequence_length - len(consigne_data)
                    if pad_length > 0:
                        consigne_pad = np.zeros((pad_length, 3))
                        reponse_pad = np.zeros((pad_length, 3))
                        consigne_data = np.vstack([consigne_data, consigne_pad])
                        reponse_data = np.vstack([reponse_data, reponse_pad])
                    
                    features_consigne.append(consigne_data)
                    features_reponse.append(reponse_data)
                    labels.append(label_dict[folder_name])

    # Convertir en tableaux NumPy
    features_consigne = np.array(features_consigne)  # [num_sequences, sequence_length, 3]
    features_reponse = np.array(features_reponse)    # [num_sequences, sequence_length, 3]
    labels = np.array(labels)                      # [num_sequences]

    return (features_consigne, features_reponse), labels, label_dict

# Essayer d'abord de charger les données à partir des fichiers .mat
# mat_data = load_data_from_mat()
mat_data = None  # Force l'utilisation des fichiers CSV au lieu des .mat

if (mat_data is not None):
    # Utiliser les données des fichiers .mat
    X_train, y_train, train_label_dict, X_test, y_test, test_label_dict = mat_data
    # Nous n'avons pas explicitement de données de validation, nous les créerons à partir des données d'entraînement
    X_val, y_val, val_label_dict = None, None, train_label_dict
else:
    # Remplacer ces lignes dans votre script
    print("Chargement des données d'entraînement depuis", TRAIN_FOLDER_1)
    X_train_1, y_train_1, train_label_dict_1 = load_data_from_csv_cached(
        TRAIN_FOLDER_1, 
        cache_file=f"cache_{os.path.basename(TRAIN_FOLDER_1)}.pkl"
    )

    print("Chargement des données d'entraînement depuis", TRAIN_FOLDER_2)
    X_train_2, y_train_2, train_label_dict_2 = load_data_from_csv_cached(
        TRAIN_FOLDER_2, 
        cache_file=f"cache_{os.path.basename(TRAIN_FOLDER_2)}.pkl"
    )

    print("Chargement des données de test depuis", TEST_FOLDER_2)
    X_test, y_test, test_label_dict = load_data_from_csv_cached(
        TEST_FOLDER_2,
        cache_file=f"cache_{os.path.basename(TEST_FOLDER_2)}_Test.pkl"
    )
    
    # Vérifier et concaténer les données
    X_train_consigne_1, X_train_reponse_1 = X_train_1
    X_train_consigne_2, X_train_reponse_2 = X_train_2
    
    # Concaténation des features et labels
    print("Concaténation des données des deux dossiers d'entraînement...")
    X_train_consigne_combined = np.vstack([X_train_consigne_1, X_train_consigne_2])
    X_train_reponse_combined = np.vstack([X_train_reponse_1, X_train_reponse_2])
    
    # Unifier les dictionnaires d'étiquettes
    unified_train_dict = {}
    label_idx = 0
    
    # Combiner les dictionnaires des deux ensembles d'entraînement
    for class_name in list(train_label_dict_1.keys()) + list(train_label_dict_2.keys()):
        if class_name not in unified_train_dict:
            unified_train_dict[class_name] = label_idx
            label_idx += 1
    
    # Remapper les étiquettes avec le dictionnaire unifié
    y_train_1_remapped = np.array([unified_train_dict[next(key for key, value in train_label_dict_1.items() if value == label)] 
                                 for label in y_train_1])
    y_train_2_remapped = np.array([unified_train_dict[next(key for key, value in train_label_dict_2.items() if value == label)] 
                                 for label in y_train_2])
    
    # Concaténer les labels
    y_train_combined = np.concatenate([y_train_1_remapped, y_train_2_remapped])
    
    # Division aléatoire du dataset combiné en ensembles d'entraînement et de validation
    print("Division du jeu de données combiné en ensembles d'entraînement et de validation...")
    indices = np.arange(len(y_train_combined))  # Utiliser des indices ordonnés pour éviter le shuffling
    val_size = int(VALIDATION_SPLIT_RATIO * len(indices))  # 20% pour la validation
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # Création des ensembles d'entraînement et de validation
    X_train_consigne = X_train_consigne_combined[train_indices]
    X_train_reponse = X_train_reponse_combined[train_indices]
    X_val_consigne = X_train_consigne_combined[val_indices]
    X_val_reponse = X_train_reponse_combined[val_indices]
    y_train = y_train_combined[train_indices]
    y_val = y_train_combined[val_indices]
    
    # Mise à jour des tuples de données
    X_train = (X_train_consigne, X_train_reponse)
    X_val = (X_val_consigne, X_val_reponse)
    train_label_dict = unified_train_dict
    val_label_dict = unified_train_dict
    
    # Vérifier la taille des tenseurs et les ajuster si nécessaire
    X_train_consigne, X_train_reponse = X_train
    X_test_consigne, X_test_reponse = X_test
    X_val_consigne, X_val_reponse = X_val
    
    print(f"Forme des données d'entraînement chargées - consigne: {X_train_consigne.shape}, réponse: {X_train_reponse.shape}")
    print(f"Forme des données de validation chargées - consigne: {X_val_consigne.shape}, réponse: {X_val_reponse.shape}")
    
    # Nous ne définissons plus de target_shape pour ne pas tronquer les données à 3600 échantillons
    # Conservons uniquement l'ajustement de la dimension temporelle et des caractéristiques si nécessaire
    
    # Fonction pour ajuster les dimensions temporelles et des caractéristiques uniquement (pas le nombre d'échantillons)
    def adjust_tensor_shape(tensor_data, target_time_steps=1000, target_features=3):
        current_shape = tensor_data.shape
        print(f"Forme actuelle: {current_shape}, cible: {current_shape[0]} échantillons × {target_time_steps} pas de temps × {target_features} caractéristiques")
        
        # Ne pas ajuster le nombre d'échantillons, conserver tel quel
        # Ajuster uniquement les dimensions temporelles et de caractéristiques si nécessaire
        if len(current_shape) == 3 and (current_shape[1] != target_time_steps or current_shape[2] != target_features):
            # Créer un nouveau tensor avec la bonne dimension temporelle
            new_tensor = np.zeros((current_shape[0], target_time_steps, target_features))
            
            # Copier les données existantes (en tronquant ou en laissant des zéros)
            time_steps = min(current_shape[1], target_time_steps)
            features = min(current_shape[2], target_features)
            new_tensor[:, :time_steps, :features] = tensor_data[:, :time_steps, :features]
            
            tensor_data = new_tensor
            print(f"Dimensions temporelles/caractéristiques ajustées à {target_time_steps}/{target_features}")
        
        return tensor_data
    
    # Ajuster uniquement la dimension temporelle et des caractéristiques si nécessaire 
    # tout en conservant le nombre d'échantillons existant
    print("Ajustement des dimensions temporelles et de caractéristiques si nécessaire...")
    X_train_consigne = adjust_tensor_shape(X_train_consigne)
    X_train_reponse = adjust_tensor_shape(X_train_reponse)
    
    print("Ajustement des dimensions des données de validation...")
    X_val_consigne = adjust_tensor_shape(X_val_consigne)
    X_val_reponse = adjust_tensor_shape(X_val_reponse)
    
    # Pour les données de test
    print("Ajustement des dimensions des données de test...")
    X_test_consigne = adjust_tensor_shape(X_test_consigne)
    X_test_reponse = adjust_tensor_shape(X_test_reponse)
    
    # Mettre à jour les tuples X_train, X_test et X_val
    
    X_train = (X_train_consigne, X_train_reponse)
    X_test = (X_test_consigne, X_test_reponse)
    X_val = (X_val_consigne, X_val_reponse)
    
    # Afficher les dimensions finales
    print(f"Forme finale des données d'entraînement - consigne: {X_train_consigne.shape}, réponse: {X_train_reponse.shape}")
    print(f"Forme finale des données de validation - consigne: {X_val_consigne.shape}, réponse: {X_val_reponse.shape}")
    print(f"Forme finale des données de test - consigne: {X_test_consigne.shape}, réponse: {X_test_reponse.shape}")

    # Vérifier la cohérence des dictionnaires d'étiquettes
    if train_label_dict != test_label_dict:
        print("Attention: Les dictionnaires d'étiquettes ne sont pas identiques entre les ensembles")
        
        # Créer un dictionnaire unifié global (train, val, test)
        unified_dict = {}
        label_idx = 0
        
        # Ajouter d'abord les classes du jeu d'entraînement
        for class_name in train_label_dict:
            if class_name not in unified_dict:
                unified_dict[class_name] = label_idx
                label_idx += 1
        
        # Ajouter les classes éventuellement manquantes du jeu de test
        for class_name in test_label_dict:
            if class_name not in unified_dict:
                unified_dict[class_name] = label_idx
                label_idx += 1
        
        print("Dictionnaire d'étiquettes unifié créé avec", len(unified_dict), "classes")
        
        # Remapper les étiquettes si nécessaire
        if train_label_dict != unified_dict:
            remapped_labels = np.zeros_like(y_train)
            for i, label in enumerate(y_train):
                old_class_name = next(key for key, value in train_label_dict.items() if value == label)
                remapped_labels[i] = unified_dict[old_class_name]
            y_train = remapped_labels
            
            # Remapper également les étiquettes de validation
            remapped_labels_val = np.zeros_like(y_val)
            for i, label in enumerate(y_val):
                old_class_name = next(key for key, value in train_label_dict.items() if value == label)
                remapped_labels_val[i] = unified_dict[old_class_name]
            y_val = remapped_labels_val
            
            train_label_dict = unified_dict
            val_label_dict = unified_dict
        
        if test_label_dict != unified_dict:
            remapped_labels = np.zeros_like(y_test)
            for i, label in enumerate(y_test):
                old_class_name = next(key for key, value in test_label_dict.items() if value == label)
                remapped_labels[i] = unified_dict[old_class_name]
            y_test = remapped_labels
            test_label_dict = unified_dict

# Si nous n'avons pas de données de validation à partir du format .mat, 
# créer un ensemble de validation à partir des données d'entraînement
if X_val is None:
    print("Création d'un ensemble de validation à partir des données d'entraînement...")
    val_size = int(VALIDATION_SPLIT_RATIO * len(y_train))  # 20% pour la validation
    # Utiliser des indices consécutifs sans shuffling pour séparer les données
    indices = np.arange(len(y_train))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # Séparer les données d'entraînement et de validation
    X_train_consigne, X_train_reponse = X_train
    
    X_val_consigne = X_train_consigne[val_indices]
    X_val_reponse = X_train_reponse[val_indices]
    y_val = y_train[val_indices]
    
    # Mettre à jour les données d'entraînement
    X_train_consigne = X_train_consigne[train_indices]
    X_train_reponse = X_train_reponse[train_indices]
    y_train = y_train[train_indices]
    
    X_train = (X_train_consigne, X_train_reponse)
    X_val = (X_val_consigne, X_val_reponse)

class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(GatedTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # init unbias gate (sigmoid)  
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        # Si x est de dimension 2, on le reshape pour avoir une dimension séquence de 1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
            
        # Première attention - standard
        norm_x = self.norm1(x)
        attn_output1 = self.attn1(norm_x, norm_x, norm_x)[0]
        attn_output1 = self.dropout(attn_output1)
        
        # Deuxième attention - nous voulons des chemins d'attention différents
        # Au lieu de transposer les dimensions, nous utilisons simplement une deuxième attention
        norm_x2 = self.norm2(x)
        attn_output2 = self.attn2(norm_x2, norm_x2, norm_x2)[0]
        attn_output2 = self.dropout(attn_output2)
        
        # Fusion par gate via sigmoid pour restreindre entre 0 et 1
        g = torch.sigmoid(self.gate)
        gated_output = g * attn_output1 + (1 - g) * attn_output2
        x = x + gated_output
        
        # Feed forward avec pré-normalization
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x


class GTN(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, nhead=4, num_layers=2, dropout=0.1):
        super(GTN, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.transformer_blocks = nn.ModuleList([
            GatedTransformerBlock(model_dim, nhead, dropout) for _ in range(num_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x peut être [batch, features] ou [batch, seq_len, features]
        batch_size = x.size(0)
        
        # Si x est déjà 3D, on utilise tel quel, sinon on ajoute une dimension
        orig_shape = len(x.shape)
        if orig_shape == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        x = self.input_proj(x)
        
        # Passer à travers les blocs transformer
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # [batch, model_dim, seq_len]
        x = self.global_pool(x).view(batch_size, -1)
        
        # Classification
        return self.classifier(x)

class GTN3D(nn.Module):
    def __init__(self, consigne_dim, reponse_dim, model_dim, num_classes, nhead=4, num_layers=2, dropout=0.1):
        super(GTN3D, self).__init__()
        # Vérifier que les dimensions correspondent pour calculer une différence
        assert consigne_dim == reponse_dim, "Les dimensions des consignes et réponses doivent être identiques pour calculer leur différence"
        
        # Projections pour les trois types d'informations: consigne, réponse, et leur différence
        self.consigne_proj = nn.Linear(consigne_dim, model_dim)
        self.reponse_proj = nn.Linear(reponse_dim, model_dim)
        self.difference_proj = nn.Linear(consigne_dim, model_dim)  # La différence a la même dimension que chaque entrée
        
        # Blocs transformer séparés pour chaque type d'information
        self.consigne_transformers = nn.ModuleList([
            GatedTransformerBlock(model_dim, nhead, dropout) for _ in range(num_layers)
        ])
        self.reponse_transformers = nn.ModuleList([
            GatedTransformerBlock(model_dim, nhead, dropout) for _ in range(num_layers)
        ])
        self.difference_transformers = nn.ModuleList([
            GatedTransformerBlock(model_dim, nhead, dropout) for _ in range(num_layers)
        ])
        
        # Bloc transformer pour fusionner les informations
        self.fusion_transformer = GatedTransformerBlock(model_dim, nhead, dropout)
        
        # Global pooling et classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)
        
    def forward(self, x_consigne, x_reponse):
        # x_consigne: [batch, features_consigne]
        # x_reponse: [batch, features_reponse]
        batch_size = x_consigne.size(0)
        
        # Calculer la différence entre consigne et réponse (écart)
        x_difference = x_consigne - x_reponse  # Cette différence capture l'écart entre consigne et réponse
        
        # Ajouter une dimension temporelle pour les transformers
        x_consigne = x_consigne.unsqueeze(1)      # [batch, 1, features_consigne]
        x_reponse = x_reponse.unsqueeze(1)        # [batch, 1, features_reponse]
        x_difference = x_difference.unsqueeze(1)  # [batch, 1, features_difference]
        
        # Projeter dans l'espace du modèle
        x_consigne = self.consigne_proj(x_consigne)      # [batch, 1, model_dim]
        x_reponse = self.reponse_proj(x_reponse)         # [batch, 1, model_dim]
        x_difference = self.difference_proj(x_difference) # [batch, 1, model_dim]
        
        # Traitement des consignes
        for block in self.consigne_transformers:
            x_consigne = block(x_consigne)
            
        # Traitement des réponses
        for block in self.reponse_transformers:
            x_reponse = block(x_reponse)
        
        # Traitement des différences - c'est ici que le modèle apprend à partir des écarts
        for block in self.difference_transformers:
            x_difference = block(x_difference)
            
        # Concaténer les représentations sur la dimension temporelle pour obtenir une séquence 3D
        # [batch, 3, model_dim] - maintenant nous avons 3 éléments: consigne, réponse et différence
        x_combined = torch.cat([x_consigne, x_reponse, x_difference], dim=1)
        
        # Fusion par transformer des trois types d'information
        fusion_output = self.fusion_transformer(x_combined)
        
        # Global pooling
        fusion_output = fusion_output.transpose(1, 2)  # [batch, model_dim, 3]
        pooled = self.global_pool(fusion_output).view(batch_size, -1)  # [batch, model_dim]
        
        # Classification
        return self.classifier(pooled)

class GTNResidu(nn.Module):
    def __init__(self, consigne_dim, reponse_dim, model_dim=128, num_classes=9, nhead=8, num_layers=6, dropout=0.1):
        super(GTNResidu, self).__init__()
        # Vérifier que les dimensions correspondent pour calculer une différence
        assert consigne_dim == reponse_dim, "Les dimensions des consignes et réponses doivent être identiques pour calculer leur différence"
        
        # Augmenter la dimension du modèle pour plus de capacité
        self.model_dim = model_dim
        
        # Feature dimension est le nombre de canaux d'entrée (3 pour les consignes/réponses)
        self.feature_dim = 3  # Fixé à 3 canaux (pour les 3 dimensions x,y,z dans chaque signal)
        
        # Projection initiale temporelle (1D CNN pour traiter les séquences temporelles)
        self.temporal_conv = nn.Sequential(
            # Entrée: [batch, feature_dim (3), seq_len] → Sortie: [batch, model_dim, seq_len]
            nn.Conv1d(self.feature_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
        )
        
        # Projection MLPs pour les caractéristiques extraites
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim*2),
            nn.BatchNorm1d(model_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim*2, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
        )
        
        # Blocs transformer améliorés pour traiter les séquences temporelles
        self.residu_transformers = nn.ModuleList([
            EnhancedGatedTransformerBlock(model_dim, nhead, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Adaptative pooling - accepte n'importe quelle longueur de séquence
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP de classification profond avec dropout progressif
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim*2),
            nn.BatchNorm1d(model_dim*2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # Dropout plus agressif dans les couches supérieures
            nn.Linear(model_dim*2, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )
        
        # Initialisation des poids pour une meilleure convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialisation des poids avec une stratégie adaptée au type de couche"""
        if isinstance(module, nn.Linear):
            # He initialization pour les couches linéaires avec ReLU/GELU
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, x_consigne, x_reponse):
        """
        x_consigne : [batch, seq_len, consigne_dim]
        x_reponse : [batch, seq_len, reponse_dim]
        """
        # Calculer le résidu (différence entre consigne et réponse) temporel
        x_residu = x_consigne - x_reponse  # [batch, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = x_residu.shape
        

        
        # Vérifier si nous recevons des données au format thesis_ai.py (3600×1000×3)
        # ou au format standard GTN.py (batch×3)
        if len(x_residu.shape) == 3 and feature_dim != self.feature_dim:
            if feature_dim > self.feature_dim:
                # Si nous avons trop de canaux dans la 3ème dimension, prenons juste les 3 premiers
                print(f"ADAPTATION: Utilisation des 3 premières dimensions sur {feature_dim} disponibles")
                x_residu = x_residu[:, :, :self.feature_dim]
            else:
                # Si nous n'avons pas assez de canaux, nous devons adapter
                print(f"ERREUR: Dimension de caractéristiques incompatible: attendu {self.feature_dim}, reçu {feature_dim}")
                raise ValueError(f"La dimension d'entrée {feature_dim} est incompatible avec le modèle qui attend {self.feature_dim}")
        
        # Transposer pour Conv1D (attend [batch, channels, seq_len])
        x_residu = x_residu.transpose(1, 2)  # [batch, feature_dim, seq_len]

        
        # Appliquer la convolution 1D temporelle
        x = self.temporal_conv(x_residu)  # [batch, model_dim, seq_len]

        
        # Transposer pour le transformer (attend [batch, seq_len, model_dim])
        x = x.transpose(1, 2)  # [batch, seq_len, model_dim]
        
        # Traiter chaque étape temporelle avec les transformers
        for block in self.residu_transformers:
            x = block(x)  # [batch, seq_len, model_dim]
        
        # Global pooling sur la dimension temporelle pour obtenir une représentation fixe
        x = x.transpose(1, 2)  # [batch, model_dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, model_dim]
        
        # Projection MLP avant classification
        x = x + self.projection(x)  # Connexion résiduelle
        
        # Classification finale
        logits = self.classifier(x)  # [batch, num_classes]
        
        return logits

class EnhancedGatedTransformerBlock(nn.Module):
    """Bloc transformer amélioré avec gating multi-couches et mécanisme d'attention plus robuste"""
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=None):
        super(EnhancedGatedTransformerBlock, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        # Normalisations pré-attention (une seule normalisation pour réduire la consommation mémoire)
        self.norm_pre = nn.LayerNorm(d_model)
        self.norm_post = nn.LayerNorm(d_model)
        
        # Réduire le nombre de têtes pour l'attention
        self.attn = nn.MultiheadAttention(
            d_model, 
            min(nhead, 4), # Limiter le nombre de têtes pour réduire la consommation mémoire
            dropout=dropout, 
            batch_first=True
        )
        
        # Gate simplifiée (utilise moins de mémoire)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward plus léger avec GELU
        # Utiliser une structure plus simple pour réduire la consommation mémoire
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
    def forward(self, x):
        # Si x est de dimension 2, ajout d'une dimension séquentielle
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
            
        # Normalisation avant attention
        norm_x = self.norm_pre(x)
        
        # Attention avec normalisation pré-attention
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        attn_output = self.dropout(attn_output)
        
        # Connexion résiduelle
        x = x + attn_output
        
        # Normalisation avant feed-forward
        ff_input = self.norm_post(x)
        
        # Feed forward avec connexion résiduelle
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x

def is_old_model_format(model_path):
    """Vérifie si le fichier du modèle est dans l'ancien format."""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        # Vérifier quelques clés caractéristiques de l'ancien modèle
        return 'residu_proj.weight' in state_dict and 'classifier.weight' in state_dict
    except:
        return False

class LegacyGTNResidu(nn.Module):
    """Version compatible avec l'ancien modèle pour charger les poids"""
    def __init__(self, consigne_dim, reponse_dim, model_dim, num_classes, nhead=4, num_layers=2, dropout=0.1):
        super(LegacyGTNResidu, self).__init__()
        # Structure simplifiée comme dans l'ancienne architecture
        self.residu_proj = nn.Linear(consigne_dim, model_dim)
        
        # Blocs transformer pour le résidu (version ancienne)
        self.residu_transformers = nn.ModuleList([
            GatedTransformerBlock(model_dim, nhead, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Classification - version simple
        self.classifier = nn.Linear(model_dim, num_classes)
        
    def forward(self, x_consigne, x_reponse):
        # Calculer uniquement le résidu (différence entre consigne et réponse)
        x_residu = x_consigne - x_reponse
        
        # Projeter dans l'espace du modèle
        x = self.residu_proj(x_residu)
        
        # Ajouter une dimension temporelle pour les transformers
        x = x.unsqueeze(1)  # [batch, 1, model_dim]
        
        # Traitement du résidu par la pile de transformers
        for block in self.residu_transformers:
            x = block(x)
        
        # Supprimer la dimension séquentielle artificielle
        x = x.squeeze(1)  # [batch, model_dim]
        
        # Classification
        x = self.classifier(x)
        
        return x

def train_model_3d(model, X_train_consigne, X_train_reponse, y_train, 
                  X_val_consigne, X_val_reponse, y_val, 
                  epochs=100, batch_size=64, learning_rate=3e-4):
    # Déplacer le modèle vers le périphérique (CPU/GPU)
    model = model.to(device)
    
    # Appliquer l'augmentation de données sur l'ensemble d'entraînement uniquement
    print("Application de l'augmentation de données sur l'ensemble d'entraînement...")
    X_train_consigne_aug, X_train_reponse_aug, y_train_aug = apply_data_augmentation(
        X_train_consigne, X_train_reponse, y_train, 
        noise_level=NOISE_LEVEL,
        num_augmentations=NUM_AUGMENTATIONS
    )
    
    # Vérifier si les entrées sont déjà des tenseurs et sur le bon périphérique
    if isinstance(X_train_consigne_aug, torch.Tensor):
        X_train_consigne_tensor = X_train_consigne_aug
    else:
        X_train_consigne_tensor = torch.FloatTensor(X_train_consigne_aug).to(device)
    
    if isinstance(X_train_reponse_aug, torch.Tensor):
        X_train_reponse_tensor = X_train_reponse_aug
    else:
        X_train_reponse_tensor = torch.FloatTensor(X_train_reponse_aug).to(device)
    
    if isinstance(y_train_aug, torch.Tensor):
        y_train_tensor = y_train_aug
    else:
        y_train_tensor = torch.LongTensor(y_train_aug).to(device)
    
    # Vérifier si les entrées de validation sont déjà des tenseurs
    if isinstance(X_val_consigne, torch.Tensor):
        X_val_consigne_tensor = X_val_consigne
    else:
        X_val_consigne_tensor = torch.FloatTensor(X_val_consigne).to(device)
    
    if isinstance(X_val_reponse, torch.Tensor):
        X_val_reponse_tensor = X_val_reponse
    else:
        X_val_reponse_tensor = torch.FloatTensor(X_val_reponse).to(device)
    
    if isinstance(y_val, torch.Tensor):
        y_val_tensor = y_val
    else:
        y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # S'assurer que tous les tenseurs sont sur le même périphérique que le modèle
    device_model = next(model.parameters()).device
    X_train_consigne_tensor = X_train_consigne_tensor.to(device_model)
    X_train_reponse_tensor = X_train_reponse_tensor.to(device_model)
    y_train_tensor = y_train_tensor.to(device_model)
    X_val_consigne_tensor = X_val_consigne_tensor.to(device_model)
    X_val_reponse_tensor = X_val_reponse_tensor.to(device_model)
    y_val_tensor = y_val_tensor.to(device_model)
    
    # Création des DataLoaders
    train_dataset = TensorDataset(X_train_consigne_tensor, X_train_reponse_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_consigne_tensor, X_val_reponse_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Définition du critère de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler de learning rate avec warm-up et cosine annealing
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs//3, T_mult=1, eta_min=learning_rate/100
    )
    
    # Historique d'entrainement
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    # Pour early stopping
    patience = 15
    patience_counter = 0
    best_val_loss = float('inf')
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for consigne, reponse, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(consigne, reponse)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradient pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Évaluation sur l'ensemble de validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for consigne, reponse, labels in val_loader:
                outputs = model(consigne, reponse)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
        
        # Sauvegarde du meilleur modèle
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), 'best_gtn_residu_model.pth')
            print(f"Meilleur modèle sauvegardé avec accuracy: {best_val_acc:.2f}%")
            
        # Early stopping sur la perte de validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping après {epoch+1} epochs")
            break
    
    # Charger le meilleur modèle pour la suite
    model.load_state_dict(torch.load('best_gtn_residu_model.pth'))
    
    return model, train_losses, val_losses, val_accuracies

def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, learning_rate=0.001):
    # Conversion des données en tenseurs PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Création des DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Définition du critère de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Historique d'entrainement
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Évaluation sur l'ensemble de validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
    
    return model, train_losses, val_losses, val_accuracies

def plot_confusion_matrix(model, X_test, y_test, class_names):
    # Conversion des données en tenseurs PyTorch
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, predicted.numpy())
    
    # Normalisation de la matrice de confusion
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    

    
    # Calcul de l'accuracy globale
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Précision globale: {accuracy:.4f}")
    
    return cm, accuracy

def plot_confusion_matrix_3d(model, X_test_consigne, X_test_reponse, y_test, class_names):
    # Utiliser un batch large pour traiter toutes les données d'un coup
    try:
        # Essayer de traiter toutes les données en une seule fois
        X_test_consigne_tensor = torch.FloatTensor(X_test_consigne).to(device)
        X_test_reponse_tensor = torch.FloatTensor(X_test_reponse).to(device)
        
        # Prédiction
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_consigne_tensor, X_test_reponse_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
            
    except RuntimeError:
        # En cas d'erreur de mémoire, utiliser un traitement par lots avec un batch size modéré
        print("Traitement des données par lots pour éviter les problèmes de mémoire...")
        batch_size = 1000  # Valeur modérée
        
        dataset = TensorDataset(
            torch.FloatTensor(X_test_consigne),
            torch.FloatTensor(X_test_reponse)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Récupérer toutes les prédictions par lots
        all_predictions = []
        
        # Prédiction par lots
        model.eval()
        with torch.no_grad():
            for consigne_batch, reponse_batch in dataloader:
                # Déplacer les données vers le périphérique
                consigne_batch = consigne_batch.to(device)
                reponse_batch = reponse_batch.to(device)
                
                # Obtenir les prédictions
                outputs = model(consigne_batch, reponse_batch)
                _, predicted = torch.max(outputs, 1)
                
                # Stocker les prédictions
                all_predictions.append(predicted.cpu().numpy())
        
        # Concaténer toutes les prédictions
        predicted = np.concatenate(all_predictions)
    
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, predicted)
    
    # Normalisation de la matrice de confusion
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    

    
    # Calcul de l'accuracy globale
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Précision globale: {accuracy:.4f}")
    
    # Libérer la mémoire
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return cm, accuracy

def calculate_accuracy_3d(model, X_consigne, X_reponse, y_true):
    """Calculate accuracy for 3D input data (consigne and reponse) with proper memory management."""
    # Use a smaller batch size to avoid memory issues
    batch_size = 32  # Reduce batch size to minimize memory usage
        
    # Always use batched processing to prevent memory errors
    print("Traitement par lots pour éviter les problèmes de mémoire (batch_size={})...".format(batch_size))
    
    # Ensure all data lengths match to prevent the size mismatch error
    n_samples = min(len(X_consigne), len(X_reponse), len(y_true))
    
    # Create a properly aligned dataset
    dataset = TensorDataset(
        torch.FloatTensor(X_consigne[:n_samples]),
        torch.FloatTensor(X_reponse[:n_samples]),
        torch.LongTensor(y_true[:n_samples])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    # Prediction by batch
    model.eval()
    with torch.no_grad():
        for batch_idx, (consigne_batch, reponse_batch, y_batch) in enumerate(dataloader):
            # Print progress
            if batch_idx % 20 == 0:
                print(f"  Traitement du lot {batch_idx}/{len(dataloader)}...")
                
            # Move data to device
            consigne_batch = consigne_batch.to(device)
            reponse_batch = reponse_batch.to(device)
            
            # Get predictions
            outputs = model(consigne_batch, reponse_batch)
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions
            total += y_batch.size(0)
            correct += (predicted.cpu() == y_batch).sum().item()
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    return accuracy

# Fonction d'augmentation de données avec bruit gaussien
def augment_data_with_gaussian_noise(X_consigne, X_reponse, y, noise_level=0.01, num_augmentations=1):
    """
    Augmente les données en ajoutant un bruit gaussien aux caractéristiques existantes
    
    Args:
        X_consigne: Données de consigne à augmenter
        X_reponse: Données de réponse correspondantes à augmenter
        y: Étiquettes correspondantes
        noise_level: Niveau de bruit relatif à ajouter (écart-type du bruit)
        num_augmentations: Nombre de versions augmentées à créer pour chaque échantillon
    
    Returns:
        X_consigne_aug, X_reponse_aug, y_aug: Données augmentées
    """
    # Calculer la magnitude du bruit basée sur l'écart-type des données originales
    consigne_std = np.std(X_consigne)
    reponse_std = np.std(X_reponse)
    
    consigne_noise_magnitude = consigne_std * noise_level
    reponse_noise_magnitude = reponse_std * noise_level
    
    # Préparer les listes pour stocker les données augmentées
    X_consigne_aug = [X_consigne]
    X_reponse_aug = [X_reponse]
    y_aug = [y]
    
    # Générer les versions augmentées
    for _ in range(num_augmentations):
        # Générer un bruit gaussien de même forme que les données d'origine
        consigne_noise = np.random.normal(0, consigne_noise_magnitude, size=X_consigne.shape)
        reponse_noise = np.random.normal(0, reponse_noise_magnitude, size=X_reponse.shape)
        
        # Ajouter le bruit aux données
        X_consigne_noisy = X_consigne + consigne_noise
        X_reponse_noisy = X_reponse + reponse_noise
        
        # Ajouter à nos listes
        X_consigne_aug.append(X_consigne_noisy)
        X_reponse_aug.append(X_reponse_noisy)
        y_aug.append(y)
    
    # Concaténer toutes les versions augmentées
    X_consigne_aug = np.vstack(X_consigne_aug)
    X_reponse_aug = np.vstack(X_reponse_aug)
    y_aug = np.concatenate(y_aug)
    
    return X_consigne_aug, X_reponse_aug, y_aug

def split_test_data_for_transfer_learning(X_test_consigne, X_test_reponse, y_test, split_ratio=0.5):
    """
    Divise les données de test en deux parties: une pour le fine-tuning (transfer learning)
    et une pour l'évaluation finale
    
    Args:
        X_test_consigne: Données de consigne de test
        X_test_reponse: Données de réponse de test
        y_test: Étiquettes de test
        split_ratio: Proportion des données à utiliser pour le fine-tuning (0.5 = moitié)
    
    Returns:
        (X_finetune_consigne, X_finetune_reponse, y_finetune): Données pour le fine-tuning
        (X_eval_consigne, X_eval_reponse, y_eval): Données pour l'évaluation
    """
    # Mélanger les indices de manière aléatoire mais reproductible
    np.random.seed(42)  # Pour la reproductibilité
    indices = np.arange(len(y_test))
    np.random.shuffle(indices)
    
    # Calculer le point de séparation
    split_point = int(len(indices) * split_ratio)
    
    # Diviser les indices
    finetune_indices = indices[:split_point]
    eval_indices = indices[split_point:]
    
    # Extraire les sous-ensembles
    X_finetune_consigne = X_test_consigne[finetune_indices]
    X_finetune_reponse = X_test_reponse[finetune_indices]
    y_finetune = y_test[finetune_indices]
    
    X_eval_consigne = X_test_consigne[eval_indices]
    X_eval_reponse = X_test_reponse[eval_indices]
    y_eval = y_test[eval_indices]
    
    print(f"Division des données de test: {len(y_finetune)} échantillons pour le fine-tuning, {len(y_eval)} pour l'évaluation")
    
    return (X_finetune_consigne, X_finetune_reponse, y_finetune), (X_eval_consigne, X_eval_reponse, y_eval)

def transfer_learning_with_advanced_finetune(model, X_finetune_consigne, X_finetune_reponse, y_finetune, 
                                     epochs=15, batch_size=16, initial_lr=5e-5, noise_level=0.01, 
                                     num_augmentations=2):
    """
    Version améliorée du transfer learning appliquant un fine-tuning progressif et ciblé avec une 
    stratégie d'apprentissage par couches. Cette approche est plus efficace pour adapter le modèle
    aux données réelles en ciblant les couches les plus pertinentes et en débloquant progressivement
    les couches inférieures.
    
    Args:
        model: Modèle pré-entraîné à ajuster
        X_finetune_consigne, X_finetune_reponse, y_finetune: Données pour le fine-tuning
        epochs: Nombre total d'époques (réparties entre les phases)
        batch_size: Taille du batch pour l'entraînement 
        initial_lr: Taux d'apprentissage initial
        noise_level: Niveau de bruit pour l'augmentation des données
        num_augmentations: Nombre de versions augmentées à créer
    
    Returns:
        Le modèle ajusté (fine-tuned)
    """
    print("Démarrage du fine-tuning avancé avec unfreezing progressif des couches...")
    
    # Déterminer automatiquement le nombre de classes
    num_classes = len(np.unique(y_finetune))
    print(f"Nombre de classes détecté: {num_classes}")
    
    # Dimensions du modèle
    consigne_dim = X_finetune_consigne.shape[2] if len(X_finetune_consigne.shape) > 2 else X_finetune_consigne.shape[1]
    reponse_dim = X_finetune_reponse.shape[2] if len(X_finetune_reponse.shape) > 2 else X_finetune_reponse.shape[1]
    model_dim = model.model_dim if hasattr(model, 'model_dim') else 64
    
    # Récupération du nombre de couches transformer du modèle source
    if hasattr(model, 'residu_transformers'):
        source_num_layers = len(model.residu_transformers)
    else:
        source_num_layers = 2  # Valeur par défaut si non trouvé
    
    print(f"Modèle source: {source_num_layers} couches transformer détectées")
    
    # Création d'une copie du modèle pour le fine-tuning avec LE MÊME NOMBRE de couches
    model_transfer = type(model)(
        consigne_dim, 
        reponse_dim,  
        model_dim, 
        num_classes,
        nhead=8,  # Augmenter le nombre de têtes d'attention
        num_layers=source_num_layers,  # Utiliser le même nombre de couches que le modèle source
        dropout=0.15  # Légèrement plus de dropout pour éviter le surapprentissage
    )
    
    # Copier tous les poids du modèle original
    try:
        model_transfer.load_state_dict(model.state_dict())
        print("Poids du modèle source chargés avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des poids: {e}")
        print("Tentative de chargement partiel des poids...")
        
        # Chargement manuel et sélectif des poids du modèle source
        source_state_dict = model.state_dict()
        target_state_dict = model_transfer.state_dict()
        
        # Ne charger que les poids qui correspondent entre les deux modèles
        for name, param in source_state_dict.items():
            if name in target_state_dict and target_state_dict[name].shape == param.shape:
                target_state_dict[name].copy_(param)
                print(f"Copié: {name}")
        
        model_transfer.load_state_dict(target_state_dict)
        print("Chargement partiel des poids terminé")
    
    model_transfer = model_transfer.to(device)
    
    # Augmentation avancée des données avec plusieurs stratégies
    print("Application d'une augmentation avancée des données de fine-tuning...")
    
    # Convertir en NumPy si nécessaire
    X_consigne_np = X_finetune_consigne.cpu().numpy() if isinstance(X_finetune_consigne, torch.Tensor) else X_finetune_consigne
    X_reponse_np = X_finetune_reponse.cpu().numpy() if isinstance(X_finetune_reponse, torch.Tensor) else X_finetune_reponse
    y_np = y_finetune
    
    # 1. Augmentation avec bruit gaussien
    X_consigne_aug, X_reponse_aug, y_aug = augment_data_with_gaussian_noise(
        X_consigne_np, X_reponse_np, y_np, 
        noise_level=noise_level,
        num_augmentations=num_augmentations
    )
    
    # 2. Augmentation supplémentaire: Time warping léger (déformation temporelle)
    # Pour des données séquentielles 3D (séquences temporelles)
    if len(X_consigne_np.shape) == 3 and X_consigne_np.shape[1] > 10:
        print("Application de time warping sur les séquences temporelles...")
        # Facteur de déformation (léger pour ne pas trop modifier le signal)
        warp_factor = 0.02
        
        # Nombre de séquences à augmenter
        n_to_augment = min(100, len(X_consigne_np))  # Limiter pour éviter explosion mémoire
        indices = np.random.choice(len(X_consigne_np), n_to_augment, replace=False)
        
        X_consigne_warped_list = []
        X_reponse_warped_list = []
        y_warped_list = []
        
        for idx in indices:
            # Créer une légère déformation temporelle 
            seq_len = X_consigne_np.shape[1]
            time_stretch = np.clip(np.random.normal(1, warp_factor), 0.98, 1.02)
            
            # Rééchantilloner la séquence
            new_len = int(seq_len * time_stretch)
            idxs = np.linspace(0, seq_len-1, new_len).astype(int)
            
            # Appliquer la déformation et redimensionner à la taille originale
            X_consigne_warped = X_consigne_np[idx, idxs, :]
            X_reponse_warped = X_reponse_np[idx, idxs, :]
            
            # Redimensionner à la taille originale si nécessaire
            if new_len != seq_len:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, new_len)
                x_new = np.linspace(0, 1, seq_len)
                
                # Interpolation pour chaque dimension de features
                X_consigne_resampled = np.zeros((seq_len, X_consigne_np.shape[2]))
                X_reponse_resampled = np.zeros((seq_len, X_reponse_np.shape[2]))
                
                for i in range(X_consigne_np.shape[2]):
                    f_consigne = interp1d(x_old, X_consigne_warped[:, i], kind='linear')
                    X_consigne_resampled[:, i] = f_consigne(x_new)
                    
                    f_reponse = interp1d(x_old, X_reponse_warped[:, i], kind='linear')
                    X_reponse_resampled[:, i] = f_reponse(x_new)
                
                X_consigne_warped = X_consigne_resampled
                X_reponse_warped = X_reponse_resampled
            
            # Ajouter aux listes
            X_consigne_warped_list.append(X_consigne_warped)
            X_reponse_warped_list.append(X_reponse_warped)
            y_warped_list.append(y_np[idx])
        
        # Convertir en array et ajouter aux données augmentées
        X_consigne_warped_array = np.array(X_consigne_warped_list)
        X_reponse_warped_array = np.array(X_reponse_warped_list)
        y_warped_array = np.array(y_warped_list)
        
        X_consigne_aug = np.vstack([X_consigne_aug, X_consigne_warped_array])
        X_reponse_aug = np.vstack([X_reponse_aug, X_reponse_warped_array])
        y_aug = np.concatenate([y_aug, y_warped_array])
    
    print(f"Dimensions finales après augmentation - X_consigne: {X_consigne_aug.shape}, X_reponse: {X_reponse_aug.shape}")
    
    # Convertir en tensors PyTorch
    X_consigne_tensor = torch.FloatTensor(X_consigne_aug).to(device)
    X_reponse_tensor = torch.FloatTensor(X_reponse_aug).to(device)
    y_tensor = torch.LongTensor(y_aug).to(device)
    
    # Création du DataLoader
    dataset = TensorDataset(X_consigne_tensor, X_reponse_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Strategy de dégel progressif des couches:
    # 1. D'abord, geler toutes les couches sauf la classification
    # 2. Puis dégeler progressivement les couches en partant du haut (les plus proches de la sortie)
    
    # ===== PHASE 1: Fine-tuning de la couche de classification uniquement =====
    print("Phase 1/3: Fine-tuning des couches de classification uniquement...")
    
    # Geler tous les paramètres
    for param in model_transfer.parameters():
        param.requires_grad = False
        
    # Dégeler uniquement les couches de classification
    for param in model_transfer.classifier.parameters():
        param.requires_grad = True
    
    # Optimiseur pour phase 1 - taux d'apprentissage plus élevé pour la classification
    optimizer_phase1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_transfer.parameters()), 
        lr=initial_lr * 3.0,
        weight_decay=1e-4
    )
    
    # Entraînement phase 1
    _train_phase(model_transfer, dataloader, optimizer_phase1, device, 
                 epochs=int(epochs * 0.25), phase_name="Phase 1")
    
    # ===== PHASE 2: Fine-tuning des couches supérieures =====
    print("Phase 2/3: Fine-tuning des couches supérieures (Classification + Transformers supérieurs)...")
    
    # Dégeler les transformers supérieurs (moitié supérieure) et projection
    # Garder les couches des convolutions temporelles gelées
    if hasattr(model_transfer, 'residu_transformers') and len(model_transfer.residu_transformers) > 1:
        num_transformer_blocks = len(model_transfer.residu_transformers)
        # Débloquer la moitié supérieure des blocs transformer
        for i in range(num_transformer_blocks // 2, num_transformer_blocks):
            for param in model_transfer.residu_transformers[i].parameters():
                param.requires_grad = True
    
    # Dégeler la couche de projection également
    if hasattr(model_transfer, 'projection'):
        for param in model_transfer.projection.parameters():
            param.requires_grad = True
    
    # Optimiseur pour phase 2 - taux d'apprentissage intermédiaire pour les couches supérieures
    optimizer_phase2 = torch.optim.AdamW([
        {'params': model_transfer.classifier.parameters(), 'lr': initial_lr},
        {'params': (p for n, p in model_transfer.named_parameters() 
                    if 'residu_transformers' in n and p.requires_grad), 'lr': initial_lr * 0.5},
        {'params': model_transfer.projection.parameters() if hasattr(model_transfer, 'projection') else [], 
         'lr': initial_lr * 0.8}
    ], weight_decay=2e-5)
    
    # Scheduler pour réduire progressivement le taux d'apprentissage
    scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=int(epochs * 0.35), eta_min=initial_lr/10
    )
    
    # Entraînement phase 2
    _train_phase(model_transfer, dataloader, optimizer_phase2, device, 
                 epochs=int(epochs * 0.35), phase_name="Phase 2", scheduler=scheduler_phase2)
    
    # ===== PHASE 3: Fine-tuning du modèle entier =====
    print("Phase 3/3: Fine-tuning de toutes les couches (modèle complet)...")
    
    # Dégeler toutes les couches
    for param in model_transfer.parameters():
        param.requires_grad = True
        
    # Optimiseur pour phase 3 - taux d'apprentissage différent pour chaque groupe de couches
    param_groups = [
        {'params': model_transfer.classifier.parameters(), 'lr': initial_lr * 0.8}  # Taux le plus haut pour classifier
    ]
    
    # Ajouter les couches temporelles si elles existent
    if hasattr(model_transfer, 'temporal_conv'):
        param_groups.append({
            'params': model_transfer.temporal_conv.parameters(),
            'lr': initial_lr * 0.1  # Taux très bas pour les premières couches
        })
    
    # Ajouter la couche de projection si elle existe
    if hasattr(model_transfer, 'projection'):
        param_groups.append({
            'params': model_transfer.projection.parameters(),
            'lr': initial_lr * 0.6  # Taux plus haut pour projection
        })
    
    # Ajouter les transformers de manière dynamique
    if hasattr(model_transfer, 'residu_transformers'):
        num_transformer_blocks = len(model_transfer.residu_transformers)
        half_point = num_transformer_blocks // 2
        
        # Transformers inférieurs (première moitié)
        lower_transformers = []
        for i in range(half_point):
            for param in model_transfer.residu_transformers[i].parameters():
                lower_transformers.append(param)
        
        # Transformers supérieurs (seconde moitié)
        upper_transformers = []
        for i in range(half_point, num_transformer_blocks):
            for param in model_transfer.residu_transformers[i].parameters():
                upper_transformers.append(param)
        
        if lower_transformers:
            param_groups.append({
                'params': lower_transformers,
                'lr': initial_lr * 0.2  # Taux bas pour les transformers inférieurs
            })
        
        if upper_transformers:
            param_groups.append({
                'params': upper_transformers,
                'lr': initial_lr * 0.4  # Taux moyen pour les transformers supérieurs
            })
    
    optimizer_phase3 = torch.optim.AdamW(param_groups, weight_decay=3e-5)
    
    # Scheduler avec warm restart pour la phase finale
    scheduler_phase3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_phase3, T_0=2, T_mult=2, eta_min=initial_lr/50
    )
    
    # Entraînement phase 3 - le reste des époques
    _train_phase(model_transfer, dataloader, optimizer_phase3, device, 
                 epochs=epochs - int(epochs*0.25) - int(epochs*0.35), 
                 phase_name="Phase 3", scheduler=scheduler_phase3)
    
    print("Fine-tuning avancé terminé!")
    
    # Charger le meilleur modèle obtenu
    try:
        model_transfer.load_state_dict(torch.load('best_transfer_model.pth'))
        print("Meilleur modèle de fine-tuning chargé avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement du meilleur modèle: {e}")
    
    return model_transfer

def _train_phase(model, dataloader, optimizer, device, epochs, phase_name, scheduler=None, patience=3):
    """
    Fonction interne pour l'entraînement d'une phase de fine-tuning
    """
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss = float('inf')
    best_epoch = 0
    no_improve_counter = 0
    
    # Utiliser du mixup pour la régularisation si la phase > 1
    use_mixup = "Phase 1" not in phase_name
    mixup_alpha = 0.2  # Paramètre de la distribution Beta pour le mixup
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (consigne, reponse, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Appliquer mixup si activé
            if use_mixup and np.random.random() < 0.5:
                # Générer le paramètre lambda de mixup
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                
                # Créer des index de permutation pour le mixup
                batch_size = consigne.size(0)
                index = torch.randperm(batch_size).to(device)
                
                # Mixup des données d'entrée
                mixed_consigne = lam * consigne + (1 - lam) * consigne[index]
                mixed_reponse = lam * reponse + (1 - lam) * reponse[index]
                
                # Forward pass avec les données mixées
                outputs = model(mixed_consigne, mixed_reponse)
                
                # Calculer la perte avec mixup (combinaison des deux étiquettes)
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
            else:
                # Forward pass normal
                outputs = model(consigne, reponse)
                loss = criterion(outputs, labels)
            
            # Backward et optimisation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            
            # Mettre à jour le scheduler si fourni
            if scheduler is not None:
                scheduler.step()
                
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Afficher la progression
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
                print(f"  {phase_name} - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.0*correct/total:.2f}%")
        
        # Statistiques d'époque
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{phase_name} - Epoch {epoch+1}/{epochs} terminée | "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Sauvegarder le meilleur modèle
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_transfer_model.pth')
            no_improve_counter = 0
            print(f"  → Meilleur modèle sauvegardé (loss: {best_loss:.4f})")
        else:
            no_improve_counter += 1
        
        # Early stopping
        if no_improve_counter >= patience:
            print(f"  → Early stopping après {epoch+1} époques (pas d'amélioration depuis {patience} époques)")
            break
    
    print(f"{phase_name} terminée. Meilleure loss: {best_loss:.4f} à l'epoch {best_epoch+1}")

if __name__ == "__main__":
    # Récupération des données structurées en consignes et réponses
    X_train_consigne, X_train_reponse = X_train
    X_test_consigne, X_test_reponse = X_test
    
    # Paramètres du modèle
    consigne_dim = X_train_consigne.shape[2] if len(X_train_consigne.shape) > 2 else X_train_consigne.shape[1]
    reponse_dim = X_train_reponse.shape[2] if len(X_train_reponse.shape) > 2 else X_train_reponse.shape[1]
    model_dim = 64  # Dimension du modèle
    num_classes = len(train_label_dict)  # Nombre de classes
    
    print(f"Dimension d'entrée consigne: {consigne_dim}")
    print(f"Dimension d'entrée réponse: {reponse_dim}")
    print(f"Dimension du résidu: {consigne_dim}")  # Le résidu a la même dimension que la consigne
    print(f"Nombre de classes: {num_classes}")
    print(f"Forme des données consigne: {X_train_consigne.shape}")
    print(f"Forme des données réponse: {X_train_reponse.shape}")
    
    # Chemin du fichier de modèle entraîné pour le modèle résidu
    model_residu_path = 'gtn_residu_model.pth'
    transfer_model_path = 'gtn_residu_model_transfer_learning.pth'
    
    # Création du modèle utilisant uniquement le résidu
    model_residu = GTNResidu(consigne_dim, reponse_dim, model_dim, num_classes, nhead=4, num_layers=2)
    
    # Vérifier si le modèle entraîné existe déjà
    if os.path.exists(model_residu_path):
        print(f"Modèle résidu déjà entraîné trouvé à {model_residu_path}. Chargement du modèle...")
        try:
            # Essayer de charger le modèle directement
            model_residu.load_state_dict(torch.load(model_residu_path))
            model_residu = model_residu.to(device)
        except RuntimeError as e:
            print(f"Erreur lors du chargement du modèle sur {device}: {e}")
            print("Tentative de chargement sur CPU puis transfert vers le périphérique...")
            # Charger d'abord sur CPU puis transférer vers MPS
            model_residu.load_state_dict(torch.load(model_residu_path, map_location='cpu'))
            model_residu = model_residu.to(device)
        
        # Variables pour afficher les graphiques (valeurs fictives car nous n'entraînons pas)
        train_losses = []
        val_losses = []
        val_accuracies = []
    else:
        print("Aucun modèle résidu pré-entraîné trouvé. Lancement de l'entraînement...")
        
        # Division en ensembles d'entraînement (80%) et de validation (20%)
        print("Division des données en ensembles d'entraînement (80%) et de validation (20%)...")
        val_size = int(VALIDATION_SPLIT_RATIO * len(X_train_consigne))
        indices = np.arange(len(X_train_consigne))
        np.random.shuffle(indices)  # Mélanger les indices pour une division aléatoire
        train_indices, val_indices = indices[val_size:], indices[:val_size]
        
        # Séparation des ensembles d'entraînement et de validation
        X_train_consigne_split = X_train_consigne[train_indices]
        X_train_reponse_split = X_train_reponse[train_indices]
        y_train_split = y_train[train_indices]
        
        X_val_consigne = X_train_consigne[val_indices]
        X_val_reponse = X_train_reponse[val_indices]
        y_val = y_train[val_indices]
        
        print(f"Ensemble d'entraînement: {len(y_train_split)} échantillons")
        print(f"Ensemble de validation: {len(y_val)} échantillons")
        
        # Appliquer l'augmentation de données uniquement sur l'ensemble d'entraînement
        print("Application de l'augmentation de données sur l'ensemble d'entraînement...")
        X_train_consigne_aug, X_train_reponse_aug, y_train_aug = apply_data_augmentation(
            X_train_consigne_split, X_train_reponse_split, y_train_split, 
            noise_level=NOISE_LEVEL, 
            num_augmentations=NUM_AUGMENTATIONS
        )
        
        print(f"Ensemble d'entraînement après augmentation: {len(y_train_aug)} échantillons")
        
        # Entraînement du modèle résidu sur les données augmentées
        print("Entraînement du modèle utilisant uniquement le résidu (différence consigne-réponse)...")
        model_residu, train_losses, val_losses, val_accuracies = train_model_3d(
            model_residu, 
            X_train_consigne_aug, X_train_reponse_aug, y_train_aug,
            X_val_consigne, X_val_reponse, y_val,
            epochs=30, batch_size=32, learning_rate=0.001
        )
        
        # Sauvegarde du modèle résidu
        torch.save(model_residu.state_dict(), model_residu_path)
        print("Modèle résidu entraîné et sauvegardé avec succès!")
    
    # Conversion du dictionnaire d'étiquettes pour l'affichage
    class_names = [None] * len(test_label_dict)
    for name, idx in test_label_dict.items():
        class_names[idx] = name
    
    # Calcul de l'accuracy sur l'ensemble de validation
    print("\nCalcul de l'accuracy sur l'ensemble de validation complet...")
    val_accuracy = calculate_accuracy_3d(model_residu, X_val_consigne, X_val_reponse, y_val)
    print(f"Accuracy sur l'ensemble de validation: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Évaluation initiale du modèle sur l'ensemble de test complet
    print("\nÉvaluation du modèle initial sur l'ensemble de test complet...")
    initial_test_accuracy = calculate_accuracy_3d(model_residu, X_test_consigne, X_test_reponse, y_test)
    print(f"Accuracy initiale sur les données de test: {initial_test_accuracy:.4f} ({initial_test_accuracy*100:.2f}%)")
    
    # Division des données de test en deux parties: une pour le fine-tuning et une pour l'évaluation
    print("\nDivision des données de test pour le transfer learning (50% / 50%)...")
    (X_finetune_consigne, X_finetune_reponse, y_finetune), (X_eval_consigne, X_eval_reponse, y_eval) = \
        split_test_data_for_transfer_learning(X_test_consigne, X_test_reponse, y_test, split_ratio=0.5)
    
    # Effectuer le transfer learning avec augmentation de données
    print("\nApplication du transfer learning avec augmentation de données...")
    model_transfer = transfer_learning_with_advanced_finetune(
        model_residu, 
        X_finetune_consigne, X_finetune_reponse, y_finetune,
        epochs=15, batch_size=16, initial_lr=1e-4,
        noise_level=0.01, num_augmentations=1
    )
    
    # Sauvegarde du modèle avec transfer learning
    torch.save(model_transfer.state_dict(), transfer_model_path)
    print(f"Modèle avec transfer learning sauvegardé à {transfer_model_path}")
    
    # Évaluation du modèle original sur l'ensemble de validation
    print("\nÉvaluation du modèle original sur l'ensemble de validation...")
    original_val_accuracy = calculate_accuracy_3d(model_residu, X_val_consigne, X_val_reponse, y_val)
    print(f"Accuracy du modèle original sur l'ensemble de validation: {original_val_accuracy:.4f} ({original_val_accuracy*100:.2f}%)")
    
    # Évaluation du modèle avec transfer learning sur l'ensemble de validation
    print("\nÉvaluation du modèle avec transfer learning sur l'ensemble de validation...")
    transfer_val_accuracy = calculate_accuracy_3d(model_transfer, X_val_consigne, X_val_reponse, y_val)
    print(f"Accuracy du modèle avec transfer learning sur l'ensemble de validation: {transfer_val_accuracy:.4f} ({transfer_val_accuracy*100:.2f}%)")
    
    # Évaluation du modèle avec transfer learning sur la deuxième partie des données de test (évaluation)
    print("\nÉvaluation du modèle avec transfer learning sur la partie d'évaluation des données de test...")
    transfer_test_accuracy = calculate_accuracy_3d(model_transfer, X_eval_consigne, X_eval_reponse, y_eval)
    print(f"Accuracy après transfer learning sur l'ensemble de test d'évaluation: {transfer_test_accuracy:.4f} ({transfer_test_accuracy*100:.2f}%)")
    
    # Pour comparaison, évaluer le modèle original sur la même partie d'évaluation du test
    print("\nÉvaluation du modèle original sur la même partie d'évaluation pour comparaison...")
    original_test_eval_accuracy = calculate_accuracy_3d(model_residu, X_eval_consigne, X_eval_reponse, y_eval)
    print(f"Accuracy du modèle original sur la partie d'évaluation du test: {original_test_eval_accuracy:.4f} ({original_test_eval_accuracy*100:.2f}%)")
    
    # Matrice de confusion pour le modèle original sur la partie d'évaluation
    print("Calcul de la matrice de confusion pour le modèle original sur la partie d'évaluation...")
    cm_original_eval, original_eval_accuracy_from_cm = plot_confusion_matrix_3d(
        model_residu, X_eval_consigne, X_eval_reponse, y_eval, class_names
    )
    
    # Matrice de confusion pour le modèle avec transfer learning
    print("Calcul de la matrice de confusion pour le modèle avec transfer learning...")
    cm_transfer, transfer_accuracy_from_cm = plot_confusion_matrix_3d(
        model_transfer, X_eval_consigne, X_eval_reponse, y_eval, class_names
    )
    
    # Affichage des résultats comparatifs
    print("\nRésumé comparatif des performances:")
    print("------------------------------------")
    print(f"Modèle original - accuracy sur l'ensemble de validation: {original_val_accuracy:.4f} ({original_val_accuracy*100:.2f}%)")
    print(f"Modèle avec transfer learning - accuracy sur l'ensemble de validation: {transfer_val_accuracy:.4f} ({transfer_val_accuracy*100:.2f}%)")
    print(f"Modèle original - accuracy sur la partie d'évaluation du test: {original_test_eval_accuracy:.4f} ({original_test_eval_accuracy*100:.2f}%)")
    print(f"Modèle avec transfer learning - accuracy sur la partie d'évaluation du test: {transfer_test_accuracy:.4f} ({transfer_test_accuracy*100:.2f}%)")
    
    # Calculer l'amélioration relative sur l'ensemble de validation
    val_improvement = ((transfer_val_accuracy - original_val_accuracy) / original_val_accuracy) * 100
    print(f"\nAmélioration relative du transfer learning sur la validation: {val_improvement:.2f}%")
    
    # Calculer l'amélioration relative sur l'ensemble de test d'évaluation
    test_improvement = ((transfer_test_accuracy - original_test_eval_accuracy) / original_test_eval_accuracy) * 100
    print(f"Amélioration relative du transfer learning sur le test: {test_improvement:.2f}%")
    
    # Enregistrer les résultats dans un fichier
    with open('results_transfer_learning.txt', 'w') as f:
        f.write("Résumé des performances avec Transfer Learning\n")
        f.write("===========================================\n\n")
        f.write(f"Date d'évaluation: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dimensions des données:\n")
        f.write(f"  - Training: {len(y_train_aug)} échantillons (après augmentation)\n")
        f.write(f"  - Validation: {len(y_val)} échantillons\n")
        f.write(f"  - Transfer Learning: {len(y_finetune)} échantillons\n") 
        f.write(f"  - Évaluation Test: {len(y_eval)} échantillons\n\n")
        f.write(f"Performances:\n")
        f.write(f"  - Modèle original sur validation: {original_val_accuracy:.4f} ({original_val_accuracy*100:.2f}%)\n")
        f.write(f"  - Modèle avec transfer learning sur validation: {transfer_val_accuracy:.4f} ({transfer_val_accuracy*100:.2f}%)\n")
        f.write(f"  - Modèle original sur test d'évaluation: {original_test_eval_accuracy:.4f} ({original_test_eval_accuracy*100:.2f}%)\n")
        f.write(f"  - Modèle avec transfer learning sur test d'évaluation: {transfer_test_accuracy:.4f} ({transfer_test_accuracy*100:.2f}%)\n\n")
        f.write(f"Amélioration relative sur validation: {val_improvement:.2f}%\n")
        f.write(f"Amélioration relative sur test: {test_improvement:.2f}%\n")
    
    # Visualisation optionnelle des matrices de confusion
    try:
        # Visualiser les matrices de confusion côte à côte
        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_original_eval, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matrice de confusion - Modèle Original\nAccuracy: {original_test_eval_accuracy:.4f}')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_transfer, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matrice de confusion - Après Transfer Learning\nAccuracy: {transfer_test_accuracy:.4f}')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        
        plt.tight_layout()
        plt.savefig('comparison_confusion_matrices.png')
        print("Matrices de confusion comparatives sauvegardées dans 'comparison_confusion_matrices.png'")
        
    except Exception as e:
        print(f"Erreur lors de la visualisation des matrices de confusion: {e}")
        print("Passez à la visualisation interactivement dans un notebook Jupyter pour plus de détails visuels.")
