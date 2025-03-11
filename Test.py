import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io  # For loading .mat files

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler

# =============================================================================
# 0. Conversion of .mat files to .csv files
# =============================================================================

def process_directory(directory):
    """
    Recursively converts all .mat files in the given directory (and its subdirectories)
    to .csv files.
    Files that do not contain the key 'trajCmds' (and/or 'trajResps') are skipped.
    """
    items = os.listdir(directory)
    for item in items:
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path) and item != '.git':
            process_directory(full_path)
        else:
            file_name, file_extension = os.path.splitext(full_path)
            if file_extension.lower() == '.mat':
                try:
                    data = scipy.io.loadmat(full_path)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")
                    continue

                # Check if the expected keys are present
                if 'trajCmds' not in data or 'trajResps' not in data:
                    print(f"File {full_path} does not contain 'trajCmds' or 'trajResps'. Skipping file.")
                    continue

                # Proceed with conversion if keys exist
                var_traj_comm = data['trajCmds']
                var_traj_responses = data['trajResps']
                df_traj_comm = pd.DataFrame(var_traj_comm)
                df_traj_responses = pd.DataFrame(var_traj_responses)
                df_merged = pd.concat([df_traj_comm, df_traj_responses], axis=1)
                csv_filename = file_name + '.csv'
                df_merged.to_csv(csv_filename, index=False, header=False)
                print(f"Converted {full_path} to {csv_filename}")


# =============================================================================
# 1. Functions for loading and preprocessing the data
# =============================================================================

def load_dataset(data_dir, feature_mode='original'):
    """
    Loads all CSV files from the directory data_dir.

    Expected CSV format (from the original conversion):
      - The CSV file contains the following columns (without header):
            Column 0: Timestamps,
            Column 1: Motor1Cmd,
            Column 2: Motor2Cmd,
            Column 3: Motor3Cmd,
            Column 4: Motor4Cmd,
            Column 5: Motor5Cmd,
            Column 6: DesiredTrajectory-x,
            Column 7: DesiredTrajectory-y,
            Column 8: DesiredTrajectory-z,
            Column 9: RealizedTrajectory-x,
            Column 10: RealizedTrajectory-y,
            Column 11: RealizedTrajectory-z,
            Column 12: (Optional extra info)

    Depending on feature_mode:
      - 'original': uses columns [1, 2, 3, 4, 9, 10, 11] (motor commands and realized trajectory)
      - 'digital': uses columns [6, 7, 8] (desired trajectory) and computes residuals using columns [9, 10, 11]
    If the CSV file has exactly 4 columns, then all columns are used.

    The label is extracted from the file name (the file name should start with the label, e.g., "Healthy_001.csv").
    """
    X, y = [], []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory {data_dir}")
    
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        n_cols = df.shape[1]
        if n_cols == 4:
            features = df.values
        elif n_cols >= 12:
            if feature_mode == 'original':
                features = df.iloc[:, [1, 2, 3, 4, 9, 10, 11]].values
            elif feature_mode == 'digital':
                desired = df.iloc[:, [6, 7, 8]].values
                realized = df.iloc[:, [9, 10, 11]].values
                residual = realized - desired
                features = np.concatenate([desired, residual], axis=1)
            else:
                raise ValueError("feature_mode must be 'original' or 'digital'")
        elif n_cols >= 8:
            features = df.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values
        else:
            print(f"File {file} has an unexpected number of columns ({n_cols}). Skipping file.")
            continue
        
        X.append(features)
        base = os.path.basename(file)
        label = base.split('_')[0]
        y.append(label)
    
    return X, y

def normalize_features(X, scaler=None):
    """
    Normalizes all sequences.
    Concatenates all sequences (along axis 0) to compute the mean and standard deviation.
    If a scaler is provided, it is used for transformation.
    
    Returns:
      - X_norm: list of normalized sequences
      - scaler: the StandardScaler object
    """
    all_data = np.concatenate(X, axis=0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(all_data)
    X_norm = [scaler.transform(x) for x in X]
    return X_norm, scaler

def encode_labels(y):
    """
    Encodes a list of labels (strings) into integers, then one-hot encodes them.
    
    Returns:
      - y_int: array of integer labels
      - y_cat: one-hot encoded array
      - le: the LabelEncoder (to retrieve the mapping)
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    y_cat = tf.keras.utils.to_categorical(y_int)
    return y_int, y_cat, le

# =============================================================================
# 2. Building the LSTM model
# =============================================================================

def build_lstm_model(timesteps, n_features, num_classes):
    """
    Builds an LSTM model with the following layers:
      - A sequence input layer
      - An LSTM layer with 100 units (return_sequences=True)
      - A Dropout layer with rate 0.1
      - A second LSTM layer with 100 units (returns only the last state)
      - A Dense layer with softmax activation to produce num_classes outputs
    """
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, n_features)))
    model.add(Dropout(0.1))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lr_scheduler(epoch, lr):
    """Reduces the learning rate by a factor of 0.1 every 20 epochs."""
    if epoch > 0 and epoch % 20 == 0:
        return lr * 0.1
    return lr

# =============================================================================
# 3. Training and evaluation pipeline
# =============================================================================

if __name__ == '__main__':
    # ----- User-defined parameters -----
    feature_mode = 'original'  # or 'digital'
    
    # Specify folders for training and testing datasets.
    # Change these paths as needed.
    training_folder = "dataset/trainingDatasets/20241017"
    test_folder = "dataset/testDatasets/20241008"  # example test folder
    
    epochs = 30
    batch_size = 32
    random_state = 42

    # ----- Convert .mat files to .csv if needed -----
    print("Converting .mat files to .csv in the training folder...")
    process_directory(training_folder)
    print("Converting .mat files to .csv in the test folder...")
    process_directory(test_folder)
    
    # ----- Load the data -----
    print("Loading training data from:", training_folder)
    X_train_list, y_train_list = load_dataset(training_folder, feature_mode=feature_mode)
    print(f"{len(X_train_list)} training sequences loaded.")
    
    print("Loading test data from:", test_folder)
    X_test_list, y_test_list = load_dataset(test_folder, feature_mode=feature_mode)
    print(f"{len(X_test_list)} test sequences loaded.")
    
    # ----- Normalization -----
    X_train_norm, scaler = normalize_features(X_train_list)
    X_test_norm, _ = normalize_features(X_test_list, scaler=scaler)
    
    # Convert lists to 3D numpy arrays: (n_samples, timesteps, n_features)
    X_train = np.array(X_train_norm)
    X_test = np.array(X_test_norm)
    
    # ----- Encode labels -----
    y_train_int, y_train_cat, le = encode_labels(y_train_list)
    y_test_int, y_test_cat, _ = encode_labels(y_test_list)
    num_classes = y_train_cat.shape[1]
    print("Detected classes:", le.classes_)
    
    # Get sequence dimensions
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"Each sequence has {timesteps} time steps and {n_features} features.")

    # ----- Build the model -----
    model = build_lstm_model(timesteps, n_features, num_classes)
    model.summary()
    
    lr_callback = LearningRateScheduler(lr_scheduler)
    
    # ----- Training -----
    print("Starting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_callback],
        shuffle=True
    )
    
    # ----- Evaluation -----
    print("Evaluating on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    
    accuracy = accuracy_score(y_test_int, y_pred_class)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test_int, y_pred_class, target_names=le.classes_))
    
    cm = confusion_matrix(y_test_int, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.show()
