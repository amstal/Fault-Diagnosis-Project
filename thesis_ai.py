import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


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

# Define a function to normalize features
def normalize_features(X):
    mean = X.mean(dim=(0, 1), keepdim=True)  # Calculate the mean of each feature
    std = X.std(dim=(0, 1), keepdim=True)  # Calculate the standard deviation of each feature
    std[std == 0] = 1e-8  # Prevent division by zero
    X_normalized = (X - mean) / std  # Normalize the features
    return X_normalized

# Define a function to calculate the model's accuracy and display metrics
def print_confusion_matrix(y_test, predicted, categories):
    precision = precision_score(y_test.cpu(), predicted.cpu(), average='weighted')
    recall = recall_score(y_test.cpu(), predicted.cpu(), average='weighted')
    f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')
    print(f'Overall Test Accuracy: {precision * 100:.2f}%')
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_test.cpu(), predicted.cpu())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print precision, recall, and F1 score for each category
    for i, category in enumerate(categories):
        precision = precision_score(y_test.cpu(), predicted.cpu(), labels=[i], average='weighted')
        recall = recall_score(y_test.cpu(), predicted.cpu(), labels=[i], average='weighted')
        f1 = f1_score(y_test.cpu(), predicted.cpu(), labels=[i], average='weighted')
        
        print(f'Category: {category}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# Define a function to convert .mat files into PyTorch tensors
def transfer_tensor(mat, X_name, Y_name, mean=None, std=None):
    # Extract nested arrays
    data_X = mat[X_name][0]  # Extract the nested objec²t array for X
    data_Y = mat[Y_name][0]
    
    # Extract X into a tensor
    data_X_combined = np.array([data_X[i] for i in range(len(data_X))])
    data_tensor_X = torch.tensor(data_X_combined, dtype=torch.float32).to(device)  # Move to GPU
    
    # Create a new tensor to store processed data
    data_tensor_X_with_residual = data_tensor_X.clone()  # Clone to preserve original data
    
    # Compute residuals and replace the last three features with them
    residual = data_tensor_X[:, :, :3] - data_tensor_X[:, :, 3:6]  # Compute residuals
    data_tensor_X_with_residual[:, :, 3:6] = residual  # Replace the last three features with residuals
    
    data_tensor_X = data_tensor_X_with_residual
    
    # Normalize features
    if mean is None or std is None:
        # If mean and standard deviation are not provided, calculate them
        mean = data_tensor_X.mean(dim=(0, 1), keepdim=True)
        std = data_tensor_X.std(dim=(0, 1), keepdim=True)
    
    data_tensor_X = (data_tensor_X - mean) / std  # Normalize features
    
    # Extract Y into a tensor
    data_Y_combined = np.array([data_Y[i] for i in range(len(data_Y))])
    data_Y_combined = data_Y_combined.flatten()
    
    # Create a dictionary mapping each category to its corresponding index
    category_to_index = {category: index for index, category in enumerate(categories)}
    
    data_Y_numeric = np.array([category_to_index[category] for category in data_Y_combined])
    
    # Convert NumPy array to PyTorch tensor
    data_tensor_Y = torch.tensor(data_Y_numeric, dtype=torch.int64).to(device)  # Move to GPU
    
    return data_tensor_X, data_tensor_Y, mean, std  # Return mean and standard deviation

# Check if CUDA is available for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load .mat files as dictionaries
mat = scipy.io.loadmat('models/my_dataset_train.mat')
mat_test = scipy.io.loadmat('models/my_dataset_test.mat')

# Convert training and testing data into tensors
data_tensor_X, data_tensor_Y, simulation_mean, simulation_std = transfer_tensor(mat, 'X_array', 'y_array')
data_tensor_X_real, data_tensor_Y_real, temp_mean, temp_std = transfer_tensor(mat_test, 'X_test_array', 'y_test_array', mean=simulation_mean, std=simulation_std)

print("tensor X:", data_tensor_X.shape)
print("tensor Y:", data_tensor_Y.shape)
print("tensor X real:", data_tensor_X_real.shape)
print("tensor Y real:", data_tensor_Y_real.shape)

# Training and validation set creation
training_ratio = 0.9
n_dataset = data_tensor_X.shape[0]
training_size = int(training_ratio * n_dataset)

# Randomly shuffle indices
indices = np.arange(n_dataset)
np.random.shuffle(indices)

# Split into training and test sets
train_indices = indices[:training_size]
test_indices = indices[training_size:]

X_train = data_tensor_X[train_indices]
X_test = data_tensor_X[test_indices]
y_train = data_tensor_Y[train_indices]
y_test = data_tensor_Y[test_indices]

# Split real-world data test set
X_test_real = data_tensor_X_real
y_test_real = data_tensor_Y_real

# Import model from models.py
from models import *

# Model parameter configuration
input_size = 6  # Based on data feature dimensions
num_classes = len(categories)
print(num_classes)

# Initialize model and move to GPU
model = LSTMModel(input_size = input_size, hidden_size = 100, num_classes = num_classes).to(device)
# model = CNNModel(input_size = input_size, num_classes=num_classes).to(device)
# model = ComplexCNN(input_size = input_size, num_classes=num_classes, num_channels=128).to(device)
# model = TransformerModel(input_size = input_size, hidden_size=32, num_classes=num_classes).to(device)
# model = TCN(input_size = input_size, output_size=num_classes, num_channels=[128,128,128]).to(device)
# model = MyTCN(input_size = input_size, num_classes=num_classes, num_channels=64).to(device)
#model = CNNModelDANN(input_size = input_size, num_classes=num_classes).to(device)
# model = AlexNetDANN(input_size = input_size, num_classes=num_classes).to(device)

# Lists to store loss and accuracy values
losses_train = []
losses_test = []
losses_test_real = []
losses_train_source = []
losses_train_target = []
acc_train = []
acc_test = []
acc_test_real = []
acc_train_source = []
acc_train_target = []

# Training settings
# During different loss function tests, using CrossEntropyLoss/not using softmax causes test set loss to increase, using softmax works normally
# criterion = nn.CrossEntropyLoss()

def criterion(logits, labels):
    # Use log_softmax to calculate log probabilities
    # logits = F.softmax(logits, dim=1)  # Simulate adding a softmax layer in network structure. Adding softmax before/after logsoftmax, or using softmax alone works normally, otherwise test set loss increases
    log_probs = F.log_softmax(logits, dim=1)
    # log_probs = F.softmax(logits, dim=1)
    # log_probs = F.softmax(log_probs, dim=1)


    # Use NLLLoss to calculate negative log likelihood loss
    loss_fn = nn.NLLLoss()
    loss = loss_fn(log_probs, labels)
    
    return loss

# def criterion(logits, labels):
#     labels = labels.to("cpu")
#     logits = logits.to("cpu")
#     # Use sigmoid to calculate probabilities
#     probs = torch.sigmoid(logits)
    
#     # Use Binary Cross Entropy to calculate loss
#     loss_fn = nn.BCELoss()  # Suitable for multi-label classification
#     one_hot_labels = torch.eye(num_classes)[labels]
#     loss = loss_fn(probs, one_hot_labels)
    
#     return loss

# parameters
mini_batch_size = 32
num_epochs = 250
learning_rate = 0.001
factor=0.1
patience=100
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

# Train model
for epoch in range(num_epochs+1):
    # Compute alpha for Gradient Reversal Layer
    p = float(epoch) / num_epochs
    if p > 1.:
        p = 1.
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    temp_loss = 0
    temp_loss_source_domain = 0
    temp_loss_target_domain = 0
    temp_acc = 0
    temp_acc_source_domain = 0
    temp_acc_target_domain = 0
    temp_loss_all = []
    temp_loss_source_domain_all = []
    temp_loss_target_domain_all = []
    temp_acc_all = []
    temp_acc_source_domain_all = []
    temp_acc_target_domain_all = []
    model.train()  # Set model to training mode
    for i in range(0, len(X_train), mini_batch_size):
        inputs = X_train[i:i + mini_batch_size].view(-1, 1000, input_size)  # Model input format
        labels = y_train[i:i + mini_batch_size]
        inputs_real = X_test_real.view(-1, 1000, input_size)  # Model input format

        source_domain_label = torch.zeros(len(inputs)).long().to(device)
        target_domain_label = torch.ones(len(inputs_real)).long().to(device)

        # Forward propagation
        source_domain_outputs = model(inputs) 

        # classify loss source domain
        _, predicted_train = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # domain loss source domain
        # print("shape sdo and sdl:", source_domain_outputs.shape, source_domain_label.shape)
        loss_source_domain = criterion(source_domain_outputs, source_domain_label)

        # forward pass for target domain
        _, target_domain_outputs = model(inputs_real)
        loss_target_domain = criterion(target_domain_outputs, target_domain_label)

        with torch.no_grad():
            temp_loss_all.append(loss.item())
            temp_loss_source_domain_all.append(loss_source_domain.item())
            temp_loss_target_domain_all.append(loss_target_domain.item())
            temp_acc_all.append(accuracy_score(labels.cpu(), predicted_train.cpu()))
            temp_acc_source_domain_all.append(accuracy_score(source_domain_label.cpu(), torch.argmax(source_domain_outputs, dim=1).cpu()))
            temp_acc_target_domain_all.append(accuracy_score(target_domain_label.cpu(), torch.argmax(target_domain_outputs, dim=1).cpu()))

        # Backpropagation and optimization
        loss_all = loss + loss_source_domain + loss_target_domain
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

    # Print test accuracy every epoch
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        temp_loss = np.mean(temp_loss_all)
        temp_loss_source_domain = np.mean(temp_loss_source_domain_all)
        temp_loss_target_domain = np.mean(temp_loss_target_domain_all)
        temp_acc = np.mean(temp_acc_all)
        temp_acc_source_domain = np.mean(temp_acc_source_domain_all)
        temp_acc_target_domain = np.mean(temp_acc_target_domain_all)


        test_outputs, _ = model(X_test.view(-1, 1000, input_size))  # Model input format
        _, predicted = torch.max(test_outputs.data, 1)
        loss = criterion(test_outputs, y_test)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())

        test_outputs_real, _ = model(X_test_real.view(-1, 1000, input_size))
        _, predicted_real = torch.max(test_outputs_real.data, 1)
        loss_real = criterion(test_outputs_real, y_test_real)
        accuracy_real = accuracy_score(y_test_real.cpu(), predicted_real.cpu())
        
        scheduler.step(loss.item())

        # Record loss and accuracy
        losses_train.append(temp_loss)
        losses_test.append(loss.item())
        losses_test_real.append(loss_real.item())
        losses_train_source.append(temp_loss_source_domain)
        losses_train_target.append(temp_loss_target_domain)
        acc_train.append(temp_acc)
        acc_test.append(accuracy)
        acc_test_real.append(accuracy_real)
        acc_train_source.append(temp_acc_source_domain)
        acc_train_target.append(temp_acc_target_domain)


        # Update and display plots
        # clear_output(wait=True)
        if epoch%50==0 and epoch!=0:
            plt.figure(figsize=(8,6))
            plt.plot(losses_test, label="losses_test", color='blue')
            plt.plot(losses_train, label="losses_train", color='orange')
            plt.plot(losses_test_real, label="losses_test_real", color='green')
            plt.plot(losses_train_source, label="losses_train_source", color='red')
            plt.plot(losses_train_target, label="losses_train_target", color='purple')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8,6))
            plt.plot(acc_test, label="acc_test", color='blue')
            plt.plot(acc_train, label="acc_train", color='orange')
            plt.plot(acc_test_real, label="acc_test_real", color='green')
            plt.plot(acc_train_source, label="acc_train_source", color='red')
            plt.plot(acc_train_target, label="acc_train_target", color='purple')
            plt.title('Acc During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Acc')
            plt.legend()
            plt.grid(True)
            plt.show()



        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch + 1}/{num_epochs}], lr: {current_lr}, '
            f'alpha: {alpha:.4f}, '
            f'Train Accuracy: {temp_acc * 100:.2f}%, Simu_val_Accuracy: {accuracy * 100:.2f}%, REAL_ACCURACY: {accuracy_real * 100:.2f}%  , '
            f'Source Accuracy: {temp_acc_source_domain * 100:.2f}%, Target Accuracy: {temp_acc_target_domain * 100:.2f}%, '
            f'Source Loss: {temp_loss_source_domain:.4f}, Target Loss: {temp_loss_target_domain:.4f}, '
            f'Train_Loss: {temp_loss:.4f}, Simu_val_Loss: {loss.item():.4f}, Real_Loss: {loss_real.item():.4f}')



# confusion matrix for simulation validation and real test set
print_confusion_matrix(y_test, predicted, categories)
print_confusion_matrix(y_test_real, predicted_real, categories)

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    test_outputs, _ = model(X_test_real.view(-1, 1000, input_size))  # Model input format
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = accuracy_score(y_test_real.cpu(), predicted.cpu())
    print(f'Test Real Accuracy: {accuracy * 100:.2f}%')
