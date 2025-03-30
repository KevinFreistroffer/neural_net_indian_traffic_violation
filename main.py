import torch
import torch.nn as nn
import pandas as pandas
import torch.optim as optim
from models.neural_network import NeuralNetworkModel
from datasets.TrafficViolationDataset import TrafficViolationDataset
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

os.environ["OPENBLAS_NUM_THREADS"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def train_model():
    # gets the features and labels columns
    # gets the features headers and labels headers
    # gets the features and labels data. processes the data converting them to numerical data. normalizes them.
    # returns numpy arrays
    dataset = TrafficViolationDataset("./data/Indian_Traffic_Violations.csv")
    
    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Use the actual dimensions from the dataset
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    model = NeuralNetworkModel(input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)  # Move model to the selected device
   
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    
    # Training loop
    num_epochs = 100  
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_features, batch_labels in train_loader:
            # Move data to the selected device
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        total_test_loss = 0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                total_test_loss += loss.item()
                num_test_batches += 1
        
        avg_test_loss = total_test_loss / num_test_batches
        test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    os.makedirs('models', exist_ok=True)

    model_path = os.path.join('models', 'traffic_violation_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_test_loss
    }, model_path)

    print("Model saved!")
    
    # Visualize training and testing losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/training_testing_loss.png')
    
    # Display the plot
    plt.show()
    
    # Evaluate final model performance
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    # Calculate R² score for each output dimension
    r2_scores = []
    for i in range(output_dim):
        r2 = r2_score(all_labels[:, i], all_outputs[:, i])
        r2_scores.append(r2)
    
    # Plot R² scores for each output dimension
    plt.figure(figsize=(12, 6))
    plt.bar(range(output_dim), r2_scores)
    plt.xlabel('Output Dimension')
    plt.ylabel('R² Score')
    plt.title('R² Score for Each Output Dimension')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/r2_scores.png')
    
    # Display the plot
    plt.show()
    
    print(f"Average R² score: {np.mean(r2_scores):.4f}")
    print("Model evaluation complete!")
    
    return model

if __name__ == "__main__":
    train_model()