import torch
import torch.nn as nn
import pandas as pandas
import torch.optim as optim
from models.neural_network import NeuralNetworkModel
from datasets.TrafficViolationDataset import TrafficViolationDataset
import logging
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def train_model():
    dataset = TrafficViolationDataset("./data/Indian_Traffic_Violations.csv")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Use the actual dimensions from the dataset
    input_dim = dataset.input_dim   # This will be 9437
    output_dim = dataset.output_dim # This will match your encoded labels
    model = NeuralNetworkModel(input_dim=input_dim, output_dim=output_dim)
   
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100  
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_loss = 0
        num_batches = 0

        for batch_features, batch_labels in train_loader:
            # Forward pass
            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        
    os.makedirs('models', exist_ok=True)

    model_path = os.path.join('models', 'traffic_violation_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, model_path)

    print("Model saved!")
    return model



if __name__ == "__main__":
    train_model()