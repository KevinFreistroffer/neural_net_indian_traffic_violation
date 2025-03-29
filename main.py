import torch
import torch.nn as nn
import pandas as pandas
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset
from data.get_dataset import get_dataset
from data.headers import (
    feature_headers,
    label_headers
)
import logging
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class TrafficViolationDataset(Dataset):
    def __init__(self, file, transform=None):
        # getting the feature data
        self.data = pandas.read_csv(file)
        self.columns = self.data.columns
        self.transform = transform
        self.feature_headers = [
            column for column in self.columns if column.lower() not in feature_headers
        ]
        self.label_headers = [
            column for column in self.columns if column.lower() in label_headers
        ]

        # Get features and labels as DataFrames (not numpy arrays yet)
        features_df = self.data[self.feature_headers]
        labels_df = self.data[self.label_headers]
        
        # Process features
        for column in features_df.columns:
            if features_df[column].dtype == 'object':
                encoded = pandas.get_dummies(features_df[column], prefix=column)
                features_df = features_df.drop(column, axis=1)
                features_df = pandas.concat([features_df, encoded], axis=1)
            elif pandas.api.types.is_datetime64_any_dtype(features_df[column]):
                features_df[column] = features_df[column].astype(np.int64)

        # Process labels
        for column in labels_df.columns:
            if labels_df[column].dtype == 'object':
                encoded = pandas.get_dummies(labels_df[column], prefix=column)
                labels_df = labels_df.drop(column, axis=1)
                labels_df = pandas.concat([labels_df, encoded], axis=1)

        # Convert to numpy arrays after preprocessing
        self.X = self.normalize_features(features_df.astype(float).values)
        self.y = labels_df.astype(float).values
        
        # Store the actual dimensions after preprocessing
        self.input_dim = self.X.shape[1]  # Get actual feature dimension
        self.output_dim = self.y.shape[1]  # Get actual label dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get this row of data ['VLT103999' 'No Helmet' 2133 ... 'Card' 'Yes' nan]
        features = torch.FloatTensor(self.X[index]) # is a row of columns
        # get this row of data ["overloading" "4544 Stree Name" "2023-01-02" etc...]
        labels = torch.FloatTensor(self.y[index]) # is a row of columns
        return features, labels
    
    def normalize_features(self, features):
        mean = np.mean(features, axis=0)
        std=np.std(features, axis=0)
        std = np.where(std == 0, 1, std)
        return (features - mean) / std


# class LinearRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.linear(x)
    
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x =self.relu(self.layer2(x))
        x= self.dropout(x)
        return self.layer3(x)
        # return self.linear(x)

def train_model():
    dataset = TrafficViolationDataset("./data/Indian_Traffic_Violations.csv")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Use the actual dimensions from the dataset
    input_dim = dataset.input_dim   # This will be 9437
    output_dim = dataset.output_dim # This will match your encoded labels
    model = Model(input_dim=input_dim, output_dim=output_dim)
   
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