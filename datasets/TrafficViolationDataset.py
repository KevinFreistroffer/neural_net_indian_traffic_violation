import torch
import pandas as pandas
import numpy as np
from torch.utils.data import Dataset
import logging
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

violation_details = [
  "violation_type",
  "fine_amount",
  "date", 
  "time",
  "location"
]

vehicle_information = [
  "type",
  "color", 
  "model_year",
  "registration_state"
]

driver_profile = [
  "age",
  "gender",
  "license_type", 
  "previous_violations"
]

enforcement_details = [
  "officer_id",
  "issuing_agency",
  "towed_status",
  "fine_payment"
]

environmental_factors = [
  "weather_conditions",
  "road_status",
  "speed_limits"
]

feature_headers = [*vehicle_information, *driver_profile, *enforcement_details, *environmental_factors]
label_headers = [*violation_details]

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

        # Convert to numpy arrays after preprocessing because Pytorch requires numpy arrays
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
