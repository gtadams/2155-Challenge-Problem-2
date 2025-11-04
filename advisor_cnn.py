"""
CNN-Based Advisor Prediction
Replace feature engineering + AutoGluon with direct CNN prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class GridDataset(Dataset):
    """Dataset for grid data"""
    def __init__(self, grids, ratings):
        """
        Parameters:
        -----------
        grids : np.ndarray, shape (N, 7, 7)
            Grid layouts with labels 0-4
        ratings : np.ndarray, shape (N,)
            Advisor ratings for each grid
        """
        self.grids = torch.FloatTensor(grids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        # Convert grid to one-hot encoding
        grid = self.grids[idx].long()
        grid_onehot = F.one_hot(grid, num_classes=5).float()  # Shape: (7, 7, 5)
        grid_onehot = grid_onehot.permute(2, 0, 1)  # Shape: (5, 7, 7) for Conv2d
        return grid_onehot, self.ratings[idx]


class AdvisorCNN(nn.Module):
    """
    CNN for predicting advisor ratings from grid layouts.
    
    Architecture:
    - Takes 7x7 grid with 5 labels (one-hot encoded)
    - Uses convolutional layers to capture spatial patterns
    - Outputs single rating prediction
    """
    def __init__(self):
        super(AdvisorCNN, self).__init__()
        
        # Input: (batch, 5, 7, 7) - 5 channels for 5 labels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)  # → (32, 7, 7)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → (64, 7, 7)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # → (64, 7, 7)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # → (64, 1, 1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)  # → (64, 1, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 2, 128)  # 64*2 because we concat avg and max pool
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling (both avg and max)
        x_avg = self.global_avg_pool(x).view(x.size(0), -1)
        x_max = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([x_avg, x_max], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x.squeeze()


def train_advisor_cnn(grids, ratings, advisor_idx, epochs=100, batch_size=32, lr=0.001):
    """
    Train CNN for one advisor.
    
    Parameters:
    -----------
    grids : np.ndarray, shape (N, 7, 7)
        Training grids
    ratings : np.ndarray, shape (N, 4)
        Ratings from all advisors
    advisor_idx : int
        Which advisor to train (0-3)
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    
    Returns:
    --------
    model : AdvisorCNN
        Trained model
    history : dict
        Training history
    """
    print(f"\n{'='*80}")
    print(f"Training CNN for Advisor {advisor_idx}")
    print(f"{'='*80}")
    
    # Get ratings for this advisor
    advisor_ratings = ratings[:, advisor_idx]
    
    # Filter to rated grids
    rated_mask = advisor_ratings > 0
    grids_rated = grids[rated_mask]
    ratings_rated = advisor_ratings[rated_mask]
    
    print(f"Training samples: {len(grids_rated)}")
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        grids_rated, ratings_rated, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = GridDataset(X_train, y_train)
    val_dataset = GridDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AdvisorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for grids_batch, ratings_batch in train_loader:
            grids_batch = grids_batch.to(device)
            ratings_batch = ratings_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(grids_batch)
            loss = criterion(predictions, ratings_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for grids_batch, ratings_batch in val_loader:
                grids_batch = grids_batch.to(device)
                ratings_batch = ratings_batch.to(device)
                
                predictions = model(grids_batch)
                loss = criterion(predictions, ratings_batch)
                
                val_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(ratings_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate R²
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(r2)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val R²={r2:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'advisor_cnn_{advisor_idx}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f'advisor_cnn_{advisor_idx}_best.pt'))
    
    print(f"\nTraining complete!")
    print(f"Best validation R²: {max(history['val_r2']):.4f}")
    
    return model, history


def predict_with_cnn(model, grids, batch_size=32):
    """
    Predict advisor ratings using trained CNN.
    
    Parameters:
    -----------
    model : AdvisorCNN
        Trained model
    grids : np.ndarray, shape (N, 7, 7)
        Grids to predict
    batch_size : int
        Batch size for prediction
    
    Returns:
    --------
    predictions : np.ndarray, shape (N,)
        Predicted ratings
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dataset without ratings (dummy ratings)
    dataset = GridDataset(grids, np.zeros(len(grids)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for grids_batch, _ in loader:
            grids_batch = grids_batch.to(device)
            preds = model(grids_batch)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)


def train_all_advisors_cnn(grids, ratings, epochs=100):
    """
    Train CNNs for all 4 advisors.
    
    Returns:
    --------
    models : list of 4 AdvisorCNN models
    """
    models = []
    
    for advisor_idx in range(4):
        model, history = train_advisor_cnn(
            grids, ratings, advisor_idx, 
            epochs=epochs
        )
        models.append(model)
    
    return models


def predict_all_advisors_cnn(models, grids):
    """
    Predict all 4 advisor ratings using CNNs.
    
    Parameters:
    -----------
    models : list of 4 AdvisorCNN models
    grids : np.ndarray, shape (N, 7, 7)
    
    Returns:
    --------
    predictions : np.ndarray, shape (N, 4)
    """
    predictions = np.zeros((len(grids), 4), dtype=np.float32)
    
    for advisor_idx, model in enumerate(models):
        print(f"Predicting advisor {advisor_idx}...")
        predictions[:, advisor_idx] = predict_with_cnn(model, grids)
    
    return predictions