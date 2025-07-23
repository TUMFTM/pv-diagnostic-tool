import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from pathlib import Path


class PollutionMLP(nn.Module):
    def __init__(self, input_size):
        super(PollutionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def train_pollution_model(clusters_dir, model_path, plots_dir=None):
    """Train pollution detection model and save it"""
    
    # Fixed test plants for consistent evaluation
    test_plants = ["gw-0002", "gw-0019", "gw-0025", "gw-0035", "gw-0040", 
                   "gw-0048", "gw-0053", "gw-0057", "gw-0065", "gw-0097"]
    
    # Load cluster results from files
    clusters_path = Path(clusters_dir)
    cluster_results = {}
    
    print("Loading pollution cluster data...")
    for cluster_file in clusters_path.glob("*_clustered.csv"):
        plant_id = cluster_file.stem.replace("_clustered", "")
        try:
            import pandas as pd
            df = pd.read_csv(cluster_file)
            # Convert DataFrame to expected format
            cluster_results[plant_id] = {
                'cluster_labels': df['cluster'].values,
                'features': df.drop(['cluster', 'date'] + [col for col in df.columns if 'cluster_proba_' in col], axis=1, errors='ignore').values
            }
            print(f"  Loaded {plant_id}: {len(df)} samples")
        except Exception as e:
            print(f"  Error loading {plant_id}: {e}")
            continue
    
    if not cluster_results:
        raise ValueError("No cluster results found. Check clustering step.")
    
    # Prepare data with plant-based split
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    for plant_id, data in cluster_results.items():
        cluster_labels = data['cluster_labels']
        features = data['features']
        
        for i, label in enumerate(cluster_labels):
            # Label as polluted if cluster is 1 (representing pollution)
            is_polluted = 1 if label == 1 else 0
            
            if plant_id in test_plants:
                X_test.append(features[i])
                y_test.append(is_polluted)
            else:
                X_train.append(features[i])
                y_train.append(is_polluted)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize model
    model = PollutionMLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with tracking
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(100):
        # Training step
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Track training metrics
        with torch.no_grad():
            train_preds = (outputs > 0.5).float()
            train_acc = (train_preds == y_train_tensor).float().mean()
            train_losses.append(loss.item())
            train_accuracies.append(train_acc.item())
            
            # Validation metrics
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_test_tensor).float().mean()
            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
    
    # Create training plots if plots_dir is provided
    if plots_dir:
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Pollution Model - Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Pollution Model - Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'pollution_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs > 0.5).float()
        test_accuracy = (test_predictions == y_test_tensor).float().mean()
        
        # Convert to numpy for sklearn metrics
        y_test_np = y_test_tensor.cpu().numpy().flatten()
        y_pred_np = test_predictions.cpu().numpy().flatten()
        
        print(f"\n=== Pollution Model Performance ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='binary')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_np, y_pred_np)
        print(f"\nConfusion Matrix:")
        print(f"[[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
        print(f" [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test_np, y_pred_np, target_names=['Clean', 'Polluted'], digits=4))
    
    # Save model and scaler
    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_size': X_train.shape[1]
    }
    torch.save(model_data, model_path)
    print(f"Model saved to: {model_path}")


def load_pollution_model(model_path):
    """Load trained pollution detection model"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model = PollutionMLP(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {
        'model': model,
        'scaler': checkpoint['scaler']
    }


def predict_pollution(model_dict, feature_vectors):
    """Predict pollution levels for given feature vectors"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    predictions = {}
    
    for plant_id, features in feature_vectors.items():
        daily_predictions = {}
        
        for date, row in features.iterrows():
            # Prepare features
            feature_array = row.values.reshape(1, -1)
            feature_scaled = scaler.transform(feature_array)
            feature_tensor = torch.FloatTensor(feature_scaled)
            
            # Predict
            with torch.no_grad():
                prediction = model(feature_tensor)
                pollution_class = int(prediction.item() > 0.5)  # Binary classification
            
            daily_predictions[str(date)] = pollution_class
        
        predictions[plant_id] = daily_predictions
    
    return predictions
