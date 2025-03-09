import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ---------------------------------------------------------------
# 🟢 Step 1: Load the Dataset from NPZ files
# ---------------------------------------------------------------
print("\n🔹 Loading data from NPZ files...")

# Load concatenated embeddings
x_data = np.load("X_flat.npz", allow_pickle=True)
x_keys = list(x_data.keys())
print(f"✅ Keys in X_flat.npz: {x_keys}")

# Load target values from y_30k.npz
y_data = np.load("y_30k.npz", allow_pickle=True)
y_keys = list(y_data.keys())
print(f"✅ Keys in y_30k.npz: {y_keys}")

# Extract data
X = x_data[x_keys[0]]
y = y_data[y_keys[0]]

# Debugging: Check raw shapes and types
print(f"🔍 Raw X type: {type(X)}, shape: {getattr(X, 'shape', 'Unknown')}")
print(f"🔍 Raw y type: {type(y)}, shape: {getattr(y, 'shape', 'Unknown')}, first values: {y[:5]}")

# Ensure `y` is a proper NumPy array
if np.isscalar(y):  
    print("⚠️ Warning: `y` is a scalar. Converting to NumPy array.")
    y = np.array([y])

# Ensure `y` is a column vector (N,1) if it's 1D
if y.ndim == 1:  
    print("🔄 Reshaping `y` to (N,1)")
    y = y.reshape(-1, 1)

# Final shape verification
print(f"✅ Fixed X shape: {X.shape}")
print(f"✅ Fixed y shape: {y.shape}")

# Ensure `X` and `y` have the same number of rows
assert X.shape[0] == y.shape[0], f"❌ Error: Mismatch in samples! X: {X.shape[0]}, y: {y.shape[0]}"

# ---------------------------------------------------------------
# 🟢 Step 2: Split Dataset into Train and Test Sets
# ---------------------------------------------------------------
print("\n🔹 Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training set: {X_train.shape}, {y_train.shape}")
print(f"✅ Test set: {X_test.shape}, {y_test.shape}")

# ---------------------------------------------------------------
# 🟢 Step 3: Train Random Forest Model (Baseline)
# ---------------------------------------------------------------
print("\n🔹 Training Random Forest Model...")

rf_model = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train.ravel())  # `ravel()` ensures `y` is 1D for RandomForest

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
print(f"✅ Random Forest MSE: {rf_mse}")

# Save Random Forest model
joblib.dump(rf_model, "ic50_rf_model.pkl")

# ---------------------------------------------------------------
# 🟢 Step 4: Train Fully Connected Neural Network (FCNN) on GPU
# ---------------------------------------------------------------

# ✅ Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")

# Convert data to PyTorch tensors and move to GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Define Neural Network Model
class IC50Predictor(nn.Module):
    def __init__(self, input_dim):
        super(IC50Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize Model and move to GPU
input_dim = X_train.shape[1]
model = IC50Predictor(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
print("\n🟢 Starting FCNN Training on GPU...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # Track loss per epoch

    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"📢 Epoch {epoch}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate FCNN model
print("\n🔹 Evaluating FCNN Model on GPU...")
model.eval()
with torch.no_grad():
    y_pred_fcnn = model(X_test_tensor).cpu().numpy()  # Move predictions back to CPU
    fcnn_mse = mean_squared_error(y_test, y_pred_fcnn)
    print(f"✅ FCNN MSE: {fcnn_mse}")

# Save FCNN Model
torch.save(model.state_dict(), "ic50_nn_model_gpu.pth")

# ---------------------------------------------------------------
# 🟢 Step 5: Compare Models & Choose Best One
# ---------------------------------------------------------------
print("\n🔍 Model Performance Comparison:")
print(f"🔹 Random Forest MSE: {rf_mse}")
print(f"🔹 FCNN MSE: {fcnn_mse}")

best_model = "FCNN (GPU)" if fcnn_mse < rf_mse else "Random Forest"
print(f"🎯 Best Model Selected: {best_model}")

print("\n✅ Training and Evaluation Completed Successfully on GPU! 🚀")

