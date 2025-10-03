# 🚀 Komaneko Training Guide - GPU Setup

## 📋 **Current System Status**

### ✅ **What's Working:**
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) - **DETECTED**
- **XGBoost**: GPU acceleration working (CUDA support)
- **PyTorch**: Installing CUDA version (2.5.1+cu121) - **IN PROGRESS**
- **Disk Space**: 140GB available (plenty of space)
- **Training Data**: 13 preprocessed parquet files ready

### ⚠️ **Current Limitations:**
- **MLflow**: Installation blocked by Windows permissions
- **TensorFlow**: Not installed (Windows compatibility issues)

## 🎯 **Training Options Available**

### **Option 1: XGBoost GPU Training (READY NOW)**

XGBoost is already working with GPU acceleration. You can train models immediately:

```python
# Example XGBoost GPU training
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your traffic data
data = pd.read_parquet("data/traffic/preprocessed/your_data.parquet")
X = data.drop(['target_column'], axis=1)
y = data['target_column']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with GPU acceleration
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.1,
    device='cuda:0',  # Use GPU
    tree_method='hist',  # Modern method
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### **Option 2: PyTorch GPU Training (AVAILABLE SOON)**

Once PyTorch CUDA installation completes (currently downloading), you can use:

```python
import torch
import torch.nn as nn

# Check GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple neural network for traffic prediction
class TrafficPredictor(nn.Module):
    def __init__(self, input_size):
        super(TrafficPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Create and train model
model = TrafficPredictor(input_size=20).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with GPU acceleration
for epoch in range(100):
    # Your training code here
    pass
```

### **Option 3: Training CLI (NEEDS FIXES)**

The training CLI (`training_pipeline/cli/training_cli.py`) has these issues:
- MLflow installation permission errors
- TensorFlow import errors

**Workaround**: Use the individual training scripts directly instead of the CLI.

## 🛠️ **How to Run Training Now**

### **Step 1: Navigate to Project Directory**
```bash
cd c:\Users\Rizu\Desktop\Dev\komaneko
```

### **Step 2: Check GPU Status**
```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Step 3: Run XGBoost Training (Works Now)**
```bash
python -c "
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with GPU
model = xgb.XGBRegressor(device='cuda:0', tree_method='hist', n_estimators=100)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f'GPU XGBoost R² Score: {score:.4f}')
"
```

### **Step 4: Run PyTorch Training (After Installation Completes)**
```bash
python simple_gpu_test.py
```

## 📊 **Expected Performance**

### **Your RTX 3050 Specifications:**
- **VRAM**: 4GB GDDR6
- **CUDA Cores**: 2048
- **Memory Bandwidth**: 192 GB/s
- **CUDA Compute Capability**: 8.6

### **Training Performance Estimates:**
- **XGBoost**: 2-3x speedup over CPU
- **PyTorch Neural Networks**: 5-10x speedup over CPU
- **Memory Limit**: ~3.5GB available for models (4GB total - system overhead)

### **Recommended Batch Sizes:**
- **Small Models**: batch_size=64-128
- **Medium Models**: batch_size=32-64
- **Large Models**: batch_size=16-32

## 🚨 **Memory Management Tips**

### **For Your 4GB GPU:**
1. **Use Mixed Precision Training**: Reduces memory usage by ~50%
2. **Gradient Accumulation**: Simulate larger batches with smaller memory
3. **Model Checkpointing**: Save memory during backpropagation
4. **Clear Cache**: `torch.cuda.empty_cache()` between training runs

### **Example Memory-Efficient Training:**
```python
import torch

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Clear cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
```

## 🐳 **Docker Environment Setup (Option 3)**

### **Why Use Docker?**
- **Full CUDA Support**: TensorFlow + PyTorch + XGBoost all GPU-enabled
- **No Windows Limitations**: Complete Linux ML environment
- **Isolated Environment**: No dependency conflicts
- **Production Ready**: Same environment for development and deployment

### **Prerequisites**
```bash
# Check if Docker is installed
docker --version

# If not installed, download Docker Desktop for Windows
# https://docs.docker.com/desktop/install/windows-install/
```

### **Setup Steps**

#### **Step 1: Start Docker Environment**
```bash
cd c:\Users\Rizu\Desktop\Dev\komaneko

# Build and start containers (first time - may take 10-15 minutes)
docker-compose up -d --build

# Check container status
docker-compose ps
```

#### **Step 2: Access GPU-Enabled Environment**
```bash
# Enter the main application container
docker-compose exec app bash

# Verify GPU access inside container
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### **Step 3: Run Training in Docker**
```bash
# Inside the Docker container
cd /app

# Run the training CLI with full GPU support
python training_pipeline/cli/training_cli.py

# Or run comprehensive GPU tests
python /app/simple_gpu_test.py
```

### **Docker Benefits for Your RTX 3050**
- **TensorFlow GPU**: Full CUDA support (not available on Windows directly)
- **PyTorch GPU**: Latest CUDA version with optimal performance
- **XGBoost GPU**: Enhanced GPU acceleration
- **MLflow**: Full experiment tracking without permission issues
- **Memory Management**: Better GPU memory handling

### **Docker Commands Reference**
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs app

# Rebuild after changes
docker-compose up -d --build

# Access container shell
docker-compose exec app bash

# Monitor GPU usage
docker-compose exec app nvidia-smi -l 1
```

## 🎯 **Next Steps**

### **Immediate (Next 5 minutes)**
1. **Wait for PyTorch CUDA installation to complete** (~5 minutes remaining)
2. **Test GPU functionality** with updated `simple_gpu_test.py`
3. **Verify all 5 tests pass** (GPU detection, PyTorch, XGBoost, Neural Network, Training CLI)

### **Short Term (Next hour)**
1. **Run XGBoost training** on your actual traffic data
2. **Experiment with PyTorch models** with GPU acceleration
3. **Set up Docker environment** for full TensorFlow support
4. **Test Training CLI Option 3** with real data

### **Long Term (This week)**
1. **Scale up training** to full dataset (with 1-month limit)
2. **Implement memory-efficient training** for 4GB VRAM
3. **Set up automated benchmarking** across all models
4. **Deploy production training pipeline**

## 📞 **Troubleshooting**

### **If GPU Not Detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **If Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision
- Clear CUDA cache regularly

### **If Training Slow:**
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure data is on GPU: `data.to(device)`
- Use DataLoader with `pin_memory=True`

## 🎉 **Success Indicators**

You'll know everything is working when:
- ✅ `nvidia-smi` shows GPU activity during training
- ✅ `torch.cuda.is_available()` returns `True`
- ✅ Training speed is 5-10x faster than CPU
- ✅ GPU memory usage visible in `nvidia-smi`
