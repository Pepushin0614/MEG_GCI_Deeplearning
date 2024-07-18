import os, sys
import re
project_root = "C:\\Users\\dl_lecture_competition_pub-MEG-competition\\"
sys.path.append(os.path.join(project_root, "src"))
os.chdir(project_root)

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
import clip
from PIL import Image
#%%　事前学習
device = "cuda"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def preprocess_images(image_paths):
    images = [preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths]
    return torch.cat(images, dim=0)

def get_clip_image_features(image_paths):
    images = preprocess_images(image_paths).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
    return image_features.cpu()

def get_all_image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

images_dir = os.path.join(project_root+"\\data\\images", "Images")

image_paths = get_all_image_paths(images_dir)

batch_size = 128
all_image_features = []
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    image_features = get_clip_image_features(batch_paths)
    all_image_features.append(image_features)


# Concatenate all features into a single tensor
all_image_features = torch.cat(all_image_features, dim=0)

# Save the features for later use
torch.save(all_image_features, os.path.join(project_root+"result", "image_features.pt"))
#%%
batch_size = 128
epochs = 80
lr = 0.001

# device: cuda:0
num_workers = 0
seed = 1234
use_wandb = False
device = "cuda"
set_seed(seed)

#%%
# ------------------
#    Dataloader
# ------------------
loader_args = {"batch_size": batch_size, "num_workers": num_workers,"pin_memory": True}
 
train_set = ThingsMEGDataset("train", project_root+"\\data")
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = ThingsMEGDataset("val", project_root+"\\data")
val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
test_set = ThingsMEGDataset("test", project_root+"\\data")
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)

models = {"BasicConvClassifier": BasicConvClassifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(device)}

#事前学習の導入 精度が低かったため導入無し
# pretrained_image_features = torch.load(os.path.join(project_root+"result", "image_features.pt"))
# pretrained_image_features = pretrained_image_features.to(torch.float32)
# models = {
#     "BasicConvClassifier": BasicConvClassifier(
#         num_classes=train_set.num_classes,
#         seq_len=train_set.seq_len,
#         in_channels=train_set.num_channels,
#         pretrained_feature_dim=pretrained_image_features.size(1)
#     ).to(device)
# }

#%%
optimizers = {
    "BasicConvClassifier": {
        "Adam": torch.optim.Adam(models["BasicConvClassifier"].parameters(), lr=lr),
        "RMSprop": torch.optim.RMSprop(models["BasicConvClassifier"].parameters(), lr=lr, alpha=0.99),
        "AdamW": torch.optim.AdamW(models["BasicConvClassifier"].parameters(), lr=lr, weight_decay=1e-2),
        "NAdam": torch.optim.NAdam(models["BasicConvClassifier"].parameters(), lr=lr)
    }
}

accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(device)
#%%
# Function to reset model parameters
def reset_model_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_and_validate(model, optimizer, train_loader, val_loader, epochs, device, step_size=10, gamma=0.1, patience=5):
    max_val_acc = 0
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_val_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y) 
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_pred = model(X)
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        

        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc)
        avg_val_loss = np.mean(val_loss)
        avg_val_acc = np.mean(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(project_root+"result", "model_last.pt"))
        if use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(project_root+"result", "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
        
        # # 早期停止のチェック
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
            
        scheduler.step()
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(project_root+"result", "model_best.pt"), map_location=device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(project_root+"result", "submission"), preds)
    # cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    return max_val_acc

# Test each optimizer
results = {}    
for model_name, model in models.items():
    for opt_name, optimizer in optimizers[model_name].items():
        print(f"\nTesting model: {model_name} with optimizer: {opt_name}")
        reset_model_parameters(model)
        model.to(device)
        max_val_acc = train_and_validate(model, optimizer, train_loader, val_loader, epochs, device)
        results[f"{model_name} with {opt_name}"] = max_val_acc

# Print results
print("\nResults:")
for model_opt_combo, val_acc in results.items():
    print(f"{model_opt_combo}: {val_acc:.3f}")

# Choose the best model and optimizer based on validation accuracy
best_model_opt_combo = max(results, key=results.get)
print(f"\nBest model and optimizer combination: {best_model_opt_combo} with validation accuracy: {results[best_model_opt_combo]:.3f}")


#%% 以下は検証したモデル
# class RNNClassifier(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         input_size: int,
#         hidden_size: int,
#         num_layers: int = 2,
#         bidirectional: bool = False,
#         dropout: float = 0.5
#     ) -> None:
#         super(RNNClassifier, self).__init__()
        
#         self.rnn = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=bidirectional,
#             batch_first=True,
#             dropout=dropout
#         )
        
#         self.dropout = nn.Dropout(dropout)
        
#         if bidirectional:
#             self.fc = nn.Linear(hidden_size * 2, num_classes)
#         else:
#             self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """_summary_
#         Args:
#             X ( b, t, c ): _description_
#         Returns:
#             X ( b, num_classes ): _description_
#         """
#         h0 = torch.zeros(self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), X.size(0), self.rnn.hidden_size).to(X.device)
#         c0 = torch.zeros(self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), X.size(0), self.rnn.hidden_size).to(X.device)
        
#         out, _ = self.rnn(X, (h0, c0))
#         out = self.dropout(out[:, -1, :])
#         out = self.fc(out)
        
#         return out


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super(ResNetBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else None

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.skip_connection:
#             identity = self.skip_connection(identity)
#         out += identity
#         return F.relu(out)

# class ResNetClassifier(nn.Module):
#     def __init__(self, num_classes, in_channels, layers):
#         super(ResNetClassifier, self).__init__()
#         self.in_channels = in_channels
#         self.layers = nn.ModuleList()
#         for out_channels, num_blocks in layers:
#             for _ in range(num_blocks):
#                 self.layers.append(ResNetBlock(in_channels, out_channels))
#                 in_channels = out_channels
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(out_channels, num_classes)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)
    

# models = {
#     "BasicConvClassifier": BasicConvClassifier(train_set.num_classes, train_set.seq_len, train_set.num_channels).to(device),
#     "RNNClassifier": RNNClassifier(num_classes=train_set.num_classes, input_size=train_set.seq_len, hidden_size=train_set.num_channels).to(device),
#     "ResNetClassifier": ResNetClassifier(num_classes=train_set.num_classes, in_channels=train_set.num_channels, layers=[(64, 2), (128, 2), (256, 2)]).to(device),
# }


# optimizers = {
#     "BasicConvClassifier": {
#         "Adam": torch.optim.Adam(models["BasicConvClassifier"].parameters(), lr=lr),
#         "RMSprop": torch.optim.RMSprop(models["BasicConvClassifier"].parameters(), lr=lr, alpha=0.99),
#         "AdamW": torch.optim.AdamW(models["BasicConvClassifier"].parameters(), lr=lr, weight_decay=1e-2),
#         "NAdam": torch.optim.NAdam(models["BasicConvClassifier"].parameters(), lr=lr)
#     },
#     "RNNClassifier": {
#         "Adam": torch.optim.Adam(models["RNNClassifier"].parameters(), lr=lr),
#         "RMSprop": torch.optim.RMSprop(models["RNNClassifier"].parameters(), lr=lr, alpha=0.99),
#         "AdamW": torch.optim.AdamW(models["RNNClassifier"].parameters(), lr=lr, weight_decay=1e-2),
#         "NAdam": torch.optim.NAdam(models["RNNClassifier"].parameters(), lr=lr)
#     },
#     "ResNetClassifier": {
#         "Adam": torch.optim.Adam(models["ResNetClassifier"].parameters(), lr=lr),
#         "RMSprop": torch.optim.RMSprop(models["ResNetClassifier"].parameters(), lr=lr, alpha=0.99),
#         "AdamW": torch.optim.AdamW(models["ResNetClassifier"].parameters(), lr=lr, weight_decay=1e-2),
#         "NAdam": torch.optim.NAdam(models["ResNetClassifier"].parameters(), lr=lr)
#     }
# }
