import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# CNN 
class CNN(nn.Module):
    def __init__(self, input_channels, conv_layers, hidden, neurons_per, dropout_rate, pool_kernel, pool_stride, output_dim=10):
        super().__init__()
        layers = []
        in_channels = input_channels

        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(32))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
            in_channels = 32

        self.conv = nn.Sequential(*layers)
        conv_output_dim = 28 // (pool_stride ** conv_layers)
        in_features = 32 * conv_output_dim * conv_output_dim

        fc_layers = []
        for _ in range(hidden):
            fc_layers.append(nn.Linear(in_features, neurons_per))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            in_features = neurons_per

        fc_layers.append(nn.Linear(in_features, output_dim))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# FEATURE MAP 
def visualize_feature_maps(model, image_tensor, device):
    model.eval()
    x = image_tensor.to(device)
    outputs = []
    layer_names = []

    for idx, layer in enumerate(model.conv):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            fmap = x.detach().cpu().squeeze(0)  # Shape: (C, H, W)
            outputs.append(fmap)
            layer_names.append(f"{layer.__class__.__name__} - {idx}")

    for fmap, name in zip(outputs, layer_names):
        num_filters = fmap.shape[0]
        cols = 8
        rows = (num_filters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2.2 * rows))
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            ax.axis('off')
            if i < num_filters:
                ax.imshow(fmap[i], cmap='gray')
                ax.set_title(f'F{i}', fontsize=6)
        fig.suptitle(f"Feature Maps after {name}", fontsize=12)
        st.pyplot(fig)


# STREAMLIT UI
st.title(" CNN Visualizer - FashionMNIST")
st.sidebar.header("Model Parameters")
conv_layers = st.sidebar.slider("Number of Conv Blocks", 1, 5, 3)
hidden_layers = st.sidebar.slider("Hidden Layers", 1, 4, 1)
neurons_per_layer = st.sidebar.slider("Neurons per Hidden Layer", 32, 256, 128, step=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.9, 0.3)
pool_kernel = st.sidebar.selectbox("MaxPool Kernel", [2,3,4,5], index=0)
pool_stride = st.sidebar.selectbox("MaxPool Stride", [1,2,3,4], index=1)

if st.button(" Train Model and Show Feature Maps"):
    model = CNN(
        input_channels=1,
        conv_layers=conv_layers,
        hidden=hidden_layers,
        neurons_per=neurons_per_layer,
        dropout_rate=dropout_rate,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    with st.spinner("Training for 10 epochs..."):
        for epoch in range(10):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            st.write(f"epoch {epoch+1} completed")

    sample_img, _ = next(iter(test_loader))
    sample_img = sample_img.to(device)
    img_np = sample_img[0].squeeze(0).cpu().numpy()
    img_uint8 = (img_np * 255).clip(0, 255).astype("uint8")
    st.image(img_uint8, caption=" Input Image", use_column_width=True)
    visualize_feature_maps(model, sample_img, device)
