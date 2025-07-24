import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_transform_for_user_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((28, 28)),                
    transforms.ToTensor(),                     

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, conv_layers, pool_kernel, pool_stride):
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

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 28, 28)
            try:
                conv_output_shape = self.conv(dummy_input).shape
                if conv_output_shape[2] <= 0 or conv_output_shape[3] <= 0:
                     raise ValueError("Convolutional layers lead to 0 or negative spatial dimensions.")
            except Exception as e:
                raise ValueError(f"Error during dummy forward pass: {e}. "
                                 f"Check 'Number of Conv Blocks' or 'MaxPool Stride'.")

    def forward(self, x):
        return self.conv(x)

def visualize_feature_maps(model, image_tensor, device):
    model.eval()
    x = image_tensor.to(device)
    outputs = []
    layer_names = []

    st.subheader("Visualizing Feature Maps of Convolutional Layers")

    for idx, layer in enumerate(model.conv):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            fmap = x.detach().cpu().squeeze(0)
            outputs.append(fmap)
            layer_names.append(f"{layer.__class__.__name__} - {idx}")

    if not outputs:
        st.warning("No feature maps were captured for visualization. Check model architecture.")
        return

    for fmap, name in zip(outputs, layer_names):
        num_filters = fmap.shape[0]
        if num_filters == 0 or fmap.shape[1] == 0 or fmap.shape[2] == 0:
            st.warning(f"Feature map for '{name}' is empty or has collapsed dimensions. Skipping visualization.")
            continue

        cols = 8
        rows = (num_filters + cols - 1) // cols
        
        fig_height = max(4, 2.2 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=(16, fig_height))
        axes = axes.flatten()

        for i in range(rows * cols):
            ax = axes[i]
            ax.axis('off')
            if i < num_filters:
                feature_map_slice = fmap[i]
                if feature_map_slice.max() - feature_map_slice.min() > 1e-6:
                    feature_map_slice = (feature_map_slice - feature_map_slice.min()) / \
                                        (feature_map_slice.max() - feature_map_slice.min())
                ax.imshow(feature_map_slice, cmap='gray') # Ensure grayscale colormap
                ax.set_title(f'F{i}', fontsize=6)
            else:
                ax.set_visible(False)

        fig.suptitle(f"Feature Maps after {name}", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)

st.title("CNN Feature Extractor Visualizer - FashionMNIST")
st.sidebar.header("Model Parameters (Convolutional Layers Only)")

conv_layers = st.sidebar.slider("Number of Conv Blocks", 1, 4, 3)
pool_kernel = st.sidebar.selectbox("MaxPool Kernel", [2, 3], index=0)
pool_stride = st.sidebar.selectbox("MaxPool Stride", [1, 2], index=1)

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Train Feature Extractor and Show Feature Maps"):
    st.write("---")

    try:
        model = CNNFeatureExtractor(
            input_channels=1,
            conv_layers=conv_layers,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with st.spinner("Training convolutional layers for 5 epochs..."):
            for epoch in range(5):
                model.train()
                total_loss = 0
                for images, _ in train_loader:
                    images = images.to(device)
                    feature_maps_output = model(images)
                    loss = torch.mean(torch.abs(feature_maps_output))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                st.write(f"Epoch {epoch+1} completed. Average 'Feature Sparsity' Loss: {total_loss / len(train_loader):.4f}")

        st.success("Training of convolutional layers complete!")

        image_for_visualization = None
        if uploaded_file is not None:
            uploaded_image_pil = Image.open(uploaded_file)
            
            st.subheader("Uploaded Input Image")
            st.image(uploaded_image_pil, caption="Original Uploaded Image", use_container_width=True)

            image_for_visualization = target_transform_for_user_image(uploaded_image_pil).unsqueeze(0).to(device)
            
            st.subheader("Transformed Input Image (28x28 Grayscale)")
            transformed_img_np = image_for_visualization.squeeze(0).squeeze(0).cpu().numpy()
            transformed_img_uint8 = (transformed_img_np * 255).clip(0, 255).astype("uint8")
            st.image(transformed_img_uint8, caption="Resized & Grayscaled for Model Input", use_container_width=False, width=100, channels='grayscale')

        else:
            st.info("No image uploaded. Using a sample image from FashionMNIST for visualization.")
            sample_img, _ = next(iter(test_loader))
            image_for_visualization = sample_img.to(device) 
            
            img_np = image_for_visualization.squeeze(0).cpu().numpy()
            img_uint8 = (img_np * 255).clip(0, 255).astype("uint8")
            st.image(img_uint8, caption="FashionMNIST Sample Input Image", use_container_width=True, channels='grayscale')

        model.eval()
        visualize_feature_maps(model, image_for_visualization, device)

    except ValueError as e:
        st.error(f"Model Configuration Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during training or visualization: {e}")
