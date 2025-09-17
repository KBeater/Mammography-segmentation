from torch.utils.data import Dataset
import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class MammoSegmentationDataset(Dataset):
  def __init__(self, carpeta, transform=None, target_size=(512, 512)):
      self.carpeta = carpeta
      self.transform = transform
      self.target_size = target_size
      self.archivos = [f for f in os.listdir(carpeta) if f.endswith(".jpg")]

  def __len__(self):
    return len(self.archivos)

  def __getitem__(self, idx):
    img_name = self.archivos[idx]
    base_name = os.path.splitext(img_name)[0]
    mask_name = base_name + "_mask.png"

    img_path=os.path.join(self.carpeta, img_name)
    mask_path=os.path.join(self.carpeta, mask_name)

    image=cv2.imread(img_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image=cv2.resize(image, self.target_size)
    mask=cv2.resize(mask, self.target_size)

    image=image/255.0
    mask=mask/255.0
    mask=(mask>0.5).astype("float32") #Binarizo la imagen por seguridad, El 0.5 es importante, hiperparametro (buscar porque)

    image=torch.tensor(image).permute(2, 0, 1).float() #Cambio el orden de los ejes (dimensiones)
    mask=torch.tensor(mask).unsqueeze(0).float() #Agrego una dimensión extra a la máscara

    return image, mask, img_name

dataset_path = "/content/drive/MyDrive/Mammography segmentation/Dataset"
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MammoSegmentationDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for images, masks, names in dataloader:
    print("Lote imágenes:", images.shape)
    print("Lote máscaras:", masks.shape)
    break

# Tomar un batch del dataloader
for images, masks, names in dataloader:
    images=images.to(device)
    masks=masks.to(device)
    break  # Solo tomamos el primer batch

N = images.shape[0]  # número de imágenes en el batch (en tu caso 8)

plt.figure(figsize=(10, 2.5 * N))  # ajusta altura

for i in range(N):

    img = images[i].permute(1, 2, 0)  # [C, H, W] → [H, W, C]
    mask = masks[i][0]  # [1, H, W] → [H, W]
    print(img.device)
    img=img.cpu().numpy()
    mask=mask.cpu().numpy()
    print(img.device)
    #print(images.shape)
    #print(masks.shape)
    #print(images.dtype)
    #print(masks.dtype)
    # Imagen original
    plt.subplot(N, 2, 2*i + 1)
    plt.imshow(img)
    plt.title(f"Imagen {i+1}")
    plt.axis('off')

    # Imagen con máscara superpuesta
    plt.subplot(N, 2, 2*i + 2)
    plt.imshow(img, alpha=0.9)
    plt.imshow(mask, cmap='Reds', alpha=0.4)
    plt.title(f"Máscara superpuesta {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
  def forward(self, x):
    return self.double_conv(x)

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1):
    super(UNet, self).__init__()

    self.enc1=DoubleConv(in_channels, 64)
    self.enc2=DoubleConv(64, 128)
    self.enc3=DoubleConv(128, 256)
    self.enc4=DoubleConv(256, 512)

    self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    self.bottleneck=DoubleConv(512, 1024)

    self.up4=nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec4=DoubleConv(1024, 512)

    self.up3=nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3=DoubleConv(512, 256)

    self.up2=nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2=DoubleConv(256, 128)

    self.up1=nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1=DoubleConv(128, 64)

    self.final_conv=nn.Conv2d(64, out_channels, kernel_size=1)
  def forward(self, x):
    enc1=self.enc1(x)
    enc2=self.enc2(self.pool(enc1))
    enc3=self.enc3(self.pool(enc2))
    enc4=self.enc4(self.pool(enc3))

    bottleneck=self.bottleneck(self.pool(enc4))

    dec4=self.up4(bottleneck)
    dec4=torch.cat((dec4, enc4), dim=1)
    dec4=self.dec4(dec4)

    dec3=self.up3(dec4)
    dec3=torch.cat((dec3, enc3), dim=1)
    dec3=self.dec3(dec3)

    dec2=self.up2(dec3)
    dec2=torch.cat((dec2, enc2), dim=1)
    dec2=self.dec2(dec2)

    dec1=self.up1(dec2)
    dec1=torch.cat((dec1, enc1), dim=1)
    dec1=self.dec1(dec1)

    return torch.sigmoid(self.final_conv(dec1)) #La salida será entre 0 y 1

learning_rate = 1e-4
epochs = 20
batch_size = 8

model = UNet().to(device)

# Función de pérdida y optimizador
criterion = nn.BCEWithLogitsLoss() # Mejor para segmentación binaria, es numericamente mas estable que usar sigmoid con BCELoss
toptimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
model.train()
running_loss = 0.0

for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
images = images.to(device)
masks = masks.to(device)
# Forward pass
outputs = model(images)
# Calcular pérdida
loss = criterion(outputs, masks)
# Backpropagation
toptimizer.zero_grad()
loss.backward()
toptimizer.step()
running_loss += loss.item() * images.size(0)
epoch_loss = running_loss / len(dataloader.dataset)
print(f"\nEpoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
print("Entrenamiento finalizado.")