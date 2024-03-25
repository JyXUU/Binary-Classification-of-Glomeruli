import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# 定义图像数据集类
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]

# 加载模型
def load_model(model_path):
    model = models.resnet18() 
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 预测函数
def predict(model, dataloader, device):
    predictions = []
    files = []
    with torch.no_grad():
        for images, image_files in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            files.extend(image_files)
    return files, predictions

# 主函数
def main(folder_path, model_path, output_csv):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(root_dir=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_model(model_path).to(device)
    
    files, predictions = predict(model, dataloader, device)
    
    # 生成CSV文件
    df = pd.DataFrame({
        'Image': files,
        'Label': predictions
    })
    df.to_csv(output_csv, index=False)
    print(f"Prediction results saved to {output_csv}")

if __name__ == "__main__":
    folder_path = sys.argv[1]
    model_path = sys.argv[2]  
    output_csv = sys.argv[3] 
    main(folder_path, model_path, output_csv)
