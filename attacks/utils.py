import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import os

class Preprocessor:
    def __init__(self, image_path) -> None:
        self.transform = v2.Compose([v2.Resize((224, 224)),
                                     v2.ToImage(),
                                     v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image_dir = image_path
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='DEFAULT')
        
    
    def load_images(self):
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directory '{self.image_dir}' not found!")
        
        # Load images using ImageFolder (expects images in subdirectories)
        train_data = datasets.ImageFolder(root=self.image_dir + '/train', transform=self.transform)
        train_data = [image for image, _ in train_data]
    
        test_data = datasets.ImageFolder(root=self.image_dir + '/test', transform=self.transform)
        test_data = [image for image, _ in test_data]

        print(f"test data shape: {train_data[0].shape}")
        print(f"test data shape: {test_data[0].shape}")

        return train_data, test_data
    
    def get_model(self):
        return self.model
