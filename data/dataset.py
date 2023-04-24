from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class Cifar100(Dataset):
    def __init__(self, paths_dir, transform):
        self.data_df = pd.read_csv(paths_dir)
        self.transform = transform


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['path']
        image = Image.open(img_path)
        label = self.data_df.iloc[idx]['label']
        image = self.transform(image)
        trigger = self.data_df.iloc[idx]['trigger']
        return image, label, trigger