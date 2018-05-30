from torch.utils.data import Dataset
import os
from PIL import Image

class UnsupervisedFolderDatset(Dataset):
    def __init__(self, root_dir, transform=None):
        self._root_dir = root_dir
        self._find_files()
        self._transform = transform

    def _find_files(self):
        self._files = []
        for filename in os.listdir(self._root_dir):
            if filename.endswith('.png'):
                self._files.append(os.path.join(self._root_dir, filename))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        filename = self._files[idx]
        with open(filename, 'rb') as f:
            im = Image.open(f)
            im = im.convert('RGB')

        if self._transform:
            im = self._transform(im)
        return im
