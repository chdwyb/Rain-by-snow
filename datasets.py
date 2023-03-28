import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()
        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        index = index % len(self.inputImages)
        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')
        input_ = ttf.to_tensor(inputImage)

        return input_, self.inputImages[index]
