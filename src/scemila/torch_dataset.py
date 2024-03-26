from torch.utils.data import Dataset
import torchvision
import torch


class TrainingImageDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        preprocess,
        transform: torchvision.transforms.Compose = None,
    ):
        """
        CustomDataset constructor.

        Args:
        images (pd.Dataframe): Dataframe of PIL or Tensor images.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        preprocess (callable): Preprocessing to be applied to image after optional transformation
        transform (callable, optional): Optional transformations to be applied to the images. Default is None.
        """
        self._image = data
        self._target = target
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self._image)

    def __getitem__(self, index):
        image, label = self._image[index], self._target[index]

        if self.transform:
            image = self.transform(image)

        image = self.preprocess(image)

        return image, label

    @property
    def targets(self):
        return self._target

    @property
    def images(self):
        return self._image


class TestImageDataset(Dataset):
    def __init__(self, data, preprocess):
        """
        CustomDataset constructor.

        Args:
        images (pd.DataFrame): Dataframe of PIL or Tensor images.
        labels (torch.Tensor): Tensor containing the labels corresponding to the images.
        preprocess (callable): Preprocessing to be applied to image.
        """
        self.image = data
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        image = self.preprocess(image)
        return image