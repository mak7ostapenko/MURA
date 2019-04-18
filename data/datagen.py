import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



class MURADataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, preload=False):
        self.path_frame = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.len = len(self.path_frame)

        self.images = None
        self.labels = None
        self.patients = None
        self.study_types = None

    def __len__(self):
        """Total number of samples in the dataset"""
        return self.len

    def __getitem__(self, index):
        """Get a sample from the dataset

        Parameters
            index : int
                Index of sample for getting

        Returns
            result : list
                Data of training sample image, label, number of patient
                study type of patient

        """

        image = Image.open(self.path_frame.path[index]).convert('RGB')
        label = self.path_frame.label[index]
        patient = self.path_frame.patient[index]
        study_type = self.path_frame.study_type[index]

        if self.transform is not None:
            image = self.transform(image)

        result = [image, label, patient, study_type]

        return result