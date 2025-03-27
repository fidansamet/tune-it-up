import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if self.opt.mixed_disc:     # use mixed dataset for additional discriminators
            self.mixed_paths = self.A_paths + self.B_paths
            self.mixed_size = self.A_size + self.B_size

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, normalize=(self.opt.preprocess != 'tensor'))
        self.transform_B = get_transform(self.opt, normalize=(self.opt.preprocess != 'tensor'))
        self.transform_mixed = get_transform(self.opt, normalize=(self.opt.preprocess != 'tensor'))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # apply image transformations
        A = self.midi_to_img(A_path, self.transform_A)
        B = self.midi_to_img(B_path, self.transform_B)
        data_item = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        if self.opt.mixed_disc:
            mixed_path = self.mixed_paths[random.randint(0, self.mixed_size - 1)]
            mixed = self.midi_to_img(mixed_path, self.transform_mixed)
            data_item['mixed'] = mixed

        if self.opt.triplet:
            random_A_path = self.A_paths[random.randint(0, self.A_size - 1)]
            random_B_path = self.B_paths[random.randint(0, self.B_size - 1)]
            random_A = self.midi_to_img(random_A_path, self.transform_A)
            random_B = self.midi_to_img(random_B_path, self.transform_B)
            data_item['A_random'] = random_A
            data_item['B_random'] = random_B

        return data_item

    def midi_to_img(self, path, transform):
        midi = np.load(path)
        midi = np.squeeze(midi, axis=2)
        img = Image.fromarray(midi)
        return transform(img)

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
