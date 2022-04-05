import numpy as np
import os
import cv2
from PIL import Image
import random
from matplotlib import pyplot as plt
import skimage.transform
import skimage.util
import torch


class MILDataset(object):

    def __init__(self, dir_images, data_frame, classes, bag_id='bag_name', input_shape=(3, 224, 224),
                 data_augmentation=False, images_on_ram=False, channel_first=True,
                 pMIL=False, proportions=None, only_primary=False, dataframe_instances=False):

        """Dataset object for MIL.
            Dataset object which aims to organize images and labels from a dataset in the form of bags.
        Args:
          dir_images: (h, w, channels)
          data_frame: pandas dataframe with ground truth information.
                      Each bag is one raw, with 'bag_name' as identifier.
          classes: list of classes of interest in data_fame (i.e. ['G3', 'G4', 'G5'])
          input_shape: image input shape (channels first).
          data_augmentation: whether to perform data augmentation (True) or not (False).
          images_on_ram: whether to load images on ram (True) or not (False). Recommended for accelerated training.

        Returns:
          MILDataset object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bag_id = bag_id
        self.data_augmentation = data_augmentation
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first
        self.pMIL = pMIL
        self.proportions = proportions
        self.images = os.listdir(dir_images)
        self.only_primary = only_primary
        self.dataframe_instances = dataframe_instances

        # Filter patches whose slide is not in the dataframe
        idx = np.in1d([ID.split('_')[0] for ID in self.images], self.data_frame[self.bag_id])
        images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
        self.images = images

        # Filter slides in the dataframe whose patches are not in the images folder
        self.data_frame = self.data_frame[
            np.in1d(self.data_frame[self.bag_id], [ID.split('_')[0] for ID in images])]

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        for i, item in enumerate([ID.split('_')[0] for ID in self.images]):
            if item not in self.D:
                self.D[item] = [i]
            else:
                self.D[item].append(i)

        self.y = self.data_frame[self.classes].values
        self.indexes = np.arange(len(self.images))

        if self.pMIL:
            self.proportions = self.data_frame[self.proportions].values

            self.O = []
            for i in np.arange(self.y.shape[0]):
                proportions = self.proportions[i, :]
                o = np.zeros((1, 3))

                if proportions[0] != 0 and proportions[1] != 0:
                    if proportions[0] == proportions[1]:
                        o[0, proportions[0] - 3] = 1
                    else:
                        o[0, proportions[0] - 3] = 0.8
                        o[0, proportions[1] - 3] = 0.2

                o = self.ordering_matrix(o.tolist()[0])
                self.O.append(o)

        if self.images_on_ram:

            # Pre-allocate images
            self.X = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
            self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)

            if self.dataframe_instances is not False:
                self.y_instances = -1 * np.ones((len(self.indexes), 4))

            # Load, and normalize images
            print('[INFO]: Training on ram: Loading images')
            for i in np.arange(len(self.indexes)):
                print(str(i) + '/' + str(len(self.indexes)), end='\r')

                ID = self.images[self.indexes[i]]
                # Load image
                x = Image.open(os.path.join(self.dir_images, ID))
                x = np.asarray(x)
                # Normalization
                x = self.image_normalization(x)
                self.X[self.indexes[i], :, :, :] = x
                self.Yglobal[self.indexes[i], 1:] = self.data_frame[classes][self.data_frame[self.bag_id] == self.images[i].split('_')[0]]

                if self.dataframe_instances is not False:
                    idx = np.argwhere(list(self.dataframe_instances['image_name'] == self.images[i]))
                    if idx.shape[0] > 0:
                        self.y_instances[i, :] = self.dataframe_instances[['NC', 'G3', 'G4', 'G5']].values[idx[0], :]

            print('[INFO]: Images loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.images[self.indexes[index]]

        if self.images_on_ram:
            x = np.squeeze(self.X[self.indexes[index], :, :, :])
        else:
            # Load image
            x = Image.open(os.path.join(self.dir_images, ID))
            x = np.asarray(x)
            # Normalization
            x = self.image_normalization(x)

        # data augmentation
        if self.data_augmentation:
            x_augm = self.image_transformation(x.copy())
        else:
            x_augm = None

        return x, x_augm

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        #if random.random() > 0.5:
        #    img = skimage.util.random_noise(img, var=random.random() ** 2)
        #if random.random() > 0.5:
        #    img = img + random.random() - 0.5
        #    img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def image_normalization(self, x):
        # image resize
        x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    def plot_image(self, x, norm_intensity=False):
        # channel first
        if self.channel_first:
            x = np.transpose(x, (1, 2, 0))
        if norm_intensity:
            x = x / 255.0

        plt.imshow(x)
        plt.axis('off')
        plt.show()

    def cifar10_test_dataset(self, dir_dataset):
        files = os.listdir(dir_dataset)
        files = [iFile for iFile in files if iFile != 'Thumbs.db']

        Y = []
        X = []
        for iFile in files:
            if 'Other' in iFile:
                y = 0
            else:
                y = int(iFile.split('_')[-2][-1])

            # Load image
            x = Image.open(os.path.join(dir_dataset, iFile))
            x = np.asarray(x)
            # Normalization
            x = self.image_normalization(x)

            Y.append(y)
            X.append(x)

        return np.array(X), np.array(Y)

    def ordering_matrix(self, p):

        if not self.only_primary:
            nRestrictions = len(np.where(np.array(p) > 0)[0])
        else:
            nRestrictions = len(np.where(np.array(p) > 0)[0]) - 1

        if nRestrictions <= 0:
            return [np.zeros((1, len(p))), np.zeros((1, len(p)))]

        # p: numpy array with proportion of used classes
        O = np.zeros((nRestrictions, len(p)))

        # Sort proportion values
        indexes = np.flip(np.argsort(p))

        for i in np.arange(0, nRestrictions):
            O[i, indexes[i]] = -1

        # p: numpy array with proportion of used classes
        if nRestrictions > 1:
            O2 = np.zeros((nRestrictions-1, len(p)))
            for i in np.arange(0, nRestrictions-1):
                O2[i, indexes[i]] = -1
                O2[i, indexes[i + 1]] = 1
        else:
            O2 = np.zeros((1, len(p)))

        return [O, O2]


class MILDataGenerator(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, max_instances=512):

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances

        self._idx = 0
        self._reset()

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):

        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # Get bag-level label
        Y = df_row[self.dataset.classes].to_list()
        Y = np.expand_dims(np.array(Y), 0)

        # Get ordering matrix
        if self.dataset.pMIL:
            O = self.dataset.O[self.indexes[self._idx]]

        # Select instances from bag
        ID = list(df_row[[self.dataset.bag_id]].values)[0]
        images_id = self.dataset.D[ID]

        # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.N)
        # Minimum number os patches in a slide (by precaution).
        if len(images_id) < 4:
            images_id.extend(images_id)

        self.instances_indexes = images_id

        # Load images and include into the batch
        X = []
        X_augm = []
        for i in images_id:
            x, x_augm = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)

        # Update bag index iterator
        self._idx += self.batch_size

        if self.dataset.pMIL:
            if self.dataset.data_augmentation:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), O, np.array(X_augm).astype('float32')
            else:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), O, None
        else:
            if self.dataset.data_augmentation:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, np.array(X_augm).astype('float32')
            else:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, None

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=True):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.indexes = np.arange(0, X.shape[0])
        self.channel_first = True

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        image = self.X[idx, :, :, :]
        label = self.Y[idx]
        if self.transform:
            image = self.image_transformation(image)

        return image, label

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img


class CustomGenerator(object):
    def __init__(self, train_dataset, bs, shuffle=True):
        self.dataset = train_dataset
        self.bs = bs
        self.shuffle = shuffle
        self.indexes = train_dataset.indexes.copy()
        self._idx = 0

    def __len__(self):
        return round(len(self.indexes) / self.bs)

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx + self.bs >= len(self.indexes):
            self._reset()
            raise StopIteration()

        # Load images and include into the batch
        X, Y = [], []
        for i in np.arange(self._idx, self._idx + self.bs):

            x, y = self.dataset.__getitem__(self.indexes[i])
            X.append(x)
            Y.append(y)

        self._idx += self.bs

        return torch.tensor(np.array(X).astype('float32')), torch.tensor(np.array(Y).astype('float32'))

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0
