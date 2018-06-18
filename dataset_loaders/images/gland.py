import numpy as np
import os
import time

from scipy import interpolate

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'

class GlandDataset(ThreadedDataset):
    ''' Segmentation of fungus slices
    '''
    name = 'gland'
    non_void_nclasses = 2
    _void_labels = []

    # optional arguments
    _cmap = {
        0: (0, 0, 0),  # Background
        1: (255, 255, 255)}  # Gland
    _mask_labels = {0: 'Background', 1: 'Gland'}
    _filenames = None
    _prefix_list = None
    def __init__(self, which_set='train', split=0.80, rand_perm=None, *args, **kwargs):

        assert which_set in ["train", "valid", "val", "testA", "testB", "unlabeled"]
        self.which_set = "val" if which_set == "valid" else which_set
        n_images=85

        if rand_perm is not None:
            self.rand_indices = rand_perm
        if self.which_set == "train":
            self.start = 0
            self.end = int(split * n_images)
        elif self.which_set == "val":
            self.start = int(split * n_images)
            self.end = n_images
        elif self.which_set == "testA":
            self.start = 0
            self.end = 60
        elif self.which_set == "testB":
            self.start = 0
            self.end = 20
        elif self.which_set == "unlabeled":
            self.start = 0
            self.end = 100

        if self.which_set in ["train", "val"]:
            self.image_path = os.path.join(self.path, 'train','img' )
            self.target_path =os.path.join(self.path, 'train', 'gt' )
        elif self.which_set == "testA":
            self.image_path = os.path.join(self.path, 'testA', 'img' )
            self.target_path = os.path.join(self.path, 'testA', 'gt' )
            self.set_has_GT = True
        elif self.which_set == "testB":
            self.image_path = os.path.join(self.path, 'testB', 'img')
            self.target_path = os.path.join(self.path, 'testB', 'gt')
            self.set_has_GT = True
        elif self.which_set == "unlabeled":
            self.image_path = os.path.join(self.path, 'train','unlabeled' )
            self.target_path = None
            self.set_has_GT = False

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(GlandDataset, self).__init__(*args, **kwargs)

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            self._prefix_list=[]
            for el in self.filenames:
                self._prefix_list.append('default')
        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = []
            # Get file names for this set
            for root, dirs, files in os.walk(self.image_path):
                for name in files:
                    self._filenames.append(os.path.join(name))

            # Note: will get modified by prefix_list
        return self._filenames

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        """Return a dict of names, per prefix/subset."""
        image_dict = {}
        # Populate self.filenames and self.prefix_list
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            image_dict[prefix] = [el for el in self.filenames[self.start:self.end]]
        return image_dict


    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        X = []
        Y = []
        F = []

        for prefix, frame in sequence:
            img = io.imread(os.path.join(self.image_path, frame))
            img =  np.array(img).astype("uint8")
            X.append(img)
            F.append(frame)

            if self.target_path is not None:
                mask = io.imread(os.path.join(self.target_path, frame))
                mask = mask.astype('int32')
                mask[mask.nonzero()] = 1
                Y.append(mask)

        X = np.array(X).astype("float32") / 255
        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = GlandDataset(
        which_set='train',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        return_one_hot=True,
        return_01c=True,
        data_augm_kwargs={
            'crop_size': (224, 224),
            'fill_mode': 'nearest',
            'horizontal_flip': True,
            'vertical_flip': True,
            'warp_sigma': 1,
            'warp_grid_size': 10,
            'spline_warp': True},
        return_list=True,
        use_threads=True)
    validiter = GlandDataset(
        which_set='val',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        return_one_hot=True,
        return_01c=True,
        data_augm_kwargs={},
        return_list=True,
        use_threads=True)
    testiter = GlandDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        overlap=0,
        return_one_hot=True,
        return_01c=True,
        data_augm_kwargs={},
        return_list=True,
        use_threads=True)

    # Get number of classes
    nclasses = trainiter.nclasses
    print("N classes: " + str(nclasses))
    void_labels = trainiter.void_labels
    print("Void label: " + str(void_labels))

    # Training info
    train_nsamples = trainiter.nsamples
    train_batch_size = trainiter.batch_size
    train_nbatches = trainiter.nbatches
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(
        train_nsamples, train_batch_size, train_nbatches))

    # Validation info
    valid_nsamples = validiter.nsamples
    valid_batch_size = validiter.batch_size
    valid_nbatches = validiter.nbatches
    print("Validation n_images: {}, batch_size: {}, n_batches: {}".format(
        valid_nsamples, valid_batch_size, valid_nbatches))

    # Testing info
    test_nsamples = testiter.nsamples
    test_batch_size = testiter.batch_size
    test_nbatches = testiter.nbatches
    print("Test n_images: {}, batch_size: {}, n_batches: {}".format(
        test_nsamples, test_batch_size, test_nbatches))

    start = time.time()
    tot = 0
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(train_nbatches):
            train_group = trainiter.next()
            valid_group = validiter.next()
            test_group = testiter.next()

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (224, 224, 1)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (224, 224, nclasses)

            # valid_group checks
            assert valid_group[0].ndim == 4
            assert valid_group[0].shape[0] <= valid_batch_size
            assert valid_group[0].shape[1:] == (512, 512, 1)
            assert valid_group[0].min() >= 0
            assert valid_group[0].max() <= 1
            assert valid_group[1].ndim == 4
            assert valid_group[1].shape[0] <= valid_batch_size
            assert valid_group[1].shape[1:] == (512, 512, nclasses)

            # test_group checks
            assert test_group[0].ndim == 4
            assert test_group[0].shape[0] <= test_batch_size
            assert test_group[0].shape[1:] == (512, 512, 1)
            assert test_group[0].min() >= 0
            assert test_group[0].max() <= 1

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
