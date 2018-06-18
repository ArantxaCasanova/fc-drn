import numpy as np
import os
import time

from scipy import interpolate

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'

class EmCvlabDataset(ThreadedDataset):
    ''' Segmentation of neuronal structures in Electron Microscopy (EM)
    stacks dataset

    EM stacks dataset is the basis of 2D segmentation of neuronal processes
    challenge [1]_. It provides a training set of 30 consecutive images
    (512 x 512 pixels) from a serial section transmission EM of the Drosophila
    first instar larva ventral nerve cord. The test set is a separate set of 30
    images, for which segmentation labels are not provided. The ground truth
    corresponds to a boundary map annotated by human experts, associating each
    pixel with one of 2 classes (cell or cell membrane).

    The dataset should be downloaded from [2]_ into the `shared_path`
    (that should be specified in the config.ini according to the
    instructions in ../README.md).

    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', test'], corresponding to
        the set to be returned.
    split: float
        A float indicating the dataset split between training and validation.
        For example, if split=0.85, 85\% of the images will be used
        for training, whereas 15\% will be used for validation.
        Will be ignored if crossval_nfolds is not None
    crossval_nfolds: int or None
        When None, cross-validation is disabled.
        Else, represents the number of folds the data will be split into.
    valid_fold: int
        An int specifying which fold to use for validation.
        If valid_fold=0, images from 0 to 5 will be used for validation.
        If valid_fold=1, images from 6 to 11, and so on.
    rand_perm: Numpy array or None
        When a Numpy array with a random permutation is provided,
        these random indices are used for the cross-validation.
        When None, no random permutation is used.

     References
    ----------
    .. [1] http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full
    .. [2] http://brainiac2.mit.edu/isbi_challenge/home
    '''
    name = 'em_cvlab'
    non_void_nclasses = 2
    _void_labels = []

    # optional arguments
    data_shape = (768, 1024, 1)
    _cmap = {
        0: (0, 0, 0),  # Non-membranes
        1: (255, 255, 255)}  # Membranes
    _mask_labels = {0: 'Non-membranes', 1: 'Membranes'}

    def __init__(self, which_set='train', split=0.50, crossval_nfolds=5,
                 valid_fold=3, rand_perm=None, *args, **kwargs):

        assert which_set in ["train", "valid", "val", "test", "unlabeled"]
        self.which_set = "val" if which_set == "valid" else which_set
        n_images=165
        if crossval_nfolds is not None:
            self.crossval = True
            if rand_perm is not None:
                self.rand_indices = rand_perm
            else:
                self.rand_indices = np.array(range(n_images))
            # number of images per fold
            img_per_fold = int(n_images / crossval_nfolds)

            if self.which_set == "train":
                self.start_1 = 0
                self.end_1 = valid_fold * img_per_fold
                self.start_2 = (valid_fold + 1) * img_per_fold
                self.end_2 = n_images
            elif self.which_set == "val":
                self.start_1 = valid_fold * img_per_fold
                self.end_1 = self.start_2 = self.end_2 = \
                    (valid_fold + 1) * img_per_fold
            elif self.which_set == "test" or self.which_set == "unlabeled":
                raise ValueError('Cannot perform cross-validation on test.')

        else:
            self.crossval = False
            if self.which_set == "train":
                self.start = 0
                self.end = int(split * n_images)
            elif self.which_set == "val":
                self.start = int(split * n_images)
                self.end = n_images
            elif self.which_set == "test":
                self.start = 0
                self.end = n_images
            elif self.which_set == "unlabeled":
                self.start = 0
                self.end = (1065 -n_images)*4 + n_images*2

        if self.which_set in ["train", "val"]:
            self.image_path = os.path.join(self.path, "training.tif")
            self.target_path = os.path.join(self.path, "training_groundtruth.tif")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "testing.tif")
            self.target_path = os.path.join(self.path, "testing_groundtruth.tif")
            self.set_has_GT = True
        elif self.which_set == "unlabeled":
            self.image_path = os.path.join(self.path, "training_unlabeled.tif")
            self.target_path = None
            self.set_has_GT = False

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(EmCvlabDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        if self.crossval:
            print({'default': (
                                   self.rand_indices[self.start_1:self.end_1]
                               ).tolist() + (
                                   self.rand_indices[self.start_2:self.end_2]).tolist()})
            return {'default': (
                                  self.rand_indices[self.start_1:self.end_1]
                              ).tolist() + (
                self.rand_indices[self.start_2:self.end_2]).tolist()}

        else:
            print({'default': range(self.start, self.end)})
            return {'default': range(self.start, self.end)}


    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from PIL import Image
        X = []
        Y = []
        F = []

        for prefix, idx in sequence:
            imgs = Image.open(self.image_path)
            imgs.seek(idx)
            imgs = np.array(imgs)[:, :, None].astype("uint8")

            if self.target_path is not None:
                targets = Image.open(self.target_path)
                targets.seek(idx)
                targets = np.array(targets) / 255

            X.append(imgs)
            if self.set_has_GT:
                Y.append(targets)
            F.append(idx)
        X = np.array(X)
        Y = np.array(Y)

        X = X.astype("float32") / 255
        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)

        return ret


def test():
    trainiter = EmCvlabDataset(
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
    validiter = EmCvlabDataset(
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
    testiter = EmCvlabDataset(
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
