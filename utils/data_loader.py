from dataset_loaders.images.camvid import CamvidDataset
from dataset_loaders.images.cityscapes import CityscapesDataset
# from dataset_loaders.images.camvid import CamvidDataset as CamvidDebugDataset
from dataset_loaders.images.isbi_em_stacks import IsbiEmStacksDataset
from dataset_loaders.images.pascalvoc import PascalVOCdataset
from dataset_loaders.images.em_cvlab import EmCvlabDataset
from dataset_loaders.images.fungus import FungusDataset
from dataset_loaders.images.gland import GlandDataset

import numpy as np

seed = 1609  # 2507 #1337
np.random.seed(seed)


def load_data(dataset, crop_size, one_hot=True,
              batch_size=[10, 10, 10], shuffle_train=True, return_0_255=False,
              return_01c=True, use_threads=False):
    data_augmentation_kwargs = {
        'rotation_range': 0.,
        'fill_mode': 'constant',
        'horizontal_flip': 0.5,
        'vertical_flip': False,
        'crop_size': crop_size,
    }

    # Build dataset iterator
    if dataset == 'camvid':
        train_iter = CamvidDataset(which_set='train',
                                   batch_size=batch_size[0],
                                   seq_per_subset=0,
                                   seq_length=0,
                                   data_augm_kwargs=data_augmentation_kwargs,
                                   return_one_hot=one_hot,
                                   return_01c=return_01c,
                                   overlap=0,
                                   use_threads=use_threads,
                                   shuffle_at_each_epoch=shuffle_train,
                                   return_list=True,
                                   return_0_255=return_0_255,
                                   fill_last_batch=True)
        val_iter = CamvidDataset(which_set='val',
                                 batch_size=batch_size[1],
                                 seq_per_subset=0,
                                 seq_length=0,
                                 return_one_hot=one_hot,
                                 return_01c=return_01c,
                                 overlap=0,
                                 use_threads=use_threads,
                                 shuffle_at_each_epoch=False,
                                 return_list=True,
                                 return_0_255=return_0_255,
                                 fill_last_batch=True)
        test_iter = CamvidDataset(which_set='test',
                                  batch_size=batch_size[2],
                                  seq_per_subset=0,
                                  seq_length=0,
                                  return_one_hot=one_hot,
                                  return_01c=return_01c,
                                  overlap=0,
                                  use_threads=use_threads,
                                  shuffle_at_each_epoch=False,
                                  return_list=True,
                                  return_0_255=return_0_255,
                                  fill_last_batch=True)

    else:
        raise NotImplementedError

    ret = [train_iter, val_iter, test_iter]

    return ret
