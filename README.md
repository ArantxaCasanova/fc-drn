# On the refinement of densely connected representation levels for semantic segmentation
## Arantxa Casanova, Guillem Cucurull, Michal Drozdzal, Adriana Romero, Yoshua Bengio

[[paper](https://arxiv.org/abs/1804.11332)] [[poster](images/poster_casanova2018.pdf)] [slides] [oral presentation]

This repository contains the Pytorch implementation of the Fully Convolutional DenseResNet (FC-DRN), the architecture for semantic segmentation introduced in the paper **On the refinement of densely connected representation levels for semantic segmentation** presented at the Workshop for Autonomous Driving (WAD) and Women in Computer Vision Workshop(WiCV), both at CVPR2018.

The architecture combines DenseNets and ResNets. This allows the model to exploit the benefits of both residual and dense connectivity patterns, namely: gradient flow, iterative refinement of representations, multi-scale feature combination and deep supervision

#### Architecture overview
![Alt text](images/model.png?raw=true "Architecture overview")


#### Comparison with state-of-the-art methods in Camvid dataset
![Alt text](images/results.png?raw=true "Comparison with state-of-the-art methods in Camvid dataset")

#### Test image - Ground truth - Prediction
![Alt text](images/sample_image.png?raw=true "Test image - Ground truth - Prediction")


## Code overview

- **run.py** - This script contains the main loop for both training, validating and testing. 
- **utils/data_loader.py** - Script to call Camvid dataset loader from 'dataset_loaders' folder.
- **utils/logging.py** - Logging and printing utilities.
- **utils/parser.py** - Parser file that contains all the possible input arguments for the 'run.py' script.
- **utils/script_utils.py** - Saving/loading checkpoints, setting up parameters for the network, creating optimizer and adjusting learning rate functions.
- **utils/utils.py** - progress bar, save images, confusion matrix and metric computation functions.

- **models/fc_drn_model.py** - FC-DRN model that accepts all the 5 variants of the architecture mentioned in the paper.
- **weights_pretrained/fc-drn-p-d/** - contains the weights of FC-DRN model, pretrained with max poolings and finetunned with dilations.
- **dataset_loaders/** - Part of dataset_loaders repository from Francesco Visin https://github.com/fvisin/dataset_loaders.


## Code usage
The code was run with Python 3.6.3 and pytorch 0.3 version (torch 0.3.0.post4).
This code is offered with minimal support.

To 'run.py' script is prepared to evaluate the performance of FC-DRN-P-D (with soft targets training).

``` 
run.py
``` 


To train a FC-DRN-P model (with max poolings and upsamplings):
```
python run.py --train --model pools --exp-name "fc-drn-p-from_scratch"
```

To train a FC-DRN-S model (with strided convolutions and upsamplings):
```
python run.py --train --model sconv --exp-name "fc-drn-s-from_scratch"
```

To train a FC-DRN-D model (with dilated convolutions and convolutions):
```
python run.py --train --model dils --exp-name "fc-drn-d-from_scratch"  --train-batch-size 2
```

To train a FC-DRN-P-D model (with max poolings, dilated convs, convs and upsamplings):
```
python run.py --train --model pools_ft_dils --exp-name "fc-drn-p-d-from_scratch"
```

To train a FC-DRN-S-D model (with strided convolutions, dilated convs, convs and upsamplings):
```
python run.py --train --model sconv_ft_dils --exp-name "fc-drn-s-d-from_scratch"
```

## Bibtex 
```
@article{casanova2018iterative,
  title={On the iterative refinement of densely connected representation levels for semantic segmentation},
  author={Casanova, Arantxa and Cucurull, Guillem and Drozdzal, Michal and Romero, Adriana and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1804.11332},
  year={2018}
}
```
