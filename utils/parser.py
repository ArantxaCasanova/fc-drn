import argparse
import os


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="FC-DRN: Fully Convolutional DenseResNet")
    parser.add_argument('--results-dir', type=str, nargs='?',
                        default="weights_pretrained/",
                        help="Temp directory to save checkpoints during "
                             "training. To disable intermediate directory,"
                             " set this argument to the same value that"
                             " 'results-dir-final'")

    parser.add_argument('--results-dir-final', type=str, nargs='?',
                        default="weights_pretrained/",
                        help="Final directory where experiment related files "
                             "are stored: log file, json file with script"
                             " configuration, weights (last epoch and best"
                             " jaccard) and evaluation images.")

    parser.add_argument("--exp-name", type=str, default='fc-drn-p-d',
                        help="Experiment name")

    parser.add_argument("--load-weights", action='store_true',
                        help="Load experiment in '--exp-name-toload' "
                             "specified folder.")
    parser.add_argument("--exp-name-toload", type=str, default='',
                        help="Introduce an experiment name to load. "
                             "Script will begin training from this "
                             "experiments' best weights.")

    parser.add_argument("--model", type=str, default='pools_ft_dils',
                        choices=['pools', 'dils', 'sconv', 'pools_ft_dils',
                                 'sconv_ft_dils'],
                        help="Change transformation types.")

    parser.add_argument("--dataset", type=str, default='camvid',
                        help="Dataset to use. Options: 'camvid'.")
    parser.add_argument("--train-batch-size", type=int, default=3)
    parser.add_argument("--val-batch-size", type=int, default=1)

    parser.add_argument("--epoch-num", type=int, default=1000,
                        help="Number of epochs to train.")
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='RMSprop',
                        help="Optimizer. Options: 'RMSprop', 'sgd', 'Adam'. ")
    parser.add_argument("--weight-decay", type=int, default=0.00001)
    parser.add_argument("--init-dils", type=str, default='identity',
                        help="Initialization for dilated convolutions."
                             " Options: 'identity', 'random'.")

    parser.add_argument("--crop-size", nargs='+', type=int, default=(324, 324),
                        help="Crop size. Enter '--crop-size w h' ")

    parser.add_argument("--loss-type", type=str, default='cce',
                        help="Loss to use. Options: 'cce', 'cce_soft'. ")

    parser.add_argument("--show-model", action='store_true',
                        help="Show number of parameters in the model and the"
                             " model itself.")
    parser.add_argument("--save-test-images", action='store_true',
                        help="Save predictions while evaluating with "
                             "'--test True'.")
    parser.add_argument("--train", action='store_true',
                        help="Train the model.")

    parser.add_argument("--test", action='store_false',
                        help="Evaulate either in the "
                             "test or the validation set.")
    parser.add_argument("--test-set", type=str, default='test',
                        help="Select the test in which to evaluate the model:"
                             " 'test' or 'val'")

    parser.add_argument("--checkpointer", action='store_false',
                        help="If True, training resumes from the 'last epoch'"
                             " weights found in the experiment folder. Useful "
                             "to save experiment at each epoch. If False,"
                             " training starts from scratch.")

    return parser.parse_args()


def save_arguments(args):
    print_args = {}
    param_names = [elem for elem in
                   filter(lambda aname: not aname.startswith('_'), dir(args))]
    for name in param_names:
        print_args.update({name: getattr(args, name)})
        print('[' + name + ']   ' + str(getattr(args, name)))

    path = os.path.join(args.results_dir, args.exp_name, 'args.json')
    import json
    with open(path, 'w') as fp:
        json.dump(print_args, fp)
    print('Args saved in ' + path)
