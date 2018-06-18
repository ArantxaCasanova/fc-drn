import os
import torch
import torch.optim as optim
from torch.nn import MaxPool2d, AvgPool2d, Parameter
from models.fc_drn_model import PreprocessBlockBottleMg, \
    PreprocessBlockStandard
from distutils.dir_util import copy_tree


def save_checkpoints(jacc, net, epoch, opt, experiment_dir, best_jacc,
                     es_step):
    # Save checkpoint every epoch and for best mean jaccard
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'jacc': jacc,
        'epoch': epoch,
        'opt': opt.state_dict()
    }
    if not os.path.exists(os.path.join(experiment_dir, 'checkpoint')):
        os.makedirs(os.path.join(experiment_dir, 'checkpoint'))
    if jacc > best_jacc:
        torch.save(state, os.path.join(experiment_dir, 'checkpoint',
                                       'best_jaccard.t7'))
        best_jacc = jacc
        es_step = 0
    else:
        es_step += 1
    # Save every epoch
    torch.save(state,
               os.path.join(experiment_dir, 'checkpoint', 'last_epoch.t7'))

    return es_step, best_jacc


def cce_soft(input, target, ignore_label):
    _, indices = torch.max(target, 1)
    void_indices = indices != ignore_label
    target = target * 0.89
    target = target + 0.01
    return torch.mean(
        torch.sum((-target[:, 0:-1, :, :] * input), dim=1)[void_indices])


def setup_net_params(model):
    resnet_config = [7] * 9
    if model in ['pools', 'sconv']:
        dilation_list = [1] * 9
        preprocess_block = None
        if model == 'pools':
            transformation = [MaxPool2d, MaxPool2d, MaxPool2d, MaxPool2d,
                              'upsample', 'upsample', 'upsample',
                              'upsample']
        else:
            transformation = ['sconv', 'sconv', 'sconv', 'sconv', 'upsample',
                              'upsample', 'upsample',
                              'upsample']
    elif model in ['pools_ft_dils', 'sconv_ft_dils']:
        preprocess_block = PreprocessBlockBottleMg
        dilation_list = [1, 1, 2, 4, 1, 1, 1, 1]
        if model == 'pools_ft_dils':
            transformation = [MaxPool2d, MaxPool2d, 'dilation_mg',
                              'dilation_mg', 'dilation', 'dilation',
                              'upsample',
                              'upsample']
        else:
            transformation = ['sconv', 'sconv', 'dilation_mg', 'dilation_mg',
                              'dilation', 'dilation', 'upsample',
                              'upsample']
    elif model in ['dils']:
        preprocess_block = PreprocessBlockBottleMg
        dilation_list = [2, 4, 16, 32, 1, 1, 1, 1]
        transformation = ['dilation', 'dilation', 'dilation', 'dilation',
                          'dilation', 'dilation', 'dilation',
                          'dilation']
    else:
        raise ValueError('Transformation type not implemented.')
    growth_rate = [30, 40, 40, 40, 50, 40, 40, 40, 30]

    n_filters_inout = 50
    filter_size_inout = 3
    subsample_inout = (2, 2)
    n_init_conv = 3
    dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    return resnet_config, dilation_list, growth_rate, transformation,\
        n_filters_inout, filter_size_inout, subsample_inout, n_init_conv, \
        dropout, preprocess_block


def create_optimizer(optimizer, net, learning_rate, weight_decay):
    if optimizer == 'sgd':
        opt = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        opt = optim.Adam(net.parameters(), lr=learning_rate,
                         weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        opt = optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9,
                            weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not recognized')
    return opt


def load_checkpoint(checkpointer, experiment_dir_final, experiment_dir, net,
                    opt, load_weights, exp_name_toload):
    best_jacc = 0
    start_epoch = 0
    # Load checkpoint.
    if checkpointer:
        print('==> Resuming from checkpoint..')
        if not os.path.isfile(os.path.join(experiment_dir_final, 'checkpoint',
                                           'last_epoch.t7')):
            print(
                "Warning: no 'last_epoch.ty' checkpoint found!,"
                " Starting from scratch...")
        else:
            if experiment_dir_final != experiment_dir:
                print('Copying experiment folder to TMP to resume training.')
                copy_tree(experiment_dir_final, experiment_dir)
            checkpoint = torch.load(
                os.path.join(experiment_dir, 'checkpoint', 'last_epoch.t7'))
            print('(Checkpointer): Checkpoint loaded. Resuming training...')
            net.load_state_dict(checkpoint['net'])
            best_jacc = checkpoint['jacc']
            start_epoch = checkpoint['epoch']
            opt.load_state_dict(checkpoint['opt'])
    else:
        print('No checkpointing option. Starting from scratch...')

    if load_weights:
        checkpoint = torch.load(
            os.path.join(exp_name_toload, 'checkpoint', 'best_jaccard.t7'))
        print(
            '(Load checkpoint from other folder): Checkpoint loaded. '
            'Resuming training...')
        pretrained_dict = checkpoint['net']
        print('  + Number of params network before loading: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        net.load_state_dict(model_dict)
        print('  + Number of params network after loading: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

    return best_jacc, start_epoch


def adjust_opt(lr, optimizer, epoch, lr_decay=0.995):
    new_lr = lr * (lr_decay ** epoch)
    print('Scheduler LR: new lr is ' + str(new_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
