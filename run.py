import os
import sys
import random
from distutils.dir_util import copy_tree

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.fc_drn_model import DenseResNet
from utils import parser
from utils.logging import create_log, write_to_log, print_metrics
from utils.script_utils import load_checkpoint, save_checkpoints, cce_soft, \
    setup_net_params, create_optimizer, \
    adjust_opt
from utils.utils import progress_bar, confusion_matrix, compute_metrics, \
    save_images
from utils.data_loader import load_data

sys.path.append("../")

rnd_seed = 1609
torch.manual_seed(rnd_seed)
random.seed(rnd_seed)
np.random.seed(rnd_seed)


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def main():
    args = parser.get_arguments()
    setattr(args, 'crop_size', tuple(getattr(args, 'crop_size')))

    # ------ Create experiment folder(s)  ------#
    experiment_dir_final = os.path.join(args.results_dir_final, args.exp_name)
    check_mkdir(experiment_dir_final)

    if args.results_dir != args.results_dir_final:
        experiment_dir = os.path.join(args.results_dir, args.exp_name)
        check_mkdir(experiment_dir)
    else:
        experiment_dir = experiment_dir_final

    # ------ Print and save arguments in experiment folder  ------#
    parser.save_arguments(args)

    # ------ Create log file and write headers  ------#
    csvfile = create_log(experiment_dir, args.checkpointer)

    # -----Load data  ------#
    train_iter, val_iter, test_iter = load_data(args.dataset, args.crop_size,
                                                one_hot=True,
                                                batch_size=[
                                                    args.train_batch_size,
                                                    args.val_batch_size,
                                                    args.val_batch_size])

    # ------ Create model  ------#
    resnet_config, dilation_list, growth_rate, transformation, \
        n_filters_inout, filter_size_inout, subsample_inout, n_init_conv, \
        dropout, preprocess_block = setup_net_params(args.model)

    net = DenseResNet(input_channels=train_iter.data_shape[2],
                      n_init_conv=n_init_conv, subsample_inout=subsample_inout,
                      n_filters_inout=n_filters_inout,
                      n_classes=train_iter.non_void_nclasses,
                      dilation_list=dilation_list, resnet_config=resnet_config,
                      growth_rate=growth_rate,
                      filter_size_inout=filter_size_inout, dropout=dropout,
                      mixing_conv=True,
                      transformation=transformation,
                      preprocess_block=preprocess_block, bn_momentum=0.01,
                      ini=args.init_dils)
    net = net.cuda()
    if args.show_model:
        print(net)
        print('  + Number of params: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

    # ------ Create optimizer  ------#
    opt = create_optimizer(args.optimizer, net, args.learning_rate,
                           args.weight_decay)

    # ------ Load checkpoints  ------#
    best_jacc, start_epoch = load_checkpoint(args.checkpointer,
                                             experiment_dir_final,
                                             experiment_dir, net, opt,
                                             args.load_weights,
                                             args.exp_name_toload)

    # ------ Loss function  ------#
    void_labels = train_iter.void_labels
    if len(void_labels) == 1:
        ind_ignore = void_labels[0]
    else:
        ind_ignore = -100
    loss_function = torch.nn.NLLLoss2d(ignore_index=ind_ignore)

    # ------ Signal if SIGTERM is received, so we can save the checkpoint
    #  (used for SLURM preemption)  ------#
    # signal.signal(signal.SIGTERM, signal_term_handler)
    # ------ Training loop  ------#
    if args.train:
        for epoch in range(start_epoch, start_epoch + args.epoch_num):
            # Early stopping step
            es_step = 0
            print('Epoch %i /%i' % (epoch, start_epoch + args.epoch_num))

            jaccard_tr, jaccard_per_class_tr, accuracy_tr, train_loss = train_(
                train_iter, net, opt, loss_function,
                args.loss_type, ind_ignore,
                n_classes=train_iter.non_void_nclasses)
            # Test in validation set
            es_step, best_jacc, code, jaccard, jaccard_per_class, accuracy,\
                val_loss = val_(val_iter, net, opt, loss_function,
                                args.loss_type, epoch, es_step, ind_ignore,
                                experiment_dir, args.patience, best_jacc,
                                n_classes=train_iter.non_void_nclasses)
            # Write lo log
            write_to_log(csvfile, epoch, jaccard_per_class_tr, train_loss,
                         accuracy_tr, jaccard_tr, jaccard_per_class, val_loss,
                         accuracy, jaccard)

            if code == 1:
                csvfile.close()

            # Lr scheduler
            adjust_opt(args.learning_rate, opt, epoch, lr_decay=0.995)

        csvfile.close()
    # ------ Test ------#
    if args.test:
        test_(test_iter, net, experiment_dir_final, loss_function,
              args.loss_type, void_labels, args.save_test_images,
              n_classes=train_iter.non_void_nclasses)

    # ------ Save results to final directory
    # if a temporal one was used ------#
    if experiment_dir != experiment_dir_final:
        print('Copying model and other training files to {}'.format(
            experiment_dir_final))
        if not os.path.exists(experiment_dir_final):
            os.makedirs(experiment_dir_final)
        copy_tree(experiment_dir, experiment_dir_final)


def train_(train_iter, net, opt, loss_function, loss_type, ind_ignore,
           n_classes):
    net.train()
    train_loss = 0
    total = 0
    # Create the confusion matrix
    cm = np.zeros((n_classes, n_classes))
    nTrain = train_iter.nbatches
    for batch_idx in range(nTrain):
        all_data = train_iter.next()
        data = all_data[0]
        target = all_data[1]

        data, target = data.transpose((0, 3, 1, 2)), target.transpose(
            (0, 3, 1, 2))
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        opt.zero_grad()

        output = net(data)
        target = target.type(torch.FloatTensor).cuda()

        _, target_indices = torch.max(target, 1)
        _, output_indices = torch.max(output, 1)
        flattened_output = output_indices.view(-1)
        flattened_target = target_indices.view(-1)

        if loss_type == 'cce_soft':
            loss = cce_soft(output, target, ignore_label=ind_ignore)
        else:
            loss = loss_function(output, target_indices)

        cm = confusion_matrix(cm, flattened_output.data.cpu().numpy(),
                              flattened_target.data.cpu().numpy(),
                              n_classes)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=4)
        opt.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)

        progress_bar(batch_idx, nTrain, 'Loss: %.3f'
                     % (train_loss / (batch_idx + 1)))

        del (output)
        del (loss)
        del (flattened_output)
        del (output_indices)

    jaccard_per_class, jaccard, accuracy = compute_metrics(cm)
    metrics_string = print_metrics(train_loss, nTrain, n_classes,
                                   jaccard_per_class, jaccard, accuracy)
    print(metrics_string)
    return jaccard, jaccard_per_class, accuracy, train_loss / (nTrain)


def val_(val_iter, net, opt, loss_function, loss_type, epoch, es_step,
         ind_ignore, experiment_dir, max_patience,
         best_jacc, n_classes):
    code = 0
    net.eval()
    test_loss = 0
    total = 0
    # Create the confusion matrix
    cm = np.zeros((n_classes, n_classes))
    nVal = val_iter.nbatches
    for batch_idx in range(nVal):
        all_data = val_iter.next()
        data = all_data[0]
        target = all_data[1]

        data, target = data.transpose((0, 3, 1, 2)), target.transpose(
            (0, 3, 1, 2))
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = net(data)

        target = target.type(torch.FloatTensor).cuda()
        _, target_indices = torch.max(target, 1)
        _, output_indices = torch.max(output, 1)
        flattened_output = output_indices.view(-1)
        flattened_target = target_indices.view(-1)

        if loss_type == 'cce_soft':
            loss = cce_soft(output, target, ignore_label=ind_ignore)
        else:
            loss = loss_function(output, target_indices)

        cm = confusion_matrix(cm, flattened_output.data.cpu().numpy(),
                              flattened_target.data.cpu().numpy(),
                              n_classes)
        test_loss += loss.data[0]
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)

        progress_bar(batch_idx, val_iter.nbatches, 'Val loss: %.3f'
                     % (test_loss / (batch_idx + 1)))

        del (output)
        del (loss)
        del (flattened_output)
        del (output_indices)

    jaccard_per_class, jaccard, accuracy = compute_metrics(cm)

    metrics_string = print_metrics(test_loss, nVal, n_classes,
                                   jaccard_per_class, jaccard, accuracy)
    print(metrics_string)

    es_step, best_jacc = save_checkpoints(jaccard, net, epoch, opt,
                                          experiment_dir, best_jacc, es_step)

    # Early stopping
    if es_step >= max_patience:
        print('Early stopping! Max mean jaccard: ' + str(best_jacc))
        code = 1
    return es_step, best_jacc, code, jaccard, jaccard_per_class, accuracy, \
        test_loss / (nVal)


def test_(test_iter, net, experiment_dir_final, loss_function, loss_type,
          void_labels, save_test_images, n_classes):
    ckt_names = ['best_jaccard.t7']

    for ckt_name in ckt_names:
        print('Testing checkpoint ' + ckt_name)
        checkpoint = torch.load(
            os.path.join(experiment_dir_final, 'checkpoint', ckt_name))
        print('Checkpoint loaded for testing...')
        net.load_state_dict(checkpoint['net'])

        net.eval()
        test_loss = 0
        total = 0
        # Create the confusion matrix
        cm = np.zeros((n_classes, n_classes))
        nTest = test_iter.nbatches
        for batch_idx in range(nTest):
            all_data = test_iter.next()
            data_ = all_data[0]
            target_ = all_data[1]

            data, target = data_.transpose((0, 3, 1, 2)), target_.transpose(
                (0, 3, 1, 2))
            data, target = torch.from_numpy(data), torch.from_numpy(target)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = net(data)

            target = target.type(torch.LongTensor).cuda()
            _, target_indices = torch.max(target, 1)
            _, output_indices = torch.max(output, 1)
            flattened_output = output_indices.view(-1)
            flattened_target = target_indices.view(-1)

            loss = loss_function(output, target_indices)

            cm = confusion_matrix(cm, flattened_output.data.cpu().numpy(),
                                  flattened_target.data.cpu().numpy(),
                                  n_classes)

            test_loss += loss.data[0]
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)

            progress_bar(batch_idx, test_iter.nbatches,
                         'Test loss: %.3f' % (test_loss / (batch_idx + 1)))

            if save_test_images:
                save_images(data_, target_, output, experiment_dir_final,
                            batch_idx, void_labels)

            del (output)
            del (loss)
            del (flattened_output)
            del (output_indices)

        jaccard_per_class, jaccard, accuracy = compute_metrics(cm)
        metrics_string = print_metrics(test_loss, nTest, n_classes,
                                       jaccard_per_class, jaccard, accuracy)
        print(metrics_string)


if __name__ == '__main__':
    main()
