import csv
import os


# import sys
# from distutils.dir_util import copy_tree

def create_log(experiment_dir, checkpointer, nclasses=11):
    open_mode = 'w'
    if os.path.isfile(
            os.path.join(experiment_dir, 'training.log')) and checkpointer:
        print('Appending to existing log file...')
        open_mode = 'a'
        csvfile = open(os.path.join(experiment_dir, 'training.log'), open_mode)
    else:
        print('Creating new log file...')
        csvfile = open(os.path.join(experiment_dir, 'training.log'), open_mode)
        log_writer = csv.writer(csvfile, delimiter=',')
        # Write header to log file
        header = ['epoch']
        for class_id in range(nclasses):
            header.append(str(class_id) + 'train_jacc_percl')
        header.append('loss')
        header.append('acc')
        header.append('jaccard')
        for class_id in range(nclasses):
            header.append(str(class_id) + 'val_jacc_percl')
        header.append('val_loss')
        header.append('val_acc')
        header.append('val_jaccard')
        #     header+='\n'
        log_writer.writerow(header)
    return csvfile


def write_to_log(csvfile, epoch, jaccard_per_class_tr, train_loss, accuracy_tr,
                 jaccard_tr, jaccard_per_class, val_loss,
                 accuracy, jaccard):
    log_line = [str(epoch)]
    for elem in jaccard_per_class_tr:
        log_line.append(str(elem))
    log_line.append(str(train_loss))
    log_line.append(str(accuracy_tr))
    log_line.append(str(jaccard_tr))
    for elem in jaccard_per_class:
        log_line.append(str(elem))
    log_line.append(str(val_loss))
    log_line.append(str(accuracy))
    log_line.append(str(jaccard))
    log_writer = csv.writer(csvfile, delimiter=',')
    log_writer.writerow(log_line)


def print_metrics(loss, nbatches, n_classes, jaccard_per_class, jaccard,
                  accuracy):
    metrics_string = ' Loss: %.3f | Metrics: ' % (loss / (nbatches))
    for i in range(n_classes):
        metrics_string += 'jacc_class%i:  %.3f -' % (i, jaccard_per_class[i])
    metrics_string += ' mean_jacc: %.3f' % jaccard
    metrics_string += ' accuracy %.3f' % accuracy
    return metrics_string

# def signal_term_handler_(csvfile, experiment_dir_final, experiment_dir):
#     print('got SIGTERM')
#     csvfile.close()
#     print('SLURM CANCELED. Copying checkpoint and log to {}'.format(
#         experiment_dir_final))
#     if experiment_dir_final != experiment_dir:
#         if not os.path.exists(experiment_dir_final):
#             os.makedirs(experiment_dir_final)
#         copy_tree(experiment_dir, experiment_dir_final)
#     sys.exit(0)
#
