'''
    - progress_bar: progress bar mimic xlua.progress.
    Code from  https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py


'''
import os
import sys
import time
import numpy as np
import scipy.misc

channel_idx = 3


def confusion_matrix(cm, output_flatten, target_flatten, num_classes):
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] += ((output_flatten == i) * (target_flatten == j)).sum()
    return cm


def compute_metrics(cm):
    # Compute metrics
    TP_perclass = cm.diagonal().astype('float32')
    jaccard_perclass = np.where((cm.sum(1) + cm.sum(0) - TP_perclass) != 0.,
                                TP_perclass / (cm.sum(1) + cm.sum(
                                    0) - TP_perclass), 0.)
    jaccard = np.mean(jaccard_perclass)
    accuracy = TP_perclass.sum() / cm.sum()

    return jaccard_perclass, jaccard, accuracy


#  _, term_width = os.popen('stty size', 'r').read().split()
#  term_width = int(term_width)
term_width = 110

TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# Converts a label mask to RGB to be shown
def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


# Converts a label mask to RGB to be shown and overlaps over an image
def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = image
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_images(batch_data, target, prediction, experiment_dir_final,
                batch_idx, void_labels):
    color_map = [(128, 128, 128), (128, 0, 0), (192, 192, 128),
                 # sky, building, column_pole
                 (128, 64, 128), (0, 0, 192), (128, 128, 0),
                 # road, sidewalk, Tree
                 (192, 128, 128), (64, 64, 128), (64, 0, 128),
                 # SignSymbol, Fence, Car
                 (64, 64, 0), (0, 128, 192), (0, 0, 0), (128, 128, 128),
                 (128, 0, 0), (192, 192, 128),
                 # sky, building, column_pole
                 (128, 64, 128), (0, 0, 192), (128, 128, 0),
                 # road, sidewalk, Tree
                 (192, 128, 128), (64, 64, 128), (64, 0, 128),
                 # SignSymbol, Fence, Car
                 (64, 64, 0), (0, 128, 192),
                 (0, 0, 0)]  # Pedestrian, Byciclist, Void
    save_img(image_batch=batch_data,
             mask_batch=np.argmax(target, 3),
             output=np.argmax(prediction.data.cpu().numpy(), 1),
             out_images_folder=os.path.join(experiment_dir_final,
                                            "predictions_best_weights"),
             epoch=0,
             color_map=color_map,
             tag="test_batch_" + str(batch_idx) + "_", void_label=void_labels)


# Save images
def save_img(image_batch, mask_batch, output, out_images_folder, epoch,
             color_map, tag, void_label):
    if image_batch.min() < -255 or image_batch.max() > 255:
        raise ValueError("The input image has pixels with values exceeding "
                         "the range [-255, 255]")

    if not os.path.exists(out_images_folder):
        os.makedirs(out_images_folder)

    if len(void_label):
        output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in range(output.shape[0]):

        img = (image_batch[j] * 255)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.dstack([np.squeeze(img, -1), np.squeeze(img, -1),
                             np.squeeze(img, -1)])
        label_out = my_label2rgb(output[j], bglabel=void_label,
                                 colors=color_map)
        label_mask = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                         image=img, bglabel=void_label,
                                         alpha=0.2)
        label_overlay = my_label2rgboverlay(output[j], colors=color_map,
                                            image=img, bglabel=void_label,
                                            alpha=0.5)

        combined_image = np.concatenate((img, label_mask, label_out,
                                         label_overlay), axis=1)
        out_name = os.path.join(out_images_folder,
                                "{}_epoch{}_img{}.png"
                                .format(tag, str(epoch), str(j)))
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)

    return images
