import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform, _calculate_correct_fan, \
    calculate_gain, dirac


class InitialConv(nn.Module):
    def __init__(self, in_channels, n_filters, filter_size, n_init_conv,
                 subsample=(2, 2), bias=True):
        super(InitialConv, self).__init__()
        """
        Creates an IDB (Initial Downsampling Block), the first block of
        the FC-DRN architecture.
        It is composed of:
             1) Convolution + ReLU activation
             2) Max pooling
             3) N times (specified with 'n_init_conv-1'), Convolution +
                ReLU activation.

        Input:
            - in_channels: int. Input image channels. Example: 3 if it
                           is RGB or 1 if it is grayscale.
            - n_filters: int. Number of channels for all convolutions.
            - filter_size: int. Size of convolution filters.
            - n_init_conv: int. Number of convolutions in IDB. After the
                           first one there is always a pooling.
            - subsample: int or tuple. Specifies the stride of the
                         pooling.
            - bias: bool. Use bias in convolutions.
        """
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=filter_size,
                               padding=(filter_size - 1) // 2, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=subsample,
                                 ceil_mode=True)
        self.additional_convs = self._make_layer(n_init_conv - 1, n_filters,
                                                 filter_size, bias)
        self.n_init_conv = n_init_conv
        self.subsample = subsample

        # Initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_conv_blocks, n_filters, filter_size, bias):
        """
        It builds 'num_conv_blocks' repetitions of: Convolution + ReLU
        activation. Last block does not have ReLU.
        """
        layers = []
        for n in range(num_conv_blocks):
            layers.append(
                nn.Conv2d(n_filters, n_filters, kernel_size=filter_size,
                          padding=(filter_size - 1) // 2, bias=bias))
            if n < num_conv_blocks - 1:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        if self.subsample[0] > 1 or self.subsample[1] > 1:
            out = self.pool(out)
        if self.n_init_conv > 1:
            out = self.additional_convs(out)
        return out


class PreprocessBlockStandard(nn.Module):
    def __init__(self, nb_filter, dropout=0., dilation=1, bias=True,
                 bn_momentum=0.1, ini='random'):
        super(PreprocessBlockStandard, self).__init__()
        """
        It creates a dilated convolution block in the shape of:
        Batch normalization, Dropout, ReLU activation, dilated
        convolution.
        It is used as a transformation in the FC-DRN-D.
        It does not change number of input feature maps.

        Input:
            - nb_filter: int. Number of input feature maps.
            - dropout: float. Percentage of dropout.
            - dilation: int. Dilation rate for the dilated convolution.
            - bias: bool. Bias in convolution.
            - bn_momentum: float. Batch-norm momentum.
            - ini: string. Initialization for the dilated convolution
                   weights. It can be 'random' or 'identity'.
        """
        self.std_block = BnReluConv(nb_filter, nb_filter, 3, dropout,
                                    bias=bias, dilation=dilation,
                                    bn_momentum=bn_momentum, ini=ini)

    def forward(self, x, crop_size=None):
        return self.std_block(x)


class PreprocessBlockBottleMg(nn.Module):
    def __init__(self, nb_filter, dropout=0., dilation=1, bias=True,
                 bn_momentum=0.1, mg=[1, 2, 1], ini='random'):
        super(PreprocessBlockBottleMg, self).__init__()
        """
        It creates a multi-grid dilated convolution block.
        If the dilation rate is 1 (traditional convolution), it only
        creates one block of:
        Batch normalization, Dropout, ReLU activation, convolution.
        If the dilation rate is bigger than 1, it creates 3 blocks of
        BN + Dropout + ReLu + Dilated convolution with dilation rates
        [mg[0]*dilation, mg[1]*dilation, mg[2]*dilation] respectively.
        It can be used as a transformation in the FC-DRN-D and for the
        finetunned architectures variants (FC-DRN-P-D and FC-DRN-S-D).
        It does not change number of input feature maps.

        Input:
            - nb_filter: int. Number of input feature maps.
            - dropout: float. Percentage of dropout.
            - dilation: int. Dilation rate for the dilated convolution.
            - bias: bool. Bias in convolution.
            - bn_momentum: float. Batch-norm momentum.
            - mg: int list. Each of the positions is a multiplier for
                  the dilation rate used in each of the 3 convolutions.
            - ini: string. Initialization for the dilated convolution
                   weights. It can be 'random' or 'identity'.
        """
        self.dil_factor = dilation
        if self.dil_factor == 1:
            self.no_dil = BnReluConv(nb_filter, nb_filter, 3, dropout,
                                     bias=bias, dilation=1,
                                     bn_momentum=bn_momentum, ini=ini)
        else:
            self.std_block = BnReluConv(nb_filter, nb_filter, 3, dropout,
                                        bias=bias,
                                        dilation=mg[0] * self.dil_factor,
                                        bn_momentum=bn_momentum, ini=ini)
            self.std_block2 = BnReluConv(nb_filter, nb_filter, 3, dropout,
                                         bias=bias,
                                         dilation=mg[1] * self.dil_factor,
                                         bn_momentum=bn_momentum, ini=ini)
            self.std_block3 = BnReluConv(nb_filter, nb_filter, 3, dropout,
                                         bias=bias,
                                         dilation=mg[2] * self.dil_factor,
                                         bn_momentum=bn_momentum, ini=ini)

    def forward(self, x, crop_size=None):
        if self.dil_factor == 1:
            return self.no_dil(x)
        else:
            out = self.std_block(x)
            out = self.std_block2(out)
            out = self.std_block3(out)
            return out


class BnReluConv(nn.Module):
    def __init__(self, input_channels, n_filters, filter_size, dropout,
                 bias=True, dilation=1, stride=(1, 1), bn_momentum=0.1,
                 ini='random'):
        super(BnReluConv, self).__init__()
        """
        It builds a block with: Batch Norm, Dropout, ReLU, Convolution.

        Input:
            - input_channels: int. Number of input feature maps.
            - n_filters: int. Number of output feature maps.
            - filter_size: int. Convolution filter size.
            - dropout: float. Percentage of dropout.
            - bias: bool. Bias in convolution.
            - dilation: int. Dilation rate for dilated convolution.
                        If 1, traditional convolution is used.
            - stride: int or tuple. Stride used in the convolution.
            - bn_momentum: float. Batch-norm momentum.
            - ini: string. Initialization for the dilated convolution
                   weights. It can be 'random' or 'identity'.
        """
        self.bn = nn.BatchNorm2d(input_channels, eps=0.001,
                                 momentum=bn_momentum)
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        if dilation == 1:
            self.conv = nn.Conv2d(input_channels, n_filters,
                                  kernel_size=filter_size,
                                  padding=(filter_size - 1) // 2, bias=bias,
                                  stride=stride)
            # Initialize modules
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_uniform(m.weight)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        # In the case where we want to use dilated convolutions
        # in the transformation blocks between ResNets
        else:

            self.conv = nn.Conv2d(input_channels, n_filters,
                                  kernel_size=filter_size, dilation=dilation,
                                  padding=((filter_size + (filter_size - 1) * (
                                              dilation - 1)) - 1) // 2,
                                  bias=bias)
            # Initialize modules
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if ini == 'identity':
                        dirac(m.weight)
                    else:
                        kaiming_uniform(m.weight)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.bn(x))
        if self.dropout > 0:
            out = self.drop(out)
        out = self.conv(out)
        return out


class Upsample_conv(nn.Module):
    def __init__(self, input_channels, n_filters, filter_size=3, bias=True):
        super(Upsample_conv, self).__init__()
        """
        This class contains 3 operations:
            1) Upsampling function with scale factor 2.
            2) Crop layer to make sure that the size after the
             upsampling matches the desired size.
            3) Convolution to smooth the upsampling

        Input:
            - input_channels: int. Number of input feature maps.
            - n_filters: int. Number of output feature maps.
            - filter_size: int. Filter size of the smoothing convolution.
            - bias: bool. Bias in the convolution.
        """
        self.up_conv = nn.Conv2d(input_channels, n_filters,
                                 kernel_size=filter_size,
                                 padding=(filter_size - 1) // 2,
                                 bias=bias)
        # Initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def crop_layer(self, layer, target_size):
        '''
        This layer crops the spatial size of the input feature maps.
        Input:
            layer: torch Variable. Input to be cropped.
            target_size: tuple or list with 2 positions. Size at the end
                         of the crop.
        '''
        dif = [(layer.shape[2] - target_size[0]) // 2,
               (layer.shape[3] - target_size[1]) // 2]
        cs = target_size
        return layer[:, :, dif[0]:dif[0] + cs[0], dif[1]:dif[1] + cs[1]]

    def forward(self, x, size):
        out = F.upsample(x, scale_factor=2)
        out = self.crop_layer(out, [size[0], size[1]])
        out = self.up_conv(out)
        return out


class Classifier(nn.Module):
    def __init__(self, input_channels, n_classes, bias=True, logsoftmax=True):
        super(Classifier, self).__init__()
        """
        A classifier of 1x1 convolution and Softmax function.

        Input:
            - input_channels: int. Input feature maps.
            - n_classes: int. Number of classes as output feature maps.
            - bias: bool. Bias in the convolution.
            - logsoftmax: bool. If True, LogSoftmax function is used
                          instead of Softmax.
        """
        self.cl_conv = nn.Conv2d(input_channels, n_classes, kernel_size=1,
                                 bias=bias)
        self.logsoftmax = logsoftmax

        if logsoftmax:
            self.softmax = torch.nn.LogSoftmax(dim=1)
        else:
            self.softmax = torch.nn.Softmax(dim=1)

        # Initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.cl_conv(x)
        out = self.softmax(out)
        if not self.logsoftmax:
            out = out.clamp(min=1e-7)
            out = out.log_()
        return out


class BasicBlock(nn.Module):
    def __init__(self, input_channels, n_filters, filter_size=3, dropout=0.,
                 dilation=1, bias=True, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        """
        This is a ResNet basic block, composed of 2 sets of:
        batch normalization, dropout, ReLU, convolution.
        Output y is computed as:
        y = F(x) + x, where F() is the BasicBlock.

        If the channels of x and F(x) are not equal,
        a 1x1 convolution is used on x to match the output channels.

        Input:
            - input_channels: int. Number of input feature maps.
            - n_filters: int. Number of output feature maps.
            - filter_size: int. Filter size of the convolution.
            - dropout: float. Dropout probability.
            - dilation: int. Dilation factor for the dilated
                        convolution. It is only applied to the first
                        set of BN + Dropout + ReLU + conv.
            - bias: bool. Bias of the convolution.
            - bn_momentum: float. Batch normalization momentum.
        """

        self.unit1 = BnReluConv(input_channels, n_filters, filter_size,
                                dropout, bias, dilation,
                                bn_momentum=bn_momentum)
        self.unit2 = BnReluConv(n_filters, n_filters, filter_size, dropout,
                                bias, 1, bn_momentum=bn_momentum)

        if input_channels != n_filters:
            self.equal_channels_conv = nn.Conv2d(input_channels, n_filters,
                                                 kernel_size=1, bias=bias)
        self.input_channels = input_channels
        self.n_filters = n_filters

        # Initialize modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_uniform(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        processed = self.unit2(self.unit1(x))
        if self.input_channels != self.n_filters:
            x = self.equal_channels_conv(x)

        out = x + processed
        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, n_filters, dropout, resnet_layers,
                 filter_size=3, bias=True, bn_momentum=0.1):
        super(ResNet, self).__init__()
        """
        This class builds a ResNet with BasicBlocks.
        Input:
            - input_channels: int. Number of input feature maps.
            - n_filters: int. Number of output feature maps.
            - dropout: float. Dropout probability.
            - resnet_layers: int. Number of BasicBlocks to build.
            - filter_size: int. Filter size of the convolution.
            - bias: bool. Bias of the convolution.
            - bn_momentum: float. Batch normalization momentum.
        """
        self.resnet = self._make_layer(input_channels, n_filters, filter_size,
                                       dropout, resnet_layers, bias,
                                       bn_momentum=bn_momentum)

    def _make_layer(self, input_channels, n_filters, filter_size, dropout,
                    resnet_layers, bias, bn_momentum=0.1):
        layers = []
        for i in range(resnet_layers):
            if i == 0:
                resnet_in_channels = input_channels
            else:
                resnet_in_channels = n_filters

            layers.append(
                BasicBlock(resnet_in_channels, n_filters, filter_size,
                           dropout, 1, bias, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.resnet(x)
        return out


class DenseResBlock(nn.Module):
    def __init__(self, dilation_list, resnet_config, growth_rate,
                 dropout, n_filters_inout, bias=True,
                 mixing_conv=True,
                 stride=2, transformation=None,
                 preprocess_block=PreprocessBlockStandard, bn_momentum=0.1,
                 ini='random'):
        super(DenseResBlock, self).__init__()
        """
        This class builds the Dense Block between IDB and FUB.
        It is composed of several ResNets, with transformations and
        1x1 mixing convolutions.
        The transformations can be: max pooling, strided/dilated/
        traditional convolution, upsampling.
        The code is made to work with 9 ResNets, with 8 transformations
        in between.
        Input:
            - dilation_list: int list of length N-1. It indicates the dilation
                             rate to use in each transformation.
                             Each position indicates each different
                             transformation. Example: [2, 1] will use
                             dilation rate 2 for the first transfor-
                             mation if the first position of
                             the 'transformation' list is a dilated conv.
                             Otherwise, dilation rate does not affect
                             the transformation.
            - resnet_config: int list of length N. It creates N ResNets
                             with M BasicBlocks, where M is the value of
                             the list at each position.
                             Example: [4, 4, 4] creates 3 ResNets with 4
                             basic blocks each.
            - growth_rate: int list of length N. Specifies the number of
                           feature maps to output per each of the N
                           ResNets.
            - dropout: float list of length N. Specifies the dropout rate
                       per each ResNet.
            - n_filters_inout: int. Number of input feature maps to the
                               DenseBlock.
            - bias: bool. Bias of the convolution.
            - mixing_conv: bool. To use the 1x1 mixing convolution after
                           each concatenation.
            - stride: int. Stride to use when using strided convolutions
                      as transformations.
            - transformation: string list of length N-1. Specifies the
                              transformation to use between each ResNet.
            - preprocess_block: Class. Which type of dilated convolution
                                block to use when using 'dilation' as
                                transformation.
            - bn_momentum: float. Batch normalization momentum.
            - ini: string. Initilization for dilated convolutions. It
                   can be 'random' or 'identity'.
        """

        # Names of paths to transform. All the paths are transformed
        # separately. Example: in ResNet2, we will need to transform
        # output of ResNet1 AND input to the DenseResBlock.
        #  The ones that can be reused for further ResNets and do not
        # need additional transformations  are just directly used when
        # necessary.
        path_names = [['input_p1', 'resnet1_p1'],  # Input to ResNet2
                      ['input_p2', 'resnet1_p2', 'resnet2_p1'],  # Input to
                      # ResNet3
                      ['input_p3', 'resnet1_p3', 'resnet2_p2', 'resnet3_p1'],
                      #  Input to ResNet4
                      ['input_p4', 'resnet1_p4', 'resnet2_p3', 'resnet3_p2',
                       'resnet4_p1'],  # Input to ResNet5
                      ['resnet5_u1'],  # Input to ResNet6
                      ['resnet4_u1', 'resnet5_u2', 'resnet6_u1'],
                      # Input to ResNet7
                      ['resnet3_u1', 'resnet4_u2', 'resnet5_u3', 'resnet6_u2',
                       'resnet7_u1'],  # Input to ResNet8
                      ['resnet2_u1', 'resnet3_u2', 'resnet4_u3', 'resnet5_u4',
                       'resnet6_u3', 'resnet7_u2', 'resnet8_u1']
                      # Input to ResNet9
                      ]
        n_filters_conv = 0
        for index in range(len(resnet_config)):
            if index == 0:
                resnet_ch_in = n_filters_inout
            else:
                resnet_ch_in = n_filters_conv

            # ------------ Build ResNet ------------ #
            resnet = ResNet(resnet_ch_in, growth_rate[index], dropout[index],
                            resnet_config[index],
                            bias=bias, bn_momentum=bn_momentum)
            setattr(self, 'resnet' + str(index + 1), resnet)

            # ------------ Transform all inputs ------------ #
            if index < len(resnet_config) - 1:
                # Do not append transformations for the last ResNet.
                for k, p_name in enumerate(path_names[index]):
                    if 'input' in p_name:
                        num_ch = n_filters_inout
                        drop_index = 0
                    else:
                        num_ch = growth_rate[int(p_name[6]) - 1]
                        drop_index = int(p_name[6]) - 1

                    if transformation[index] == nn.MaxPool2d or \
                            transformation[index] == nn.AvgPool2d:
                        t_block = transformation[index](kernel_size=(2, 2),
                                                        stride=(
                                                        stride, stride),
                                                        ceil_mode=True)
                    elif transformation[index] == 'sconv':
                        t_block = BnReluConv(num_ch, num_ch, 3,
                                             dropout[drop_index], bias=bias,
                                             stride=(2, 2),
                                             bn_momentum=bn_momentum)
                    elif transformation[index] == 'upsample':
                        t_block = Upsample_conv(num_ch, num_ch, filter_size=3,
                                                bias=bias)
                    elif transformation[index] == 'dilation':
                        t_block = preprocess_block(num_ch, dropout[index],
                                                   dilation_list[index], bias,
                                                   ini=ini)
                    elif transformation[index] == 'dilation_mg':
                        t_block_pre = BnReluConv(num_ch, num_ch, 3,
                                                 dropout[drop_index],
                                                 bias=bias,
                                                 dilation=dilation_list[
                                                       index],
                                                 bn_momentum=bn_momentum,
                                                 ini=ini)
                        t_block = BnReluConv(num_ch, num_ch, 3,
                                             dropout[drop_index], bias=bias,
                                             dilation=2 * dilation_list[
                                                   index],
                                             bn_momentum=bn_momentum,
                                             ini=ini)
                        t_block_post = BnReluConv(num_ch, num_ch, 3,
                                                  dropout[drop_index],
                                                  bias=bias,
                                                  dilation=dilation_list[
                                                        index],
                                                  bn_momentum=bn_momentum,
                                                  ini=ini)
                    else:
                        raise ValueError('Transformation {} for ResNet' + str(
                            index + 1) + ' not understood'.format(
                            transformation[index]))

                    if transformation[index] == 'dilation_mg':
                        setattr(self, p_name + '_pre', t_block_pre)
                        setattr(self, p_name, t_block)
                        setattr(self, p_name + '_post', t_block_post)
                    else:
                        setattr(self, p_name, t_block)

            # ------------ Mixing convolution 1x1 ------------ #
            n_filters_conv = n_filters_inout + sum(growth_rate[:index + 1])
            if mixing_conv:
                mixing = BnReluConv(n_filters_conv, n_filters_conv,
                                    filter_size=1, dropout=dropout[index],
                                    bias=bias, dilation=1,
                                    bn_momentum=bn_momentum)
                setattr(self, 'mixing_r' + str(index + 1), mixing)

        self.transformation = transformation
        self.mixing = mixing_conv

    def forward(self, x):
        res1 = self.resnet1(x)
        in_p1 = self.input_p1(x)
        res1_p1 = self.resnet1_p1(res1)
        res2_in = torch.cat((in_p1, res1_p1), 1)
        if self.mixing:
            res2_in = self.mixing_r1(res2_in)

        res2 = self.resnet2(res2_in)

        in_p2 = self.input_p2(in_p1)
        res1_p2 = self.resnet1_p2(res1_p1)
        res2_p1 = self.resnet2_p1(res2)
        res3_in = torch.cat((in_p2, res1_p2, res2_p1), 1)
        if self.mixing:
            res3_in = self.mixing_r2(res3_in)

        res3 = self.resnet3(res3_in)
        in_p3 = self.input_p3(in_p2)
        res1_p3 = self.resnet1_p3(res1_p2)
        res2_p2 = self.resnet2_p2(res2_p1)
        res3_p1 = self.resnet3_p1(res3)
        res4_in = torch.cat((in_p3, res1_p3, res2_p2, res3_p1), 1)
        if self.mixing:
            res4_in = self.mixing_r3(res4_in)

        res4 = self.resnet4(res4_in)
        in_p4 = self.input_p4(in_p3)
        res1_p4 = self.resnet1_p4(res1_p3)
        res2_p3 = self.resnet2_p3(res2_p2)
        res3_p2 = self.resnet3_p2(res3_p1)
        res4_p1 = self.resnet4_p1(res4)
        res5_in = torch.cat((in_p4, res1_p4, res2_p3, res3_p2, res4_p1), 1)
        if self.mixing:
            res5_in = self.mixing_r4(res5_in)

        res5 = self.resnet5(res5_in)
        res5_u1 = self.resnet5_u1(res5, res4.size()[2:])
        res6_in = torch.cat((in_p3, res1_p3, res2_p2, res3_p1, res4, res5_u1),
                            1)
        if self.mixing:
            res6_in = self.mixing_r5(res6_in)

        res6 = self.resnet6(res6_in)
        res4_u1 = self.resnet4_u1(res4, res3.size()[2:])
        res5_u2 = self.resnet5_u2(res5_u1, res3.size()[2:])
        res6_u1 = self.resnet6_u1(res6, res3.size()[2:])
        res7_in = torch.cat(
            (in_p2, res1_p2, res2_p1, res3, res4_u1, res5_u2, res6_u1), 1)
        if self.mixing:
            res7_in = self.mixing_r6(res7_in)

        res7 = self.resnet7(res7_in)
        res3_u1 = self.resnet3_u1(res3, res2.size()[2:])
        res4_u2 = self.resnet4_u2(res4_u1, res2.size()[2:])
        res5_u3 = self.resnet5_u3(res5_u2, res2.size()[2:])
        res6_u2 = self.resnet6_u2(res6_u1, res2.size()[2:])
        res7_u1 = self.resnet7_u1(res7, res2.size()[2:])
        res8_in = torch.cat((in_p1, res1_p1, res2, res3_u1, res4_u2, res5_u3,
                             res6_u2, res7_u1), 1)
        if self.mixing:
            res8_in = self.mixing_r7(res8_in)

        res8 = self.resnet8(res8_in)
        res2_u1 = self.resnet2_u1(res2, res1.size()[2:])
        res3_u2 = self.resnet3_u2(res3_u1, res1.size()[2:])
        res4_u3 = self.resnet4_u3(res4_u2, res1.size()[2:])
        res5_u4 = self.resnet5_u4(res5_u3, res1.size()[2:])
        res6_u3 = self.resnet6_u3(res6_u2, res1.size()[2:])
        res7_u2 = self.resnet7_u2(res7_u1, res1.size()[2:])
        res8_u1 = self.resnet8_u1(res8, res1.size()[2:])
        res9_in = torch.cat((x, res1, res2_u1, res3_u2, res4_u3, res5_u4,
                             res6_u3, res7_u2, res8_u1), 1)
        if self.mixing:
            res9_in = self.mixing_r8(res9_in)

        res9 = self.resnet9(res9_in)
        out = torch.cat((x, res1, res2_u1, res3_u2, res4_u3, res5_u4, res6_u3,
                         res7_u2, res8_u1, res9), 1)
        if self.mixing:
            out = self.mixing_r9(out)

        return out


class DenseResNet(nn.Module):
    def __init__(self, input_channels, n_init_conv, subsample_inout,
                 n_filters_inout, n_classes, dilation_list,
                 resnet_config, growth_rate,
                 filter_size_inout, stride=2, dropout=None, bias=True,
                 mixing_conv=False,
                 transformation=None,
                 preprocess_block=PreprocessBlockStandard, logsoftmax=True,
                 bn_momentum=0.1,
                 ini='random'):
        super(DenseResNet, self).__init__()
        """
        Creates FC-DRN arquitecture. It is composed of:
             - an IDB (Initial downsampling block), called ini_conv.
             - a Dense Block containing all densely connected ResNets.
             - an FUB (final upsampling bock), called final_upsample
             - Final classifier

            Input:
                - input_channels: Channels of input images. 3 if RGB
                 images, 1 if grayscale.
                - n_init_conv:  Number of Conv + Relu blocks in IDB.
                - subsample_inout: Downsample factor used in IDB. Same
                 factor to upsample final feature maps in FUB.
                - filter_size_inout: Filter size for IDB and FUB.
                - n_filters_out: Number of channels after IDB and after
                 FUB.
                - n_classes: Number of classes in the dataset to set
                 output_channels in classifier.
                - dilation_list: A list with N-1 dilation factors, where
                 N is the number of ResNets we use in the model.
                 Example: [1,1,2,4,1,1,1] for a setup with 9 ResNets
                 and two dilations: 3rd and 3th transformation block.
                - resnet_config: A list with N positions. Each position
                 indicates a ResNet and the number in the position
                  indicates the number of Residual Blocks for that
                  ResNet. Example: [7] * 9, there are 7 Residual Blocks
                   per each of the 9 ResNets.
                - growth_rate: A list with N positions. Each position
                indicates the number of feature maps that the given
                ResNet outputs.
                - stride: Stride factor used for max poolings, in the
                case we use max poolings as transformations between
                 ResNets.
                - dropout: A list of N positions, indicating the dropout
                 factor per each ResNet.
                - bias: If using bias for the convolutions or not.
                - mixing_conv: A 1x1 convolution after the concatenation
                 to mix all feature channels. In FC-DRN this is always
                 active.
                - transformation: A list with N-1 transformations, each
                 of them used between Resnets.
                  It can be: MaxPool2d, AvgPool2d, 'sconv', 'dilation',
                   'dilation_mg', 'upsample'.
                - preprocess_block: Block used for 'dilation'
                 transformation: It can be preprocess_block_standard,
                 which uses a single instance of BN+ReLU+Conv or
                  preprocess_block_bottle_mg, which uses 3 instances
                  with dil_factor multiplied by 1, 2 and 1 respectively.
                -logsoftmax: If to apply the log_softmax pytorch
                function or not.
                - bn_momentum: Batch norm momentum.
                - ini: Initialization for the dilated convolutions.
                It can be 'random' or 'identity'.
            Output:
                - out: segmentation map with shape:
                 bs x w x h x n_classes. Same resolution as input.
        """
        self.ini_conv = InitialConv(input_channels, n_filters_inout,
                                    filter_size=filter_size_inout,
                                    n_init_conv=n_init_conv,
                                    subsample=subsample_inout, bias=bias)
        self.denseres_block = DenseResBlock(dilation_list, resnet_config,
                                            growth_rate,
                                            dropout, n_filters_inout,
                                            bias=bias,
                                            mixing_conv=mixing_conv,
                                            stride=stride,
                                            transformation=transformation,
                                            preprocess_block=preprocess_block,
                                            bn_momentum=bn_momentum, ini=ini)
        self.final_upsample = Upsample_conv(n_filters_inout +
                                            sum(growth_rate[:9]),
                                            n_filters_inout,
                                            filter_size=filter_size_inout,
                                            bias=bias)

        self.classifier = Classifier(n_filters_inout, n_classes, bias=bias,
                                     logsoftmax=logsoftmax)

    def forward(self, x):
        out = self.ini_conv(x)
        out = self.denseres_block(out)
        out = self.final_upsample(out, x.size()[2:])
        out = self.classifier(out)
        return out


def kaiming_uniform(tensor, a=0, mode='fan_in'):
    """
    Fills the input Tensor or Variable with values according to the
    method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015),
    using a uniform distribution. The resulting tensor will have values
    sampled from
    :math:`U(-bound, bound)` where
    :math:`bound = \sqrt{2 / ((1 + a^2) \\times fan\_in)} \\times \sqrt{3}`.
    Also known as He initialisation.
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the negative slope of the rectifier used after this layer
        (0 for ReLU by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in
            the forward pass. Choosing `fan_out` preserves the
            magnitudes in the backwards pass.

    """
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('relu', a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-bound, bound)
