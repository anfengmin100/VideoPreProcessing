# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 17:15:43

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

class TSN(nn.Module):
    def __init__(
            self, num_class, num_segments, modality,
            base_model='tea', new_length=None, consensus_type='avg',
            before_softmax=True, dropout=0.5, img_feature_dim=256,
            crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
            is_shift=False, shift_div=8, shift_place='blockres',
            fc_lr5=False):
        super(TSN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""Initializing TSN with base model: {}.
                      TSN Configurations:
                      input_modality:     {}
                      num_segments:       {}
                      new_length:         {}
                      consensus_module:   {}
                      dropout_ratio:      {}
                      img_feature_dim:    {}""".format(base_model, self.modality,
                                                       self.num_segments, self.new_length,
                                                       consensus_type, self.dropout,
                                                       self.img_feature_dim)))
        
        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        
        # afm add snip sampling
        self.new_length = 3 # for snip sampling
        self.single_frame_channel = 3
        self._reconstruct_first_layer()

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        
        if 'tea' in base_model:
            if self.num_segments == 8:
                from ops.tea50_8f import tea50_8f
                self.base_model = tea50_8f(pretrained=True)
            if self.num_segments == 16:
                from ops.tea50_16f import tea50_16f
                self.base_model = tea50_16f(pretrained=True)
            
            #self.input_size = 224
            self.input_size = 112
            self.base_model.last_layer_name = 'fc'
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'RGB':
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        inorm = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5 and m.out_features == self.num_class:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5 and m.out_features == self.num_class:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.InstanceNorm1d):
                inorm.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.InstanceNorm2d):
                inorm.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.InstanceNorm3d):
                inorm.extend(list(m.parameters()))

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
                {
                    'params': first_conv_weight,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "first_conv_weight",
                },
                {
                    'params': first_conv_bias,
                    'lr_mult': 2,
                    'decay_mult': 0,
                    'name': "first_conv_bias",
                },
                {
                    'params': normal_weight,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "normal_weight",
                },
                {
                    'params': normal_bias,
                    'lr_mult': 2,
                    'decay_mult': 0,
                    'name': "normal_bias",
                },
                {
                    'params': bn,
                    'lr_mult': 1,
                    'decay_mult': 0,
                    'name': "BN scale/shift",
                },
                {
                    'params': inorm,
                    'lr_mult': 1,
                    'decay_mult': 0,
                    'name': "IN scale/shift",
                },
                {
                    'params': custom_ops,
                    'lr_mult': 1,
                    'decay_mult': 1,
                    'name': "custom_ops",
                },
                # for fc
                {
                    'params': lr5_weight,
                    'lr_mult': 5,
                    'decay_mult': 1,
                    'name': "lr5_weight",
                },
                {
                    'params': lr10_bias,
                    'lr_mult': 10,
                    'decay_mult': 0,
                    'name': "lr10_bias",
                },
        ]

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
            # base_out: nt * 2048
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)
            # base_out: n * num_cls

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:,:,1:,:,:,:].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:,:,x,:,:,:] = input_view[:,:,x,:,:,:] - input_view[:,:,x-1,:,:,:]
            else:
                new_data[:,:,x-1,:,:,:] = input_view[:,:,x,:,:,:] - input_view[:,:,x-1,:,:,:]

        return new_data

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner

        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model
    
    def _reconstruct_first_layer(self):
        print('Reconstructing first conv...')
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x],
                                                          nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.single_frame_channel * \
                                             self.new_length,) + \
                          kernel_size[2:]
        # if not(self.single_frame_channel == 3):
        if self.modality == 'Flow':
            new_kernels = params[0].data.mean(dim=1,
                                              keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernels = params[0].data.repeat([1, self.new_length, ] + \
                                                [1] * (len(kernel_size[2:]))).contiguous()

        new_conv = nn.Conv2d(self.single_frame_channel * \
                             self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride,
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = "conv1"

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        #return self.input_size * 256 // 224
        return self.input_size * 128 // 112

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
