"""
Bonito Model template
"""

from base64 import encode
from os import sep
import numpy as np
from bonito.nn import Permute, layers
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout, Identity, ReLU

from fast_ctc_decode import beam_search, viterbi_search


class Model(Module):
    """
    Model template for QuartzNet style architectures
    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.encoder = Encoder(config)
        self.decoder = Decoder(self.features, len(self.alphabet))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        x = x.exp().cpu().numpy().astype(np.float32)
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq

    def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        T, N, C = log_probs.shape
        weights = weights or torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)])
        log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
        loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'total_loss': loss + label_smoothing_loss, 'loss': loss, 'label_smooth_loss': label_smoothing_loss}

    def loss(self, log_probs, targets, lengths):
        return self.ctc_label_smoothing_loss(log_probs, targets, lengths)

class Encoder(Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        features = self.config['input']['features']
        activation = layers[self.config['encoder']['activation']]()
        encoder_layers = []
        self.residual_layers = [] # List of indices of which blocks in `encoder_layers` have some sort of residual connection 

        for i, layer in enumerate(self.config['block']):
            if 'short_residual' in layer and layer['short_residual']:
                encoder_layers.append(
                    ShortenableBlock(
                        features, 
                        layer['filters'], 
                        activation,
                        residual=layer['default_residual'], # NOTE: Only shortenable blocks have default_residual flag. It's only on for dynamic skip shortening. When false, skips are statically shortened from the start.
                        repeat=layer['repeat'], 
                        kernel_size=layer['kernel'],
                        stride=layer['stride'], 
                        dilation=layer['dilation'],
                        dropout=layer['dropout'], 
                        separable=layer['separable'],
                    )
                )
                if layer['short_residual']:
                    self.residual_layers.append(i)
            else:
                encoder_layers.append(
                    Block(
                        features, layer['filters'], activation,
                        repeat=layer['repeat'], kernel_size=layer['kernel'],
                        stride=layer['stride'], dilation=layer['dilation'],
                        dropout=layer['dropout'], residual=layer['residual'],
                        separable=layer['separable'],
                    )
                )
                if layer['residual']:
                    self.residual_layers.append(i)

            features = layer['filters']

        self.encoder = Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x


class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x = _x + self.residual(x)
        return self.activation(_x)


class ShortenableSubBlock(Module):
    """
    A sub-block of a ShortenableBlock that consists of the
    TCSConv, BN, Activation, Dropout layers
    all wrapped up into a single Module so that we can 
    easily wrap them in a shortened skip connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False, dropout=0.0):

        super(ShortenableSubBlock, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

        self.bn1d = BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.act = ReLU()
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        _x = x
        if self.separable:
            _x = self.depthwise(_x)
            _x = self.pointwise(_x)
        else:
            _x = self.conv(_x)
        _x = self.bn1d(_x)
        _x = self.act(_x)
        _x = self.dropout(_x)
        return _x

class ShortenableBlock(Module):
    """
    A block with shortened skip connections
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(ShortenableBlock, self).__init__()

        self.use_default_res = residual # When default res is False, use short residuals instead
        self.conv = ModuleList()

        _in_channels = in_channels
        self.activation = activation
        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for _ in range(repeat):
            self.conv.extend(
                self.get_subblock(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )
            _in_channels = out_channels

        # add the shortened residual connection
        self.short_residual = Identity()
        # In the case that the number of input channels doesn't match
        # the number of output channels, project the shortcut through 
        # a 1x1 convolution to match them up
        self.projected_residual = Sequential(*self.get_tcs(in_channels, out_channels))

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_subblock(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            ShortenableSubBlock(
                in_channels, out_channels, kernel_size, 
                stride=stride, dilation=dilation,
                padding=padding, bias=bias, separable=separable
            )
        ]

    def forward(self, x):
        _x = x
        for _, layer in enumerate(self.conv):
            x_in = _x
            _x = layer(_x)
            if not self.use_default_res:
                if x_in.shape != _x.shape:
                    _x = _x + self.projected_residual(x_in)
                else:
                    _x = _x + self.short_residual(x_in)
        if self.use_default_res:
            _x = _x + self.projected_residual(x)
        return _x


class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = Sequential(
            Conv1d(features, classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )

    def forward(self, x):
        return log_softmax(self.layers(x), dim=-1)


# Model modifier functions
def update_skip_removal(model, how_often, epoch):
    """
    Remove skip connection every epoch, starting from the head of the network.
    At epoch i, skip connection (i // how_often + 1) is removed. Note epochs and skip connections are 1-indexed.

    Returns True if model modified; otherwise, False

    model: QuartzNet model to modify
    epoch: Current epoch
    """
    # In the CTC QuartzNet model, the residual layers are layers [1, 2, 3, 4, 5],
    # which is 1-indexed. Epochs are also 1-indexed. This code is model-specific
    # and thus hacky.
    skip_to_remove = epoch if how_often == 1 else epoch // how_often + 1
    if (epoch - 1) % how_often == 0 and skip_to_remove in model.encoder.residual_layers:
        print(f'\n\nskip {skip_to_remove} REMOVED at epoch {epoch}')
        assert model.encoder.encoder[skip_to_remove].use_res
        model.encoder.encoder[skip_to_remove].use_res = False
        return True

    return False


def update_skip_shorten(model, how_often, epoch):
    """
    Shorten skip connection every epoch, starting from the head of the network.
    At epoch i, skip connection (i // how_often + 1) is shortened into smaller skips.
    Note epochs and skip connections are 1-indexed.

    Returns True if model modified; otherwise, False

    model: QuartzNet model to modify
    epoch: Current epoch
    """
    # In the CTC QuartzNet model, the residual layers are layers [1, 2, 3, 4, 5],
    # which is 1-indexed. Epochs are also 1-indexed. This code is model-specific
    # and thus hacky.
    skip_to_shorten = epoch if how_often == 1 else epoch // how_often + 1
    if (epoch - 1) % how_often == 0 and skip_to_shorten in model.encoder.residual_layers:
        print(f'\n\nskip {skip_to_shorten} shortened at epoch {epoch}')
        assert model.encoder.encoder[skip_to_shorten].use_default_res
        model.encoder.encoder[skip_to_shorten].use_default_res = False
        return True

    return False

