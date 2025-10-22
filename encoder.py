import copy
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import yaml
from torch import Tensor, nn
from torch.nn.modules.module import register_module_buffer_registration_hook

from feature_extractor import FeatureExtractor, generate_padding_mask

EPS = torch.finfo(torch.float32).eps


# For VggTransformer non-streaming structure
class TransformerPreLNEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.0,
                 final_norm=False):
        super(TransformerPreLNEncoderLayer, self).__init__()

        self.final_norm = final_norm
        self.nhead = nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model) if self.final_norm else None

    def forward(self,
                x: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        prev_x = x
        x = self.norm1(x)
        x = self.self_attn(x,
                           x,
                           x,
                           attn_mask=src_mask,
                           key_padding_mask=src_key_padding_mask)[0]
        x = self.dropout(x)
        x = x + prev_x

        prev_x = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + prev_x

        if self.final_norm:
            x = self.norm3(x)

        return x


class Conv2dSubsampling(nn.Module):

    def __init__(self, idim, odim, num_layers=2, stride="2,2"):
        super(Conv2dSubsampling, self).__init__()
        # The stride parameter is split by ',' here, so it must be a string
        stride = self.stride = list(map(int, stride.split(",")))

        self.num_layers = num_layers
        self.stride = stride

        layers = [("subsampling/pad0", nn.ConstantPad2d((0, 0, 2, 0), 0))]
        layers += [("subsampling/conv0", nn.Conv2d(1, 32, 3, (stride[0], 1))),
                   ("subsampling/relu0", nn.ReLU())]
        for i in range(1, num_layers):
            layers += [(f"subsampling/pad{i}", nn.ConstantPad2d((0, 0, 2, 0),
                                                                0))]
            layers += [(f"subsampling/conv{i}",
                        nn.Conv2d(32, 32, 3, (stride[i], 1))),
                       (f"subsampling/relu{i}", nn.ReLU())]
        layers = OrderedDict(layers)
        self.conv = nn.Sequential(layers)
        self.affine = nn.Linear(32 * (idim - 2 * num_layers), odim)
        self.norm = nn.LayerNorm(odim)

    def forward(self, feats, feat_lengths=None):
        outputs = feats.unsqueeze(1)  # [T, C, B, D]
        outputs = outputs.permute(2, 1, 0, 3)  # [B, C, T, D]
        outputs = self.conv(outputs)
        outputs = outputs.permute(2, 0, 1, 3).contiguous()

        T, B, C, D = outputs.size()
        outputs = self.affine(outputs.view(T, B, C * D))

        outputs = self.norm(outputs)

        if feat_lengths is not None:
            feat_lengths = torch.as_tensor(feat_lengths)
            for i in range(self.num_layers):
                feat_lengths = (feat_lengths - 1) // self.stride[i] + 1

        return outputs, feat_lengths


class PositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param int max_len: maximum input length
    :param reverse: whether to reverse the input position

    """

    def __init__(self, d_model, max_len=2000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.reverse = reverse
        self.scale = math.sqrt(self.d_model)
        self.pe = None

        self._extend_pe(torch.tensor(0.0).expand(max_len, 1))

    def _extend_pe(self, x):
        """Reset the positional encodings."""
        T = x.size(0)
        if self.pe is None or self.pe.size(0) < T:
            pe = torch.zeros(T, self.d_model)
            if self.reverse:
                position = torch.arange(T - 1, -1, -1.0,
                                        dtype=torch.float32).unsqueeze(1)
            else:
                position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                -(math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(1)

        self.pe = self.pe.to(x)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (time, batch, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (time, batch, ...)

        """
        self._extend_pe(x)
        outputs = self.scale * x + self.pe[:x.size(0), :]
        return outputs


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, layer_drop=0.0, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.layer_drop = layer_drop

    def reset_parameters(self, layer_index_offset=0):
        for layer_idx, enc_layer in enumerate(self.layers):
            enc_layer.reset_parameters(layer_index=layer_idx +
                                       layer_index_offset + 1)

    def forward(self, src: Tensor, layer: Optional[int] = None, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) \
                -> Tuple[Tensor, Optional[List[Tensor]]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for idx, mod in enumerate(self.layers):
            output = mod(output,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
            if ((idx + 1) == layer):
                return output

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VGGTFEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 nhead,
                 d_model,
                 dim_feedforward,
                 num_encoder_layers,
                 dropout=0.0,
                 layer_drop=0.0,
                 activation="gelu",
                 subsampling="conv2d",
                 conv2d_stride=None,
                 num_conv_layers=2):
        super(VGGTFEncoder, self).__init__()

        self.subsampling = Conv2dSubsampling(input_size,
                                             d_model,
                                             num_layers=num_conv_layers,
                                             stride=conv2d_stride)
        self.pe = PositionalEncoding(d_model)
        self.pe_dropout = nn.Dropout(dropout)
        self.conv2d_stride = conv2d_stride
        self.num_encoder_layers = num_encoder_layers
        encoder_norm = None

        # FB type
        encoder_layer = TransformerPreLNEncoderLayer(d_model,
                                                     nhead,
                                                     dim_feedforward,
                                                     dropout,
                                                     final_norm=True)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          layer_drop, encoder_norm)

    def forward(self, x, batch_sizes=None):
        x, batch_sizes = self.subsampling(x.transpose(0, 1), batch_sizes)
        x = self.pe_dropout(self.pe(x))
        key_padding_mask = generate_padding_mask(x, batch_sizes)
        x = self.encoder(x, mask=None, src_key_padding_mask=key_padding_mask)

        return x, batch_sizes

    def get_mid_emb(self, x, batch_sizes=None, layer=None):
        x, batch_sizes = self.subsampling(x.transpose(0, 1), batch_sizes)
        x = self.pe_dropout(self.pe(x))
        key_padding_mask = generate_padding_mask(x, batch_sizes)
        x = self.encoder(x,
                         layer=layer,
                         mask=None,
                         src_key_padding_mask=key_padding_mask)

        return x, batch_sizes


@dataclass
class ModelConfig:
    """
    A flattened Dataclass containing all configuration items from the YAML
    """
    # Top-level anchors
    feat_dim: int = 80
    left: int = 0
    right: int = 0
    stride: int = 1
    delta_order: int = 0

    # Fields from feature.fbank
    dither: float = 0.0
    frame_length: int = 25
    frame_shift: int = 10
    preemphasis: float = 0.97
    freq: int = 16000
    high_freq: int = -200
    low_freq: int = 40
    num_mel_bins: int = 80  # (from *feat_dim)

    # Fields from feature
    cmvn: str = "cmvn.npy"

    # Fields from encoder
    encoder_type: str = "VGGtf_encoder"  # (Original 'type' field)

    # Fields from encoder.VGGtf_encoder
    subsampling: str = "conv2d"
    num_conv_layers: int = 2
    conv2d_stride: str = "2,3"  # Keep as str type for .split(',')
    d_model: int = 1280
    nhead: int = 20
    num_encoder_layers: int = 26
    dim_feedforward: int = 5120
    dropout: float = 0.0
    layer_drop: float = 0.0
    activation: str = "gelu"
    # Note: feat_dim, delta_order, left, right, stride in
    # encoder.VGGtf_encoder were references to the top-level anchors.
    # In this flattened structure, we only keep the top-level versions.


def build_VGGtf_encoder(ckpt_encoder_path,
                        cmvn_path):  # <-- Changed type hint here
    """
    Builds the VGGTFEncoder using the FlatModelConfig dataclass
    """
    # Direct attribute access
    model_cfg = ModelConfig()
    model_cfg.cmvn = cmvn_path
    feat_dim = model_cfg.feat_dim
    delta_order = model_cfg.delta_order
    left = model_cfg.left
    right = model_cfg.right
    input_size = (delta_order + 1) * (1 + left + right) * feat_dim

    nhead = model_cfg.nhead
    d_model = model_cfg.d_model
    dim_feedforward = model_cfg.dim_feedforward
    num_encoder_layers = model_cfg.num_encoder_layers
    dropout = model_cfg.dropout
    layer_drop = model_cfg.layer_drop
    activation = model_cfg.activation

    subsampling = model_cfg.subsampling
    num_conv_layers = model_cfg.num_conv_layers
    conv2d_stride = model_cfg.conv2d_stride  # Expected to be str

    model = VGGTFEncoder(input_size,
                         nhead,
                         d_model,
                         dim_feedforward,
                         num_encoder_layers,
                         dropout=dropout,
                         layer_drop=layer_drop,
                         activation=activation,
                         subsampling=subsampling,
                         conv2d_stride=conv2d_stride,
                         num_conv_layers=num_conv_layers)
    feature_extractor = FeatureExtractor(model_cfg)
    state_dict = torch.load(ckpt_encoder_path)

    model_filter_keys = {}
    for key, value in state_dict.items():
        if key.startswith('semantic_tokenizer'):
            if "kmeans" not in key:
                if 'gpu_feature' not in key:
                    name = key.replace('semantic_tokenizer.model.', '')
                    model_filter_keys[name] = value

    model.load_state_dict(model_filter_keys, strict=True)
    return model, feature_extractor


class EmbeddingExtractor(torch.nn.Module):

    def __init__(self, ckpt_encoder_path, cmvn_path) -> None:
        super().__init__()
        self.model, self.feature_extractor = build_VGGtf_encoder(
            ckpt_encoder_path, cmvn_path)

    def forward(self, wavs: torch.Tensor, wavs_lens: torch.Tensor,
                n_layer: int):
        wavs = wavs.float()
        feats, feat_lens = self.feature_extractor(wavs, wavs_lens)
        feats, feat_lens = self.model.get_mid_emb(feats, feat_lens, n_layer)
        embed = feats.transpose(0, 1).contiguous()
        return embed, feat_lens
