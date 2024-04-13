import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(pretrained=True)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )

        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()
        # Done: Implement a proper decoder with skip connections instead of the following

        # 1x1 conv
        out_low_level_feature_channles = int(skip_4x_ch/2)
        self.conv1x1 = torch.nn.Conv2d(skip_4x_ch, out_low_level_feature_channles, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_low_level_feature_channles)
        self.relu = torch.nn.ReLU()

        # 3x3 conv
        self.final_features_after_concat = bottleneck_ch + out_low_level_feature_channles
        self.conv3x3 = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.final_features_after_concat, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        ])

        # final classification head
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # Done: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        features_4x = F.interpolate(features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)

        # 1x1 conv
        low_level_features_reduced_channels = self.conv1x1(features_skip_4x)
        low_level_features_reduced_channels = self.bn(low_level_features_reduced_channels)
        low_level_features_reduced_channels = self.relu(low_level_features_reduced_channels)

        concatenated_all = torch.cat([low_level_features_reduced_channels, features_4x], dim=1)
            
        # 3x3 conv    
        conv3x3_out = self.conv3x3(concatenated_all)

        # interpolate by 4
        final_out_interpolated_features = F.interpolate(conv3x3_out, scale_factor=4, mode='bilinear', align_corners=False)

        # final classification head
        predictions_4x = self.features_to_predictions(final_out_interpolated_features)
        
        return predictions_4x, features_4x

class DecoderDeeplabV3p_task5(torch.nn.Module):
    def __init__(self, bottleneck_ch, ch_layer_3, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p_task5, self).__init__()
        # Done: Implement a proper decoder with skip connections instead of the following

        # 1x1 conv FOR LAYER 2
        out_low_level_layer_2_channles = int(skip_4x_ch/2)
        self.conv1x1_layer_2 = torch.nn.Conv2d(skip_4x_ch, out_low_level_layer_2_channles, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn_layer_2 = torch.nn.BatchNorm2d(out_low_level_layer_2_channles)
        self.relu_layer_2 = torch.nn.ReLU()

        # 1x1 conv FOR LAYER 3
        out_low_level_layer_3_channels = int(ch_layer_3/2)
        self.conv1x1_layer_3 = torch.nn.Conv2d(ch_layer_3, out_low_level_layer_3_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn_layer_3 = torch.nn.BatchNorm2d(out_low_level_layer_3_channels)
        self.relu_layer_3 = torch.nn.ReLU()


        # 3x3 conv for bottleneck + layer 3
        self.features_after_concat_bottleneck_layer3 = bottleneck_ch + out_low_level_layer_3_channels
        self.conv3x3_bottleneck_layer3 = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.features_after_concat_bottleneck_layer3, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        ])

        # 3x3 conv for bottleneck + layer3 + layer 2
        self.features_after_concat_bottleneck_layer2 = bottleneck_ch + out_low_level_layer_2_channles
        self.conv3x3_bottleneck_layer2 = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.features_after_concat_bottleneck_layer2, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU()
        ])

        # final classification head
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_layer2, features_layer3, features_bottleneck):
        """
        DeepLabV3+ with extra skip connections style decoder
        :param features_layer2: features of encoder of layer 2
        :param features_layer3: features of encoder of layer 3
        :param features_bottleneck: features of encoder of layer 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # Done: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        features_layer3 = F.interpolate(features_layer3, size=features_layer2.shape[2:], mode='bilinear', align_corners=False)

        # 1x1 conv FOR LAYER 2
        low_level_layer_2_features_reduced_channels = self.conv1x1_layer_2(features_layer2)
        low_level_layer_2_features_reduced_channels = self.bn_layer_2(low_level_layer_2_features_reduced_channels)
        low_level_layer_2_features_reduced_channels = self.relu_layer_2(low_level_layer_2_features_reduced_channels)

        # 1x1 conv FOR LAYER 3
        low_level_layer_3_features_reduced_channels = self.conv1x1_layer_3(features_layer3)
        low_level_layer_3_features_reduced_channels = self.bn_layer_3(low_level_layer_3_features_reduced_channels)
        low_level_layer_3_features_reduced_channels = self.relu_layer_3(low_level_layer_3_features_reduced_channels)


        # concat bottleneck + layer 3
        features_4x = F.interpolate(features_bottleneck, size=features_layer3.shape[2:], mode='bilinear', align_corners=False)
        concatenated_bottleneck_layer3 = torch.cat([low_level_layer_3_features_reduced_channels, features_4x], dim=1)
        conv3x3_out_bottleneck_layer3 = self.conv3x3_bottleneck_layer3(concatenated_bottleneck_layer3)

        # concat bottleneck + layer 2 previous from layer 3
        conv3x3_out_bottleneck_layer3 = F.interpolate(conv3x3_out_bottleneck_layer3, size=features_layer2.shape[2:], mode='bilinear', align_corners=False)
        concatenated_bottleneck_layer2 = torch.cat([low_level_layer_2_features_reduced_channels, conv3x3_out_bottleneck_layer3], dim=1)
        concatenated_bottleneck_layer2 = self.conv3x3_bottleneck_layer2(concatenated_bottleneck_layer2)

        # interpolate by 4
        final_out_interpolated_features = F.interpolate(concatenated_bottleneck_layer2, scale_factor=4, mode='bilinear', align_corners=False)

        # final classification head
        predictions_4x = self.features_to_predictions(final_out_interpolated_features)

        return predictions_4x, features_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()

        self.rates = rates

        setattr(self, 'atrous_conv_1', ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))
        for rate in self.rates:
            setattr(self, f'atrous_conv_{rate}', ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate))
        
        self.image_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.image_pooling_conv = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

        self.final_concatenated_channels = out_channels + len(self.rates)*out_channels + out_channels

        self.out_conv = ASPPpart(in_channels=self.final_concatenated_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1)


    def forward(self, x):
        
        size_h, size_w = x.size()[2:]
        
        x_1 = self.atrous_conv_1(x)

        atrous_outputs = []
        for rate in self.rates:  
            atrous_conv = getattr(self, f'atrous_conv_{rate}')
            atrous_outputs.append(atrous_conv(x))

        image_pooled = self.image_pooling(x)
        image_pooled_conv = self.image_pooling_conv(image_pooled)
        image_pooled_conv_upsampled = F.interpolate(image_pooled_conv, size=(size_h, size_w), mode="bilinear", align_corners=False)

        concatenated_outputs = torch.cat([x_1] + atrous_outputs + [image_pooled_conv_upsampled], dim=1)

        out = self.out_conv(concatenated_outputs)

        return out


class MLP(torch.nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, int(dim * expansion))
        self.act = nn.GELU()
        self.proj2 = nn.Linear(int(dim * expansion), dim)

    def forward(self, x):
        """
        MLP with pre-normalization with one hidden layer
        :param x: batched tokens with dimesion dim (B,N,C)
        :return: tensor (B,N,C)
        """
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dim, dim_head, temperature):
        super().__init__()

        self.dk = temperature * torch.sqrt(torch.tensor(dim_head, dtype=torch.float)) 
        self.Q = nn.Linear(dim, dim_head)
        self.K = nn.Linear(dim, dim_head)
        self.V = nn.Linear(dim, dim_head)

    def forward(self, x):

        # 1. obtain query, key, value
        query = self.Q(x) 
        key   = self.K(x)
        value = self.V(x)

        # 3. compute the similairty between query and key (dot product)
        dot_prod = query @ key.transpose(2,1)
        dot_prod = dot_prod / self.dk

        # 4. obtain the convex combination of the values based on softmax of similarity
        convex_compinations = F.softmax(dot_prod, dim=1)
        x = convex_compinations @ value

        return x

class SelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out = nn.Linear(dim, dim)
        self.temperature = 1.0
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

        # TODO: Implement self attention, you need a projection
        # and the normalization layer
        assert dim % num_heads == 0
        self.dim_head = dim // num_heads 
        
        self.ScaledDotProductAttentions = nn.ModuleList()
        for head in range(num_heads):
            self.ScaledDotProductAttentions.append(ScaledDotProductAttention(dim, self.dim_head, self.temperature))

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x, pos_embed):
        """
        Pre-normalziation style self-attetion
        :param x: batched tokens with dimesion dim (B,N,C)
        :param pos_embed: batched positional with shape (B, N, C)
        :return: tensor processed with output shape same as input
        """
        B, N, C = x.shape


        # TODO: Implement self attention, you need:
        # 2. if positional embed is not none add to the query tensor
        if pos_embed is not None:
            x = x + pos_embed
            x = self.dropout(x)

        # pre layer norm
        x = self.layer_norm(x)

        # att heads
        att_head_outputs = []
        for head in range(self.num_heads):
            out = self.ScaledDotProductAttentions[head](x)
            att_head_outputs.append(out)

        x = torch.cat(att_head_outputs, dim=2)
        x = self.out(x)

        # dropout
        x = self.dropout(x)

        
        # shortcut connection is done in the Transformer block

        # Remember that the num_heads speciifc the number of indepdendt heads to unpack
        # the operation in. Namely 2 heads, means that the channels are unpacked into 
        # 2 independent: something.reshape(B, N, self.num_heads, C // self.num_heads)
        # and rearrange accordingly to perform the operation such that you work only with
        # N and self.dim // self.num_heads
        # Remember that also the positional embedding needs to follow a similar rearrangement
        # to be consistent with the shapes of the query
        # Remember to rearrange the output tensor such that the output shape is B N C again
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=4, expansion=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.mlp = MLP(dim, expansion=expansion)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads)

    def forward(self, x, pos_embed=None):
        x = self.attn(x, pos_embed) + x
        x = self.mlp(x) + x
        return x
    

class LatentsExtractor(torch.nn.Module):
    def __init__(self, num_latents: int = 256):
        super().__init__()
        edge = num_latents ** 0.5
        assert edge.is_integer(), "Remeber to give num_bins a number whose sqrt is still an integere, e.g., 256"
        self.edge = int(edge)
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_down = F.interpolate(x, size=(self.edge, self.edge), mode="bilinear", align_corners=False)
        x_down_flat = x_down.permute(0, 2, 3, 1).view(B, self.edge*self.edge, C)
        return x_down_flat
    
