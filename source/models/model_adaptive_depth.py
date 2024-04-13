import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, DecoderDeeplabV3p, TransformerBlock, LatentsExtractor


class ModelAdaptiveDepth(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.decoder = DecoderDeeplabV3p(ch_out_encoder_bottleneck, ch_out_encoder_4x, 256)

        self.conv3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latent_extractor = LatentsExtractor(cfg.num_bins)
        self.att_layer1 = TransformerBlock(256, cfg.num_heads, cfg.expansion)
        self.att_layer2 = TransformerBlock(256, cfg.num_heads, cfg.expansion)
        self.att_layer3 = TransformerBlock(256, cfg.num_heads, cfg.expansion)
        self.att_layer4 = TransformerBlock(256, cfg.num_heads, cfg.expansion)

        self.projector = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 1))

        self.positional_enbeddings = self.get_positional_enbeddings(cfg.num_bins, 256) 
        self.positional_enbeddings = nn.Parameter(self.positional_enbeddings, requires_grad=False)

    def get_positional_enbeddings(self, sequence_length, dimension):
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.pow(10000.0, torch.arange(0, dimension, 2).float() / dimension)
        embeddings = torch.zeros(sequence_length, dimension)
        embeddings[:, 0::2] = torch.sin(position.float() / div_term)
        embeddings[:, 1::2] = torch.cos(position.float() / div_term)
        return embeddings

    def forward(self, x):
        B, _, H, W = x.shape
        input_resolution = (H, W)

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_4x, _ = self.decoder(features_lowest, features[4])

        # TODO: Implement the adaptive extraction of bins and depth computation
        queries = self.conv3x3(features_4x)

        # 1. extract "latents" (i.e., learnable bins) with LatentsExtractor
        latents_extracted = self.latent_extractor(features_4x)

        # 2. process them with a few layers of TrasformerBlocks
        latents_extracted_processed = self.att_layer1(latents_extracted + self.positional_enbeddings)
        latents_extracted_processed = self.att_layer2(latents_extracted_processed)
        latents_extracted_processed = self.att_layer3(latents_extracted_processed)
        latents_extracted_processed = self.att_layer4(latents_extracted_processed)

        # get the similarity of the multi dimensional bins with the queries
        similarity = torch.einsum("bcij,bnc->bnij", queries, latents_extracted_processed )

        # 3. for each latent, project it to a scalar value to obtain the depth
        #    value each corresponds to (which is learnable based on the image)
        projected_latents = self.projector(latents_extracted_processed).squeeze(-1)
        positive_projected_latents = F.softmax(projected_latents, dim=1)

        # create adaptive depth_values with comulative sum
        adaptive_depth_values = torch.cumsum(positive_projected_latents * 300.0, dim=1)
        adaptive_depth_values = adaptive_depth_values.unsqueeze(-1).unsqueeze(-1)

        # 4. for each pixel, compute the probabilities that it belongs to one of
        #    the bins: softmax of the dot product between queries and latents
        # The predefined depth_values and latents, will become learnable and
        # determined by the specific input
        predictions_4x = (F.softmax(similarity, dim=1) * adaptive_depth_values).sum(dim=1, keepdim=True)

        predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            current_precition = predictions_1x[:, offset:offset+num_ch, :, :]
            out[task] = current_precition
            offset += num_ch

        return out