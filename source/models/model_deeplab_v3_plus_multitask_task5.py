import torch
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p_task5


class ModelDeepLabV3PlusMultiTask_task5(torch.nn.Module):
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
        # print("ch_out_encoder_bottleneck, ch_out_encoder_4x", ch_out_encoder_bottleneck, ch_out_encoder_4x)
        ch_layer_3 = 128 if cfg.model_encoder_name in ['resnet18', 'resnet34'] else 512

        self.bottleneck_size = 256
 
        # TODO: Implement aspps and decoders for each task as a ModuleDict.
        self.aspps = torch.nn.ModuleDict({
            "aspp_semseg": ASPP(ch_out_encoder_bottleneck, self.bottleneck_size, rates=(3, 6, 9)),
            "aspp_si_log": ASPP(ch_out_encoder_bottleneck, self.bottleneck_size, rates=(3, 6, 9))
        })

        self.decoders = torch.nn.ModuleDict({
            "decoder_semseg": DecoderDeeplabV3p_task5(self.bottleneck_size, ch_layer_3, ch_out_encoder_4x, ch_out-1),
            "decoder_si_log": DecoderDeeplabV3p_task5(self.bottleneck_size, ch_layer_3, ch_out_encoder_4x, 1)
        })

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]
        features_layer2 = features[4]
        features_layer3 = features[8]

        # Pass the decoder and aspp for semseg
        semseg_aspp_out = self.aspps["aspp_semseg"](features_lowest)
        semseg_decoder_out, _ = self.decoders["decoder_semseg"](features_layer2, features_layer3, semseg_aspp_out)
        semseg_decoder_out = F.interpolate(semseg_decoder_out, size=input_resolution, mode='bilinear', align_corners=False)

        # Pass the decoder and aspp for depth
        si_log_aspp_out = self.aspps["aspp_si_log"](features_lowest)
        si_log_decoder_out, _ = self.decoders["decoder_si_log"](features_layer2, features_layer3, si_log_aspp_out)
        si_log_decoder_out = F.interpolate(si_log_decoder_out, size=input_resolution, mode='bilinear', align_corners=False)

        concatenated_outputs = torch.cat((semseg_decoder_out, si_log_decoder_out), dim=1)

        out = {}

        # TODO: implement the forward pass for each task as in the previous exercise.
        # However, now task's aspp and decoder need to be forwarded separately.
        offset = 0
        for task, num_ch in self.outputs_desc.items():
            current_precition = concatenated_outputs[:, offset:offset+num_ch, :, :]
            # be sure that depth is > 0, you can use other operators than exp
            out[task] = current_precition.exp().clamp(0.1, 300.0) if task == "depth" else current_precition
            offset += num_ch

        return out
