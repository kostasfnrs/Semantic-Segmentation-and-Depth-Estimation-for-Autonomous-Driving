import torch
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusMultiTask(torch.nn.Module):
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

        # TODO: Implement aspps and decoders for each task as a ModuleDict.
        self.aspps = torch.nn.ModuleDict({
            "aspp_semseg": ASPP(ch_out_encoder_bottleneck, 256),
            "aspp_si_log": ASPP(ch_out_encoder_bottleneck, 256)
        })

        self.decoders = torch.nn.ModuleDict({
            "decoder_semseg": DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out-1),
            "decoder_si_log": DecoderDeeplabV3p(256, ch_out_encoder_4x, 1)
        })

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]
        features_low_level = features[4]

        semseg_aspp_out = self.aspps["aspp_semseg"](features_lowest)
        semseg_decoder_out, _ = self.decoders["decoder_semseg"](semseg_aspp_out, features_low_level)
        semseg_decoder_out = F.interpolate(semseg_decoder_out, size=input_resolution, mode='bilinear', align_corners=False)


        si_log_aspp_out = self.aspps["aspp_si_log"](features_lowest)
        si_log_decoder_out, _ = self.decoders["decoder_si_log"](si_log_aspp_out, features_low_level)
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
