import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Config
from .WavLM import WavLM, WavLMConfig
from .modules import MLP, WeightedSum, FeatureProcessor, BiLSTMCNNBlock


class DistilMOS(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.ssl_backbone = getattr(model_cfg, 'ssl_backbone', 'wavlm').lower()
        freeze_ssl = getattr(model_cfg, 'freeze_wavlm', False)

        if self.ssl_backbone == 'wavlm':
            wavlm_cfg_dict = getattr(model_cfg, 'wavlm_cfg', None)
            if wavlm_cfg_dict is None:
                wavlm_cfg_path = getattr(model_cfg, 'wavlm_cfg_path', None)
                if wavlm_cfg_path is None:
                    wavlm_cfg_path = getattr(model_cfg, 'wavlm_path', None)
                if wavlm_cfg_path:
                    try:
                        raw = torch.load(wavlm_cfg_path, map_location='cpu')
                        if isinstance(raw, dict) and isinstance(raw.get("cfg"), dict):
                            wavlm_cfg_dict = raw["cfg"]
                            print(f"ssl_backbone: wavlm (init from cfg in {wavlm_cfg_path}, no pretrained load)")
                    except Exception:
                        pass
            if wavlm_cfg_dict is None:
                wavlm_cfg_dict = {
                    "relative_position_embedding": True,
                    "gru_rel_pos": True,
                    "num_buckets": 320,
                    "max_distance": 800,
                }
                print("ssl_backbone: wavlm (init from default wavlm-base cfg, no pretrained load)")
            if wavlm_cfg_dict is not None and not isinstance(wavlm_cfg_dict, dict):
                raise ValueError("wavlm_cfg 必须是 dict")
            cfg = WavLMConfig(wavlm_cfg_dict)
            self.ssl = WavLM(cfg)
            self.ssl_num_layers = cfg.encoder_layers
            self.ssl_embed_dim = cfg.encoder_embed_dim
        elif self.ssl_backbone == 'w2v2':
            w2v2_cfg_dict = getattr(model_cfg, 'w2v2_cfg', None)
            if w2v2_cfg_dict is None:
                w2v2_cfg = Wav2Vec2Config()
            else:
                w2v2_cfg = Wav2Vec2Config(**w2v2_cfg_dict)
            print("ssl_backbone: w2v2 (init from config only, no pretrained load)")
            self.ssl = Wav2Vec2Model(w2v2_cfg)
            self.ssl_num_layers = w2v2_cfg.num_hidden_layers
            self.ssl_embed_dim = w2v2_cfg.hidden_size
        else:
            raise ValueError(f"ssl_backbone must be 'wavlm' or 'w2v2', got '{self.ssl_backbone}'")

        if freeze_ssl:
            print(f"Freezing {self.ssl_backbone} parameters")
            for p in self.ssl.parameters():
                p.requires_grad = False
            self.ssl.eval()

        self.weighted_sum = WeightedSum(num_layers=self.ssl_num_layers, embed_dim=self.ssl_embed_dim)

        strides = getattr(model_cfg, "downsample_strides", [1, 1, 1])
        kernels = getattr(model_cfg, "downsample_kernels", [5, 3, 3])
        hidden_dim = getattr(model_cfg, "hidden_dim", 256)
        self.strides = strides

        self.feature_processor = FeatureProcessor(d_in=self.ssl_embed_dim, d_mid=hidden_dim, d_out=hidden_dim, strides=strides, kernels=kernels)

        self.target_layers = getattr(model_cfg, "target_layers", list(range(1, self.ssl_num_layers + 1)))
        self.predictors = nn.ModuleList([
            MLP(input_dim=hidden_dim, output_dim=200, hidden_dim=hidden_dim, num_layers=3)
            for i in self.target_layers
        ])

        self.decoder = BiLSTMCNNBlock(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def get_padding_mask(self, padding_mask, strides):
        if padding_mask is None:
            return None
        if isinstance(strides, int):
            strides = [strides]
        elif strides is None:
            strides = []
        original_dtype = padding_mask.dtype
        if padding_mask.dtype is not torch.bool:
            mask = padding_mask > 0
        else:
            mask = padding_mask
        for s in strides:
            if s is None or s <= 1:
                continue
            batch_size, t_in = mask.shape
            t_out = (t_in + s - 1) // s
            pad_len = t_out * s - t_in
            if pad_len > 0:
                tail_pad = torch.ones(batch_size, pad_len, device=mask.device, dtype=torch.bool)
                mask = torch.cat([mask, tail_pad], dim=1)
            mask = mask.view(batch_size, t_out, s).all(dim=2)
        if original_dtype is torch.bool:
            return mask
        if torch.is_floating_point(padding_mask):
            return mask.float()
        return mask.long()

    def extract_all_layer_features(self, source, padding_mask=None):
        if self.ssl_backbone == 'wavlm':
            return self._extract_wavlm(source, padding_mask)
        else:
            return self._extract_w2v2(source, padding_mask)

    def _extract_wavlm(self, source, padding_mask=None):
        if self.ssl.feature_grad_mult > 0:
            features = self.ssl.feature_extractor(source)
            if self.ssl.feature_grad_mult != 1.0:
                from .WavLM import GradMultiply
                features = GradMultiply.apply(features, self.ssl.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.ssl.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.ssl.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.ssl.forward_padding_mask(features, padding_mask)
        if self.ssl.post_extract_proj is not None:
            features = self.ssl.post_extract_proj(features)
        features = self.ssl.dropout_input(features)
        x = features
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.ssl.encoder.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        if not self.ssl.encoder.layer_norm_first:
            x = self.ssl.encoder.layer_norm(x)
        x = torch.nn.functional.dropout(x, p=self.ssl.encoder.dropout, training=self.training)
        x = x.transpose(0, 1)
        layer_outputs = []
        pos_bias = None
        for i, layer in enumerate(self.ssl.encoder.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.ssl.encoder.layerdrop):
                x, _, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=None, pos_bias=pos_bias)
            layer_outputs.append(x.transpose(0, 1))
        return layer_outputs, padding_mask

    def _extract_w2v2(self, source, padding_mask=None):
        attention_mask = (~padding_mask).long() if padding_mask is not None else None
        outputs = self.ssl(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        layer_outputs = [hs for hs in hidden_states[1:]]
        feature_padding_mask = None
        if hasattr(outputs, 'extractor_attention_mask') and outputs.extractor_attention_mask is not None:
            feature_padding_mask = ~outputs.extractor_attention_mask.bool()
        elif padding_mask is not None:
            valid_audio_lengths = (~padding_mask).long().sum(dim=1)
            try:
                out_lengths = self.ssl._get_feat_extract_output_lengths(valid_audio_lengths)
            except Exception:
                B, T_feat = layer_outputs[-1].size(0), layer_outputs[-1].size(1)
                T_audio = source.size(1)
                ratio = (T_feat / max(T_audio, 1))
                out_lengths = (valid_audio_lengths.float() * ratio).round().long().clamp(min=0, max=T_feat)
            B, T_feat = layer_outputs[-1].size(0), layer_outputs[-1].size(1)
            range_tensor = torch.arange(T_feat, device=source.device).unsqueeze(0).expand(B, -1)
            valid_mask = range_tensor < out_lengths.unsqueeze(1)
            feature_padding_mask = ~valid_mask
        return layer_outputs, feature_padding_mask

    def forward(self, batch):
        source = batch['wav']
        padding_mask = batch['audio_padding_mask']
        token_padding_mask = batch.get('token_padding_mask', None)

        layer_outputs, padding_mask = self.extract_all_layer_features(source, padding_mask)

        ssl_features = self.weighted_sum(layer_outputs)

        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            ssl_features = ssl_features * mask

        ssl_features = self.feature_processor(ssl_features)
        if token_padding_mask is not None and self.training:
            _, l1, _ = ssl_features.shape
            _, l2 = token_padding_mask.shape
            if l1 != l2:
                min_l = min(l1, l2)
                ssl_features = ssl_features[:, :min_l, :]
                token_padding_mask = token_padding_mask[:, :min_l]
                for i in self.target_layers:
                    batch[f'tokens_{i}'] = batch[f'tokens_{i}'][:, :min_l]
        else:
            token_padding_mask = self.get_padding_mask(padding_mask, self.strides)

        hidden_states = ssl_features

        if self.training:
            pred_features = [self.predictors[i](ssl_features) for i in range(len(self.target_layers))]
        else:
            pred_features = None

        ssl_features = self.decoder(ssl_features, padding_mask=token_padding_mask)

        frame_mos = self.linear(ssl_features)

        if token_padding_mask is not None:
            mask = (~token_padding_mask).float()
            frame_mos = frame_mos * mask.unsqueeze(-1)
            utt_mos = frame_mos.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        else:
            utt_mos = frame_mos.mean(dim=1)

        return {
            "mos": utt_mos,
            "frame_mos": frame_mos,
            "pred_features": pred_features,
            "token_padding_mask": token_padding_mask,
            "hidden_states": hidden_states,
        }
