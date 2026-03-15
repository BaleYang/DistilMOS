import argparse
import csv
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download


HF_REPO_ID = "BaleYang/DistilMOS"
TARGET_SR = 16000



def _load_yaml_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {path}")
    return cfg


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Invalid checkpoint format: expected dict.")
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if all(torch.is_tensor(v) for v in ckpt_obj.values()):
        return ckpt_obj
    raise ValueError("Invalid checkpoint format: no model/state_dict found.")


def _download_hf_files(backbone):
    ckpt_file = f"{backbone}/model.pth.tar"
    cfg_file = f"{backbone}/model_cfg.yaml"
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=ckpt_file,
        repo_type="model",
    )
    cfg_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=cfg_file,
        repo_type="model",
    )
    return Path(cfg_path), Path(ckpt_path)


def _load_model(backbone, device):
    cfg_path, ckpt_path = _download_hf_files(backbone)
    raw_cfg = _load_yaml_cfg(cfg_path)
    raw_cfg["ssl_backbone"] = backbone
    model_cfg = SimpleNamespace(**raw_cfg)

    from model.predictor import DistilMOS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(io.StringIO()):
            model = DistilMOS(model_cfg).to(device)
            ckpt_obj = torch.load(ckpt_path, map_location="cpu")
            state_dict = _extract_state_dict(ckpt_obj)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match DistilMOS. "
            "Please regenerate and upload checkpoints with load_model.py first."
        )

    model.eval()
    print("Model loaded successfully.")
    return model


def _load_wav(path, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    wav = wav.float()
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _collect_wavs(input_path):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if p.is_file():
        if p.suffix.lower() != ".wav":
            raise ValueError(f"Only .wav files are supported: {p}")
        return [p]
    wavs = sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() == ".wav"])
    if not wavs:
        raise ValueError(f"No .wav files found under: {input_path}")
    return wavs


def _collate_waveforms(wavs):
    lengths = [w.shape[0] for w in wavs]
    max_len = max(lengths)
    bsz = len(wavs)
    batch_wav = torch.zeros(bsz, max_len, dtype=torch.float32)
    padding_mask = torch.ones(bsz, max_len, dtype=torch.bool)
    for i, w in enumerate(wavs):
        length = w.shape[0]
        batch_wav[i, :length] = w
        padding_mask[i, :length] = False
    return batch_wav, padding_mask


def _predict(model, wav_paths, batch_size, device):
    results = []
    with torch.no_grad():
        for start in range(0, len(wav_paths), batch_size):
            batch_paths = wav_paths[start : start + batch_size]
            batch_wavs = [_load_wav(p) for p in batch_paths]
            wav_tensor, padding_mask = _collate_waveforms(batch_wavs)
            batch = {
                "wav": wav_tensor.to(device),
                "audio_padding_mask": padding_mask.to(device),
            }
            out = model(batch)["mos"]
            if out.ndim == 2 and out.size(-1) == 1:
                out = out[:, 0]
            out = out.detach().cpu().float().tolist()
            for path, mos in zip(batch_paths, out):
                results.append((str(path), float(mos)))
    return results


def _save_csv(results, output_path):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "mos"])
        for path, mos in results:
            writer.writerow([path, f"{mos:.6f}"])
    print(f"Saved prediction CSV: {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load DistilMOS from Hugging Face and predict MOS.")
    parser.add_argument("--input", required=True, help="A wav file path or a directory containing wav files.")
    parser.add_argument("--ssl_backbone", default="wavlm", choices=["wavlm", "w2v2"], help="Backbone to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for directory inference.")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = _load_model(
        backbone=args.ssl_backbone,
        device=device,
    )

    wav_paths = _collect_wavs(args.input)
    is_dir = Path(args.input).is_dir()
    batch_size = max(1, args.batch_size if is_dir else 1)
    results = _predict(model, wav_paths, batch_size=batch_size, device=device)

    for path, mos in results:
        print(f"{path}\t{mos:.4f}")

    mos_values = [x[1] for x in results]
    print(f"Summary: files={len(results)}, mean_mos={sum(mos_values)/len(mos_values):.4f}")

    if args.output:
        _save_csv(results, args.output)


if __name__ == "__main__":
    main()
