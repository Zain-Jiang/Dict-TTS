import argparse
import glob
import importlib
import os
import subprocess
import librosa
from data_gen.tts.base_pre_align import BasePreAlign
from utils import audio, get_encoding
from utils.hparams import hparams, set_hparams
from utils.rnnoise import rnnoise

if __name__ == "__main__":
    set_hparams()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', type=str, default='', help='input dir')
    parser.add_argument('--output_dir', type=str, default='', help='output dir')
    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    processed_data_dir = hparams['processed_data_dir']
    pre_align_args = hparams['pre_align_args']
    pre_align_args['sox_to_wav'] = True
    pre_align_args['trim_sil'] = True
    pre_align_args['sox_resample'] = True
    pre_align_args['denoise'] = True

    pkg = ".".join(hparams["pre_align_cls"].split(".")[:-1])
    cls_name = hparams["pre_align_cls"].split(".")[-1]
    process_cls = getattr(importlib.import_module(pkg), cls_name)
    pre_aligner = process_cls()
    txt_processor = pre_aligner.txt_processor

    mfa_process_dir = f'{input_dir}/mfa_outputs'
    subprocess.check_call(f'rm -rf {mfa_process_dir}', shell=True)
    os.makedirs(mfa_process_dir, exist_ok=True)
    os.makedirs(f'{mfa_process_dir}/wav_inputs', exist_ok=True)

    for idx, txt_fn in enumerate(glob.glob(f'{input_dir}/*.txt')):
        base_fn = os.path.splitext(txt_fn)[0]
        basename = os.path.basename(base_fn)
        if os.path.exists(base_fn + '.wav'):
            wav_fn = base_fn + '.wav'
        elif os.path.exists(base_fn + '.mp3'):
            wav_fn = base_fn + '.mp3'
        else:
            continue
        # process text
        encoding = get_encoding(txt_fn)
        with open(txt_fn, encoding=encoding) as f:
            txt_raw = " ".join(f.readlines()).strip()
        phs, _, phs_for_align, _ = pre_aligner.process_text(txt_processor, txt_raw, hparams['pre_align_args'])
        with open(f'{mfa_process_dir}/{basename}.lab', 'w') as f:
            f.write(phs_for_align)
        # process wav
        new_wav_fn = pre_aligner.process_wav(idx, basename, wav_fn, mfa_process_dir, pre_align_args)
        subprocess.check_call(f'cp "{new_wav_fn}.wav" "{mfa_process_dir}/{basename}.wav"', shell=True)

    subprocess.check_call(
        f'BASE_DIR={hparams["processed_data_dir"]} '
        f'ALIGN_INPUT_DIR={mfa_process_dir} '
        f'ALIGN_OUTPUT_DIR={input_dir} '
        f'bash scripts/run_mfa_align.sh; '
        f'rm {input_dir}/oovs_found.txt', shell=True)
