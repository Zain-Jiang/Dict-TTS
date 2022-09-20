# Dict-TTS
The implementation of our paper "Dict-TTS: Learning to Pronounce with Prior Dictionary Knowledge for Text-to-Speech"

## Requirements

**Install the dependencies**
```bash
# Install Python 3 fisrt. (Anaconda recommended)
export PYTHONPATH=.
# build a virtual env
conda create -n dict_tts
conda activate dict_tts
# install requirements
pip install -U pip
pip install Cython numpy>=1.21.4
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
```

**Install the aligner (MFA 2.0)**
```bash
# with pip
bash scripts/install_mfa2.sh

# or with conda
conda config --add channels conda-forge
conda install montreal-forced-aligner
```

**Download the datasets (for example, Biaobei)**
Download Biaobei from `https://www.data-baker.com/open source.html` to `data/raw/biaobei`

**Download the pre-trained vocoder**
```
mkdir pretrained
mkdir pretrained/hifigan_hifitts
```
download `model_ckpt_steps_2168000.ckpt`, `config.yaml`, from https://drive.google.com/drive/folders/1n_0tROauyiAYGUDbmoQ__eqyT_G4RvjN?usp=sharing to `pretrained/hifigan_hifitts`

**Download the pre-trained language model**
download `roformer-chinese-base`, from https://huggingface.co/junnyu/roformer_chinese_base to `pretrained/roformer-chinese-base`

**Obtain the dictionary**
You can use the dictionary in ./data/zh-dict.json or crawl the dictionary from the dictionary website mentioned in our paper.

## Choose the config file (for example, DictTTS's config)
```bash
export CONFIG=egs/datasets/audio/biaobei/dict_tts.yaml 
```

## Preprocess
**pre-align**
```bash
python data_gen/tts/bin/pre_align.py --config $CONFIG
```
**mfa-align**
```bash
python data_gen/tts/bin/mfa_train.py --config $CONFIG
python data_gen/tts/bin/mfa_align.py --config $CONFIG
```
**binarize**
```bash
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config $CONFIG
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name dicttts_biaobei_wo_gumbel --reset --hparams="ds_workers=4,max_updates=300000,num_valid_plots=10,use_word_input=True,vocoder_ckpt=pretrained/hifigan_hifitts,max_sentences=60,val_check_interval=2000,valid_infer_interval=2000,binary_data_dir=data/binary/biaobei,word_size=4500,use_dict=True"
```

## Infer (GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG --exp_name dicttts_biaobei_wo_gumbel --infer --hparams="ds_workers=4,max_updates=300000,num_valid_plots=10,use_word_input=True,vocoder_ckpt=pretrained/hifigan_hifitts,max_sentences=60,val_check_interval=2000,valid_infer_interval=2000,binary_data_dir=data/binary/biaobei,word_size=4500,use_dict=True"
```

## Eval the pronunciation error rate (PER)
The PER of the current version is about 1.93 %.
```bash
python scripts/get_pron_error.py
```

## Directory Structure

- `egs`: the config files in the experiments，which is read by `utils/hparams.py`
- `data_gen`: preprocess and binarize the dataset
- `modules`: model
- `scripts`: some scripts used in the experiments
- `tasks`: dataloader, training and inference
- `utils`: utils
- `data`: data folder
    - `raw`: raw files
    - `processed`: preprocessed files
    - `binary`: binary files
- `checkpoints`: checkpoint, tensorboard logs, and inference results。