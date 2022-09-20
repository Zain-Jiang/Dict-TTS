import importlib
import json
import os

from utils.hparams import set_hparams, hparams

set_hparams()

spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
if os.path.exists(spk_map_fn):
    spk_map = json.load(open(spk_map_fn))
else:
    binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer = binarizer_cls()
    spk_map = binarizer.build_spk_map()
print("| Spk map: ", spk_map)
