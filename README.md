Multiverb IVA MT
================
Generating diverse verb variants with VerbNet and Conditional Beam Search for enhanced performance of Intelligent Virtual Assistants (IVA) training set translation.

Usage:
```python
from iva_mt.iva_mt import IVAMT

translator = IVAMT("pl")
#for single-best translation
translator.translate("set the temperature on <a>my<a> thermostat")
#for multi-variant translation
translator.generate_alternative_translations("set the temperature on <a>my<a> thermostat")
```
Available languages (en2xx): [pl](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-pl), [es](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-es), [de](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-de), [fr](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-fr), [pt](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-pt), [sv](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-sv), [zh](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-zh), [ja](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-ja), [tr](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-tr), [hi](https://huggingface.co/cartesinus/iva_mt_wslot-m2m100_418M-en-hi)

To use GPU and batching, provide information about device:
```python
IVAMT("pl", device="cuda:0", batch_size=16)
```
On V100 this allows to translate ~100 sentences/minute.

To use baseline M2M100:
```python
IVAMT("pl", model_name="facebook/m2m100_418M")
```

## Training M2M100 Model

In this repository, we provide a script `train.py` to facilitate the training of M2M100 models on your specified translation tasks. To run the training script, it is recommended to have a GPU for computational acceleration. When training on Google Colab, it's advisable to use an A100 GPU as V100 might not have sufficient memory.

### Prerequisites

- Ensure that you have installed the necessary libraries by running the following command:
```bash
pip install transformers datasets sacrebleu
```

### Usage

1. Customize your training configuration by creating a JSON file (e.g., `config/iva_mt_wslot-m2m100_418M-en-pl.json`). In this file, specify the source language, target language, learning rate, weight decay, number of training epochs, and other relevant parameters.

2. Execute the training script by running the following command:
```bash
python train.py --config config/iva_mt_wslot-m2m100_418M-en-pl.json
```

### Configuration File

The configuration file should contain the following parameters:

- `src_lang`: Source language code (e.g., "en" for English).
- `tgt_lang`: Target language code (e.g., "pl" for Polish).
- `learning_rate`: Learning rate for the optimizer.
- `weight_decay`: Weight decay for the optimizer.
- `num_train_epochs`: Number of training epochs.
- `model_space`: The namespace for the model.
- `model_name`: The name of the model.
- `dataset`: The name of the dataset to be used for training.

Example Configuration:
```json
{
    "src_lang": "en",
    "tgt_lang": "pl",
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "model_space": "facebook",
    "model_name": "m2m100_418M",
    "dataset": "wmt16"
}
```

### Running on Google Colab

If you are running the script on Google Colab, ensure to switch to a runtime with a GPU for better performance. It is recommended to use an A100 GPU as V100 might have memory limitations depending on the size of the model and the dataset.
