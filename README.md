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
