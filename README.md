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
Available languages (en2xx): **pl, es, de, fr, pt, sv, zh, ja, tr, hi**

To use GPU and batching, provide information about device:
```python
IVAMT("pl", device="cuda:0", batch_size=16)
```
On V100 this allows to translate ~100 sentences/minute.
