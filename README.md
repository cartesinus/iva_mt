Multiverb IVA MT
================
Generating diverse verb variants with VerbNet and Conditional Beam Search for enhanced performance of Intelligent Virtual Assistants (IVA) training set translation.

Usage:
```python
from iva_mt.iva_mt import IVAMT

translator = IVAMT("pl")
translator.translate("set the temperature on <a>my<a> thermostat")
translator.generate_alternative_translations("set the temperature on <a>my<a> thermostat")
```
