Multiverb IVA MT
================
Generating diverse verb variants with VerbNet and Conditional Beam Search for enhanced performance of Intelligent Virtual Assistants (IVA) training set translation.

Usage:
```python
import iva_mt

iva_mt = iva_mt("pl")
iva_mt.translate("set the temperature on <a>my<a> thermostat")
iva_mt.generate_alternative_translations("set the temperature on <a>my<a> thermostat")
```
