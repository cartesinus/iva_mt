import unittest
from iva_mt.iva_mt import IVAMT

class TestIVAMT(unittest.TestCase):

    def setUp(self):
        self.translator = IVAMT("pl")

    def test_load_verb_ontology(self):
        # Load verb ontology
        self.translator.load_verb_ontology()

        # Check if verb_ont attribute is loaded with data
        self.assertTrue(len(self.translator.verb_ont) > 0)

        # Check if verb_ont is a dictionary
        self.assertIsInstance(self.translator.verb_ont, dict)

        # Check if the keys in verb_ont are strings
        self.assertTrue(all(isinstance(key, str) for key in self.translator.verb_ont.keys()))

        # Check if the values in verb_ont are lists
        self.assertTrue(all(isinstance(value, list) for value in self.translator.verb_ont.values()))

    def test_translate(self):
        input_text = "set the temperature on <a>my<a> thermostat"
        translation = self.translator.translate(input_text)
        self.assertIsNotNone(translation)

    def test_generate_alternative_translations(self):
        input_text = "set the temperature on <a>my<a> thermostat"
        translations = self.translator.generate_alternative_translations(input_text)
        self.assertIsNotNone(translations)
        self.assertTrue(len(translations) > 1)

    def test_generate_unconstrained_translations(self):
        input_text = "set the temperature on <a>my<a> thermostat"
        translations = self.translator.generate_unconstrained_translations(input_text)
        self.assertIsNotNone(translations)
        self.assertTrue(len(translations) > 1)

if __name__ == '__main__':
    unittest.main()

