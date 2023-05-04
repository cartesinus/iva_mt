#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a class for translating and generating multiple variants of input text using
the M2M100 model from Hugging Face's Transformers library. It can be used for generating translations
with different verb alternatives and validating translations with proper tag handling.
"""

import json
import re
import string
from os import listdir
from os.path import isfile, join
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, PhrasalConstraint


def check_tags(src_utt, tgt_utt):
    """
    Check if the target utterance (tgt_utt) contains the same tags as the source utterance (src_utt)
    and if both opening and closing tags are present.

    Args:
    src_utt (str): Source utterance.
    tgt_utt (str): Target utterance.

    Returns:
    bool: True if all tags in the source utterance are present and properly formatted in the target
    utterance, False otherwise.
    """
    for tag_letter in list(string.ascii_lowercase):
        tag = "<" + tag_letter + ">"
        if tag in src_utt:
            if not tag in tgt_utt:
                return False
            #check if both opening and closing tags are present
            if len(re.findall(tag, tgt_utt)) != 2:
                return False

    return True


class iva_mt:
    """
    Class for generating single and multiple translations of input text using the M2M100 model.
    Supports generating translations with different verb alternatives and validating translations
    with proper tag handling.

    Example usage:
    iva_mt = iva_mt("pl")
    iva_mt.translate("set the temperature on <a>my<a> thermostat")
    iva_mt.generate_alternative_translations("set the temperature on <a>my<a> thermostat")
    """
    def __init__(self, lang):
        """
        Initialize the iva_mt class with the specified target language (lang).

        Args:
        lang (str): Target language code (e.g. "pl" for Polish).
        """
        model_name = "cartesinus/iva_mt_wslot-m2m100_418M-en-" + lang
        self.lang = lang
        self.verb_ont = []
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang="en", tgt_lang=lang)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    def translate(self, input_text):
        """
        Generate a single translation for a given input text.

        Args:
        input_text (str): Source text to translate.

        Returns:
        list: A list containing a single translated string.
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt")
        lang_id = self.tokenizer.get_lang_id(self.lang)
        generated_tokens = self.model.generate(**input_ids, forced_bos_token_id=lang_id)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def simple_verb_sub(self, sentence_src, sentence_tgt):
        """
        Generate alternative translations of the target sentence (sentence_tgt) by substituting
        verbs from the source sentence (sentence_src) with their alternatives.

        Args:
        sentence_src (str): Source sentence.
        sentence_tgt (str): Target sentence to generate alternatives for.

        Returns:
        list: A list of alternative translations with verb substitutions.
        """
        subs = []
        src_verb = [w for w in sentence_src.split() if w in self.verb_ont]
        tgt_verbs = []
        if src_verb:
            tgt_verbs = [w for w in sentence_tgt.split() if w in self.verb_ont[src_verb[0]]]

        if tgt_verbs:
            for verb in tgt_verbs:
                for v in self.verb_ont[src_verb[0]]:
                    if v != verb:
                        variant = re.sub(verb, v, sentence_tgt)
                        if variant not in subs:
                            subs.append(variant)

        return subs

    def generate_alternative_translations(self, input_text):
        """
        Generate multiple translation variants for the given input text using verb alternatives
        from the verb ontology and simple verb substitution.

        Args:
        input_text (str): Source text to translate.

        Returns:
        list: A list of alternative translations.
        """
        # Load verb ontology if not already loaded
        if not self.verb_ont:
            self.load_verb_ontology()

        lang_id = self.tokenizer.get_lang_id(self.lang)

		# Get single translation and add it to the alternatives list
        single_trans = self.translate(input_text)
        alternatives = [single_trans[0]]

		# Generate alternative translations using simple verb substitution
        verb_alternatives = self.get_verb_alternatives(input_text)
        alternatives.extend(self.simple_verb_sub(input_text, single_trans[0]))
        for constraint in verb_alternatives:
            input_ids = self.tokenizer(input_text, return_tensors="pt")
            generated_tokens = self.model.generate(**input_ids,
                                                   forced_bos_token_id=lang_id,
                                                   #constraints=constraints,
                                                   force_words_ids=constraint,
                                                   num_beams=5,
                                                   num_return_sequences=1,
                                                   no_repeat_ngram_size=2,
                                                   remove_invalid_values=True)
            variant = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if variant not in alternatives \
               and abs(len(input_text.split()) - len(variant[0].split())) <= 2 \
               and check_tags(input_text, variant):
                alternatives.append(variant[0])

        return alternatives

    def load_verb_ontology(self):
        """
        Load the verb ontology from a JSON file located in the 'verb_translations' directory.
        The file should have a name starting with 'en2' followed by the target language code.
        The loaded ontology will be stored in the 'verb_ont' attribute.
        """
        ont_path = 'verb_translations/'
        ont_files = [f for f in listdir(ont_path) if isfile(join(ont_path, f)) and f.startswith('en2' + self.lang) ]
        with open(ont_path + ont_files[0], "r") as f:
            json_data = json.load(f)
        self.verb_ont = json_data

    def get_verb_alternatives(self, sentence):
        """
        Retrieve verb alternatives for the verbs found in the input sentence using the verb ontology.
        The verb alternatives are returned as forced word IDs that can be used as constraints during
        translation.

        Args:
        sentence (str): The input sentence for which to find verb alternatives.

        Returns:
        list: A list of lists containing the forced word IDs for each verb alternative found in the
        verb ontology.
        """
        constrains = []
        verb = [w for w in sentence.split() if w in self.verb_ont]
        for alternative in self.verb_ont[verb[0]]:
            force_words_ids = self.tokenizer([alternative], add_special_tokens=False).input_ids
            constrains.append(force_words_ids)

        return constrains

    def generate_unconstrained_translations(self, input_text, num_variants=5):
        """
        Generate multiple translations for a given input text using beam search without constraints.
        This method does not utilize the verb ontology or specific constraints for generating
        translations.

        Args:
        input_text (str): Source text to translate.
        num_variants (int, optional): Number of alternative translations to generate. Defaults to 5.

        Returns:
        list: A list of alternative translations generated using beam search.
        """
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids,
                                      num_beams=num_variants,
                                      num_return_sequences=num_variants,
                                      forced_bos_token_id=self.tokenizer.get_lang_id(self.lang))
        translations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return translations

