#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a class for translating and generating multiple variants of input text using
the M2M100 model from Hugging Face's Transformers library. It can be used for generating
translations with different verb alternatives and validating translations with proper tag handling.
"""

import os
import re
import json
import string
import tarfile
from typing import List
from os import listdir
from os.path import isfile, join

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, PhrasalConstraint
from peft import PeftModel, PeftConfig


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
    tags = ["<" + tag_letter + ">" for tag_letter in string.ascii_lowercase]
    tags_in_src = [tag for tag in tags if tag in src_utt]

    return all(
        tag in tgt_utt and len(re.findall(tag, tgt_utt)) == 2
        for tag in tags_in_src
    )


class IVAMT:
    """
    Class for generating single and multiple translations of input text using the M2M100 model.
    Supports generating translations with different verb alternatives and validating translations
    with proper tag handling.

    Example usage:
    iva_mt = iva_mt("pl")
    iva_mt.translate("set the temperature on <a>my<a> thermostat")
    iva_mt.generate_alternative_translations("set the temperature on <a>my<a> thermostat")
    """
    def __init__(
            self, tgt_lang, src_lang='en', device="cpu", batch_size=1, model_name="iva_mt",
            peft_model_id=None
        ):
        """
        Initialize the IVAMT class with the specified target language (lang), and optionally load a
        PEFT-trained model.

        Args:
        lang (str): Target language code (e.g. "pl" for Polish).
        device (str, optional): Device used for inference, e.g., "cuda:0". Defaults to "cpu".
        batch_size (int, optional): Batch size used for inference. Defaults to 1.
        model_name (str, optional): HF model name (e.g., "facebook/m2m100_418M"). By default, set to
            custom models ("iva_mt").
        peft_model_id (str, optional): Identifier of the PEFT-trained model on the Hugging Face
            Model Hub or the path to a local directory containing the PEFT-trained model files.
            Defaults to None. If provided, a PEFT-trained model is loaded for inference.

        Raises:
        RuntimeError: If model and tokenizer loading fails.

        Attributes:
        lang (str): Target language code.
        device (torch.device): Device used for inference.
        batch_size (int): Batch size used for inference.
        tokenizer (transformers.M2M100Tokenizer): Tokenizer for the M2M100 model.
        model (transformers.M2M100ForConditionalGeneration or PeftModel): M2M100 model for
            translation. If peft_model_id is provided, a PEFT-trained model is used.

        Example:
        iva_mt = IVAMT(
            "pl", device="cuda:0", peft_model_id="stevhliu/roberta-large-lora-token-classification"
        )
        """
        tokenizer_model = model_name
        if model_name == "iva_mt":
            model_name = f"cartesinus/iva_mt_wslot-m2m100_418M-{src_lang}-{tgt_lang}"
            tokenizer_model = model_name
        elif model_name.endswith('.tgz'):
            tokenizer_model = f"cartesinus/iva_mt_wslot-m2m100_418M-{src_lang}-{tgt_lang}"
            model_name = self._unpack_model_if_archive(model_name)

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.verb_ont = []
        self.device = torch.device(device)

        try:
            if peft_model_id:
                peft_config = PeftConfig.from_pretrained(peft_model_id)
                self.inference_model = M2M100ForConditionalGeneration.from_pretrained(
                    peft_config.base_model_name_or_path
                )
                self.tokenizer = M2M100Tokenizer.from_pretrained(
                    peft_config.base_model_name_or_path, src_lang=src_lang, tgt_lang=tgt_lang
                )
                self.model = PeftModel.from_pretrained(self.inference_model, peft_model_id)
            else:
                print(f"Loading tokenizer: {tokenizer_model}")
                self.tokenizer = M2M100Tokenizer.from_pretrained(
                    tokenizer_model, src_lang=src_lang, tgt_lang=tgt_lang
                )
                print(f"Loading model: {model_name}")
                self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model and tokenizer: {str(e)}")

        self.model.to(self.device)
        self.batch_size = batch_size

    def translate(self, input_text):
        """
        Generate a single translation for a given input text. Input text can be string or list of
        strings.

        Args:
        input_text (str or list(str)): Source text/s to translate.

        Returns:
        list: A list containing a single translated string for each element of input_text.
        """
        input_ids = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids.to(self.device)
        lang_id = self.tokenizer.get_lang_id(self.tgt_lang)
        generated_tokens = self.model.generate(**input_ids, forced_bos_token_id=lang_id)

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def batch_translate(self, input_texts: List[str]):
        """
        Translate utterances in batch
        Args:
            input_texts: List of input strings

        Returns:
        list: A list containing a single translated string for each element of input_text.
        """
        output = []
        for input_text in [input_texts[x:x + self.batch_size] for x in \
                range(0, len(input_texts), self.batch_size)]:
            output.extend(self.translate(input_text))
        return output

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
        src_verb = next((w for w in sentence_src.split() if w in self.verb_ont), None)

        if src_verb:
            tgt_verbs = [w for w in sentence_tgt.split() if w in self.verb_ont[src_verb]]
            return list(set(
                re.sub(verb, v, sentence_tgt)
                for verb in tgt_verbs
                for v in self.verb_ont[src_verb]
                if v != verb
            ))
        else:
            return []

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

        lang_id = self.tokenizer.get_lang_id(self.src_lang)

		# Get single translation and add it to the alternatives list
        single_trans = self.translate(input_text)
        alternatives = [single_trans[0]]

		# Generate alternative translations using simple verb substitution
        verb_alternatives = self.get_verb_alternatives(input_text)
        alternatives.extend(self.simple_verb_sub(input_text, single_trans[0]))
        for constraint in verb_alternatives:
            input_ids = self.tokenizer(input_text, return_tensors="pt")
            input_ids.to(self.device)
            generated_tokens = self.model.generate(**input_ids,
                                                   forced_bos_token_id=lang_id,
                                                   #constraints=constraints,
                                                   force_words_ids=constraint,
                                                   num_beams=5,
                                                   num_return_sequences=1,
                                                   no_repeat_ngram_size=2,
                                                   remove_invalid_values=True)
            variant = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if variant[0] not in alternatives \
               and abs(len(input_text.split()) - len(variant[0].split())) <= 2 \
               and check_tags(input_text, variant[0]):
                alternatives.append(variant[0])

        return alternatives

    def load_verb_ontology(self):
        """
        Load the verb ontology from a JSON file located in the 'verb_translations' directory.
        The file should have a name starting with 'en2' followed by the target language code.
        The loaded ontology will be stored in the 'verb_ont' attribute.
        """
        ont_path = os.path.join(os.path.dirname(__file__),
                                '../data/verb_translations/en2' + self.src_lang + '/')
        ont_files = [f for f in os.listdir(ont_path)
                     if os.path.isfile(os.path.join(ont_path, f))
                     and f.startswith('en2' + self.src_lang)]

        # Sort the list of files in descending order, so the highest version number is first element
        ont_files.sort(reverse=True)

        with open(os.path.join(ont_path, ont_files[0]), "r") as f:
            json_data = json.load(f)

        self.verb_ont = json_data

    def get_verb_translation(self, lang, verb):
        """
        Retrieve the translation of a given verb in the specified target language.

        Args:
        lang (str): The target language code (e.g., 'pl', 'fr', 'es', etc.).
        verb (str): The verb in the source language (English) for which to find translations.

        Returns:
        list: A list of verb translations in the target language if the verb is found in the verb
            ontology; an empty list otherwise.

        Note:
        The method will load the verb ontology if it has not been loaded already.
        """
        if not self.verb_ont:
            self.load_verb_ontology()

        if verb in self.verb_ont:
            return self.verb_ont[verb]
        else:
            return []

    def get_verb_alternatives(self, sentence):
        """
        Retrieve verb alternatives for the verbs found in the input sentence using the verb
        ontology. The verb alternatives are returned as forced word IDs that can be used as
        constraints during translation.

        Args:
        sentence (str): The input sentence for which to find verb alternatives.

        Returns:
        list: A list of lists containing the forced word IDs for each verb alternative found in the
        verb ontology.
        """
        # Find the verb in the sentence that is present in the verb ontology
        verb = next((w for w in sentence.split() if w in self.verb_ont), None)

        # If a verb is found, generate a list of forced word IDs for each alternative
        if verb:
            return [
                self.tokenizer([alternative], add_special_tokens=False).input_ids
                for alternative in self.verb_ont[verb]
            ]
        else:
            return []

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
                                      forced_bos_token_id=self.tokenizer.get_lang_id(self.src_lang))
        translations = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return translations

    def _unpack_model_if_archive(self, model_path: str) -> str:
        """
        Checks if the model_path is a .tgz archive, and if so, unpacks it.

        Args:
            model_path (str): Path to the model archive or directory.

        Returns:
            str: Path to the unpacked model directory.
        """
        necessary_files = ['config.json', 'pytorch_model.bin']
        filename = os.path.basename(model_path)
        model_name = filename.split('-')[:-1]
        model_name = '-'.join(model_name)

        cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--tcl--" + model_name)
        if os.path.exists(cache_path) and all(
                os.path.isfile(os.path.join(cache_path, f)) for f in necessary_files):
            print(f"Extraction path '{cache_path}' already exists. Skipping extraction.")
            return cache_path

        os.makedirs(cache_path, exist_ok=True)
        print(f"Extracting local model to {cache_path}")
        with tarfile.open(model_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    print(f"Extracting {member.name} to {cache_path}")
                    # Extract member to extraction_path directly, ignoring its original path
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=cache_path)

        return cache_path
