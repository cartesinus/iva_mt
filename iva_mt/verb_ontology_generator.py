#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates a JSON file containing translations of verbs in the context of virtual assistants. It takes an input TSV file containing verb-to-VerbNet class mappings and outputs a JSON file with translations of those verbs in the specified target language. The script utilizes the VerbNet and WordNet libraries to process the input and generate translations.

Example usage:

python verb_ontology_generator.py -i input.tsv -o output.json -l es

This command reads the verb-to-VerbNet class mappings from 'input.tsv', generates Spanish translations of the verbs, and writes the translations to 'output.json'.
"""

import argparse
import json
import os
import csv
import pandas as pd
import re
import string
import xml.etree.ElementTree as ET
from nltk.corpus import verbnet
from nltk.corpus import wordnet as wn


def get_wn_class_from_vn(verb, verbnet_class):
    """
    Given a verb and its VerbNet class, returns a list of corresponding WordNet classes.

    Args:
        verb (str): The verb to find WordNet classes for.
        verbnet_class (str): The VerbNet class of the given verb.

    Returns:
        list: A list of WordNet classes corresponding to the given verb and VerbNet class.
    """
    result = []

    if verbnet_class:
        vn_class = verbnet.vnclass(verbnet_class)

        for m in vn_class.findall('./MEMBERS/MEMBER'):
            if 'name' in m.attrib and m.attrib['name'] == verb:
                result = m.attrib['wn'].split()
                break

    return result


def get_word_syset_in_lang(vn_str, lang):
    """
    Given a WordNet class string and a language code, returns a list of synset lemmas in the specified language.

    Args:
        vn_str (str): The WordNet class string.
        lang (str): The target language code (e.g. 'en', 'es', 'fr').

    Returns:
        list: A list of synset lemmas in the specified language corresponding to the given WordNet class string.
    """
    if vn_str in ['deactive%2:30:00'] or vn_str == '':
        return []
    elif vn_str.startswith('?'):
        vn_str = vn_str[1:]
    return wn.synset_from_sense_key(vn_str + "::").lemmas(lang)


def get_translations_for_verb(verb, verbnet_classes, lang):
    """
    Given a verb map row and a language code, returns the verb and its translations in the specified language.

    Args:
        row (pd.Series): A row from the verb_map DataFrame.
        lang (str): The target language code (e.g. 'en', 'es', 'fr').

    Returns:
        tuple: A tuple containing the verb (str) and a list of its translations (list of str) in the specified language.
    """
    translations = []

    for vn_class in verbnet_classes.split(','):
        if vn_class != '-':  # this is label when there is no verb in verbnet
            for wn_synset in get_wn_class_from_vn(verb, vn_class):
                for wn_lemma in get_word_syset_in_lang(wn_synset, lang):
                    if wn_lemma.name() not in translations:
                        translations.append(wn_lemma.name())

    return translations


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser(description='Extract all Lemmas in given language from VerbNet class (Levin class).')
    arg_parse.add_argument('-i', '--input', help='')
    arg_parse.add_argument('-o', '--output', help='')
    arg_parse.add_argument('-l', '--lang', help='')
    args = arg_parse.parse_args()

    verb_map = pd.read_csv(args.input, sep='\t', keep_default_na=False)

    verb_trans = {}
    for idx, row in verb_map.iterrows():
        verb = row['verb']
        translations = get_translations_for_verb(verb, row['verbnet_class'], args.lang)
        if translations:
            verb_trans[verb] = translations

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(verb_trans, f, ensure_ascii=False, indent=4)
