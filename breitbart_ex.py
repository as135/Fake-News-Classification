#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Author: Kelly Geyer, klg2@rice.edu
Date: May 26, 2017

Description: Script for collecting articles from https://newsapi.org/. This example focuses on breitbart-news-api
'''


import os, json, uuid, re
import datetime
import urllib2
from bs4 import BeautifulSoup


def create_utf8_rewrite_hash():
    """
    This function rewrites UTF-8 characters to ASCII in a rational manner. Strictly speaking (and in python) any ascii
    character >= 128 is not valid
    :return rewrite_hash: Dictionary object where keys are UTF-8 characters and values are corresponding ASCII values
    """
    rewrite_hash = dict([])
    rewrite_hash[u'\xA0'] = " "  # NO-BREAK SPACE
    rewrite_hash[u'\xA1'] = " "  # INVERTED EXCLAMATION MARK
    rewrite_hash[u'\xA2'] = " cents "  # CENT SIGNS
    rewrite_hash[u'\xA3'] = " pounds "  # POUND SIGN
    rewrite_hash[u'\xA4'] = " "  # CURRENCY SIGN
    rewrite_hash[u'\xA5'] = " yen "  # YEN SIGN
    rewrite_hash[u'\xA6'] = " "  # BROKEN BAR
    rewrite_hash[u'\xA7'] = " "  # SECTION SIGN
    rewrite_hash[u'\xA8'] = " "  # DIAERESIS
    rewrite_hash[u'\xA9'] = " "  # COPYRIGHT SIGN
    rewrite_hash[u'\xAA'] = " "  # FEMININE ORDINAL INDICATOR
    rewrite_hash[u'\xAB'] = " "  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    rewrite_hash[u'\xAC'] = " "  # NOT SIGN
    rewrite_hash[u'\xAD'] = " "  # SOFT HYPHEN
    rewrite_hash[u'\xAE'] = " "  # REGISTERED SIGN
    rewrite_hash[u'\xAF'] = " "  # MACRON
    rewrite_hash[u'\xB0'] = " degrees "  # DEGREE SIGN
    rewrite_hash[u'\xB1'] = " plus-or-minus "  # PLUS-MINUS SIGN
    rewrite_hash[u'\xB2'] = " "  # SUPERSCRIPT TWO
    rewrite_hash[u'\xB3'] = " ";  # SUPERSCRIPT THREE
    rewrite_hash[u'\xB4'] = "'"  # ACUTE ACCENT
    rewrite_hash[u'\xB5'] = " micro "  # MICRO SIGN
    rewrite_hash[u'\xB6'] = " "  # PILCROW SIGN
    rewrite_hash[u'\xB7'] = " "  # MIDDLE DOT
    rewrite_hash[u'\xB8'] = " "  # CEDILLA
    rewrite_hash[u'\xB9'] = " "  # SUPERSCRIPT ONE
    rewrite_hash[u'\xBA'] = " "  # MASCULINE ORDINAL INDICATOR
    rewrite_hash[u'\xBB'] = " "  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    rewrite_hash[u'\xBC'] = " 1/4 "  # VULGAR FRACTION ONE QUARTER
    rewrite_hash[u'\xBD'] = " 1/2 "  # VULGAR FRACTION ONE HALF
    rewrite_hash[u'\xBE'] = " 3/4 "  # VULGAR FRACTION THREE QUARTERS
    rewrite_hash[u'\xBF'] = " "  # INVERTED QUESTION MARK

    rewrite_hash[u'\xC0'] = "A"  # LATIN CAPITAL LETTER A WITH GRAVE
    rewrite_hash[u'\xC1'] = "A"  # LATIN CAPITAL LETTER A WITH ACUTE
    rewrite_hash[u'\xC2'] = "A"  # LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    rewrite_hash[u'\xC3'] = "A"  # LATIN CAPITAL LETTER A WITH TILDE
    rewrite_hash[u'\xC4'] = "A"  # LATIN CAPITAL LETTER A WITH DIAERESIS
    rewrite_hash[u'\xC5'] = "A"  # LATIN CAPITAL LETTER A WITH RING ABOVE
    rewrite_hash[u'\xC6'] = "AE"  # LATIN CAPITAL LETTER AE
    rewrite_hash[u'\xC7'] = "C"  # LATIN CAPITAL LETTER C WITH CEDILLA
    rewrite_hash[u'\xC8'] = "E"  # LATIN CAPITAL LETTER E WITH GRAVE
    rewrite_hash[u'\xC9'] = "E"  # LATIN CAPITAL LETTER E WITH ACUTE
    rewrite_hash[u'\xCA'] = "E"  # LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    rewrite_hash[u'\xCB'] = "E"  # LATIN CAPITAL LETTER E WITH DIAERESIS
    rewrite_hash[u'\xCC'] = "I"  # LATIN CAPITAL LETTER I WITH GRAVE
    rewrite_hash[u'\xCD'] = "I"  # LATIN CAPITAL LETTER I WITH ACUTE
    rewrite_hash[u'\xCE'] = "I"  # LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    rewrite_hash[u'\xCF'] = "I"  # LATIN CAPITAL LETTER I WITH DIAERESIS

    rewrite_hash[u'\xD0'] = "Th"  # LATIN CAPITAL LETTER ETH
    rewrite_hash[u'\xD1'] = "N"  # LATIN CAPITAL LETTER N WITH TILDE
    rewrite_hash[u'\xD2'] = "O"  # LATIN CAPITAL LETTER O WITH GRAVE
    rewrite_hash[u'\xD3'] = "O"  # LATIN CAPITAL LETTER O WITH ACUTE
    rewrite_hash[u'\xD4'] = "O"  # LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    rewrite_hash[u'\xD5'] = "O"  # LATIN CAPITAL LETTER O WITH TILDE
    rewrite_hash[u'\xD6'] = "O"  # LATIN CAPITAL LETTER O WITH DIAERESIS
    rewrite_hash[u'\xD7'] = "x"  # MULTIPLICATION SIGN
    rewrite_hash[u'\xD8'] = "O"  # LATIN CAPITAL LETTER O WITH STROKE
    rewrite_hash[u'\xD9'] = "U"  # LATIN CAPITAL LETTER U WITH GRAVE
    rewrite_hash[u'\xDA'] = "U"  # LATIN CAPITAL LETTER U WITH ACUTE
    rewrite_hash[u'\xDB'] = "U"  # LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    rewrite_hash[u'\xDC'] = "U"  # LATIN CAPITAL LETTER U WITH DIAERESIS
    rewrite_hash[u'\xDD'] = "Y"  # LATIN CAPITAL LETTER Y WITH ACUTE
    rewrite_hash[u'\xDE'] = "Th"  # LATIN CAPITAL LETTER THORN
    rewrite_hash[u'\xDF'] = "ss"  # LATIN SMALL LETTER SHARP S

    rewrite_hash[u'\xE0'] = "a"  # LATIN SMALL LETTER A WITH GRAVE
    rewrite_hash[u'\xE1'] = "a"  # LATIN SMALL LETTER A WITH ACUTE
    rewrite_hash[u'\xE2'] = "a"  # LATIN SMALL LETTER A WITH CIRCUMFLEX
    rewrite_hash[u'\xE3'] = "a"  # LATIN SMALL LETTER A WITH TILDE
    rewrite_hash[u'\xE4'] = "a"  # LATIN SMALL LETTER A WITH DIAERESIS
    rewrite_hash[u'\xE5'] = "a"  # LATIN SMALL LETTER A WITH RING ABOVE
    rewrite_hash[u'\xE6'] = "ae"  # LATIN SMALL LETTER AE
    rewrite_hash[u'\xE7'] = "c"  # LATIN SMALL LETTER C WITH CEDILLA
    rewrite_hash[u'\xE8'] = "e"  # LATIN SMALL LETTER E WITH GRAVE
    rewrite_hash[u'\xE9'] = "e"  # LATIN SMALL LETTER E WITH ACUTE
    rewrite_hash[u'\xEA'] = "e"  # LATIN SMALL LETTER E WITH CIRCUMFLEX
    rewrite_hash[u'\xEB'] = "e"  # LATIN SMALL LETTER E WITH DIAERESIS
    rewrite_hash[u'\xEC'] = "i"  # LATIN SMALL LETTER I WITH GRAVE
    rewrite_hash[u'\xED'] = "i"  # LATIN SMALL LETTER I WITH ACUTE
    rewrite_hash[u'\xEE'] = "i"  # LATIN SMALL LETTER I WITH CIRCUMFLEX
    rewrite_hash[u'\xEF'] = "i"  # LATIN SMALL LETTER I WITH DIAERESIS

    rewrite_hash[u'\xF0'] = "th"  # LATIN SMALL LETTER ETH
    rewrite_hash[u'\xF1'] = "n"  # LATIN SMALL LETTER N WITH TILDE
    rewrite_hash[u'\xF2'] = "o"  # LATIN SMALL LETTER O WITH GRAVE
    rewrite_hash[u'\xF3'] = "o"  # LATIN SMALL LETTER O WITH ACUTE
    rewrite_hash[u'\xF4'] = "o"  # LATIN SMALL LETTER O WITH CIRCUMFLEX
    rewrite_hash[u'\xF5'] = "o"  # LATIN SMALL LETTER O WITH TILDE
    rewrite_hash[u'\xF6'] = "o"  # LATIN SMALL LETTER O WITH DIAERESIS
    rewrite_hash[u'\xF7'] = " divided by "  # DIVISION SIGN
    rewrite_hash[u'\xF8'] = "o"  # LATIN SMALL LETTER O WITH STROKE
    rewrite_hash[u'\xF9'] = "u"  # LATIN SMALL LETTER U WITH GRAVE
    rewrite_hash[u'\xFA'] = "u"  # LATIN SMALL LETTER U WITH ACUTE
    rewrite_hash[u'\xFB'] = "u"  # LATIN SMALL LETTER U WITH CIRCUMFLEX
    rewrite_hash[u'\xFC'] = "u"  # LATIN SMALL LETTER U WITH DIAERESIS
    rewrite_hash[u'\xFD'] = "y"  # LATIN SMALL LETTER Y WITH ACUTE
    rewrite_hash[u'\xFE'] = "th"  # LATIN SMALL LETTER THORN
    rewrite_hash[u'\xFF'] = "y"  # LATIN SMALL LETTER Y WITH DIAERESIS

    rewrite_hash[u'\u0100'] = "A"  # LATIN CAPTIAL LETTER A WITH MACRON
    rewrite_hash[u'\u0101'] = "a"  # LATIN SMALL LETTER A WITH MACRON
    rewrite_hash[u'\u0102'] = "A"  # LATIN CAPITAL LETTER A WITH BREVE
    rewrite_hash[u'\u0103'] = "a"  # LATIN SMALL LETTER A WITH BREVE
    rewrite_hash[u'\u0104'] = "A"  # LATIN CAPITAL LETTER A WITH OGONEK
    rewrite_hash[u'\u0105'] = "a"  # LATIN SMALL LETTER A WITH OGONEK
    rewrite_hash[u'\u0106'] = "C"  # LATIN CAPITAL LETTER C WITH ACUTE
    rewrite_hash[u'\u0107'] = "c"  # LATIN SMALL LETTER C WITH ACUTE
    rewrite_hash[u'\u0108'] = "C"  # LATIN CAPITAL LETTER C WITH CIRCUMFLEX
    rewrite_hash[u'\u0109'] = "c"  # LATIN SMALL LETTER C WITH CIRCUMFLEX
    rewrite_hash[u'\u010A'] = "C"  # LATIN CAPITAL LETTER C WITH DOT ABOVE
    rewrite_hash[u'\u010B'] = "c"  # LATIN SMALL LETTER C WITH DOT ABOVE
    rewrite_hash[u'\u010C'] = "C"  # LATIN CAPITAL LETTER C WITH CARON
    rewrite_hash[u'\u010D'] = "c"  # LATIN SMALL LETTER C WITH CARON
    rewrite_hash[u'\u010E'] = "D"  # LATIN CAPITAL LETTER D WITH CARON
    rewrite_hash[u'\u010F'] = "d"  # LATIN SMALL LETTER D WITH CARON

    rewrite_hash[u'\u0110'] = "D"  # LATIN CAPITAL LETTER D WITH STROKE
    rewrite_hash[u'\u0111'] = "d"  # LATIN SMALL LETTER D WITH STROKE
    rewrite_hash[u'\u0112'] = "E"  # LATIN CAPITAL LETTER E WITH MACRON
    rewrite_hash[u'\u0113'] = "e"  # LATIN SMALL LETTER E WITH MACRON
    rewrite_hash[u'\u0114'] = "E"  # LATIN CAPITAL LETTER E WITH BREVE
    rewrite_hash[u'\u0115'] = "e"  # LATIN SMALL LETTER E WITH BREVE
    rewrite_hash[u'\u0116'] = "E"  # LATIN CAPITAL LETTER E WITH DOT ABOVE
    rewrite_hash[u'\u0117'] = "e"  # LATIN SMALL LETTER E WITH DOT ABOVE
    rewrite_hash[u'\u0118'] = "E"  # LATIN CAPITAL LETTER E WITH OGONEK
    rewrite_hash[u'\u0119'] = "e"  # LATIN SMALL LETTER E WITH OGONEK
    rewrite_hash[u'\u011A'] = "E"  # LATIN CAPITAL LETTER E WITH CARON
    rewrite_hash[u'\u011B'] = "e"  # LATIN SMALL LETTER E WITH CARON
    rewrite_hash[u'\u011C'] = "G"  # LATIN CAPITAL LETTER G WITH CIRCUMFLEX
    rewrite_hash[u'\u011D'] = "g"  # LATIN SMALL LETTER G WITH CIRCUMFLEX
    rewrite_hash[u'\u011E'] = "G"  # LATIN CAPITAL LETTER G WITH BREVE
    rewrite_hash[u'\u011F'] = "g"  # LATIN SMALL LETTER G WITH BREVE

    rewrite_hash[u'\u0120'] = "G"  # LATIN CAPITAL LETTER G WITH DOT ABOVE
    rewrite_hash[u'\u0121'] = "g"  # LATIN SMALL LETTER G WITH DOT ABOVE
    rewrite_hash[u'\u0122'] = "G"  # LATIN CAPITAL LETTER G WITH CEDILLA
    rewrite_hash[u'\u0123'] = "g"  # LATIN SMALL LETTER G WITH CEDILLA
    rewrite_hash[u'\u0124'] = "H"  # LATIN CAPITAL LETTER H WITH CIRCUMFLEX
    rewrite_hash[u'\u0125'] = "h"  # LATIN SMALL LETTER H WITH CIRCUMFLEX
    rewrite_hash[u'\u0126'] = "H"  # LATIN CAPITAL LETTER H WITH STROKE
    rewrite_hash[u'\u0127'] = "h"  # LATIN SMALL LETTER H WITH STROKE
    rewrite_hash[u'\u0128'] = "I"  # LATIN CAPITAL LETTER I WITH TILDE
    rewrite_hash[u'\u0129'] = "i"  # LATIN SMALL LETTER I WITH TILDE
    rewrite_hash[u'\u012A'] = "I"  # LATIN CAPITAL LETTER I WITH MACRON
    rewrite_hash[u'\u012B'] = "i"  # LATIN SMALL LETTER I WITH MACRON
    rewrite_hash[u'\u012C'] = "I"  # LATIN CAPITAL LETTER I WITH BREVE
    rewrite_hash[u'\u012D'] = "i"  # LATIN SMALL LETTER I WITH BREVE
    rewrite_hash[u'\u012E'] = "I"  # LATIN CAPITAL LETTER I WITH OGONEK
    rewrite_hash[u'\u012F'] = "i"  # LATIN SMALL LETTER I WITH OGONEK

    rewrite_hash[u'\u0130'] = "I"  # LATIN CAPITAL LETTER I WITH DOT ABOVE
    rewrite_hash[u'\u0131'] = "i"  # LATIN SMALL LETTER DOTLESS I
    rewrite_hash[u'\u0132'] = "IJ"  # LATIN CAPITAL LIGATURE IJ
    rewrite_hash[u'\u0133'] = "ij"  # LATIN SMALL LIGATURE IJ
    rewrite_hash[u'\u0134'] = "J"  # LATIN CAPITAL LETTER J WITH CIRCUMFLEX
    rewrite_hash[u'\u0135'] = "j"  # LATIN SMALL LETTER J WITH CIRCUMFLEX
    rewrite_hash[u'\u0136'] = "K"  # LATIN CAPITAL LETTER K WITH CEDILLA
    rewrite_hash[u'\u0137'] = "k"  # LATIN SMALL LETTER K WITH CEDILLA
    rewrite_hash[u'\u0138'] = "k"  # LATIN SMALL LETTER KRA
    rewrite_hash[u'\u0139'] = "L"  # LATIN CAPITAL LETTER L WITH ACUTE
    rewrite_hash[u'\u013A'] = "l"  # LATIN SMALL LETTER L WITH ACUTE
    rewrite_hash[u'\u013B'] = "L"  # LATIN CAPITAL LETTER L WITH CEDILLA
    rewrite_hash[u'\u013C'] = "l"  # LATIN SMALL LETTER L WITH CEDILLA
    rewrite_hash[u'\u013D'] = "L"  # LATIN CAPITAL LETTER L WITH CARON
    rewrite_hash[u'\u013E'] = "l"  # LATIN SMALL LETTER L WITH CARON
    rewrite_hash[u'\u013F'] = "L"  # LATIN CAPITAL LETTER L WITH MIDDLE DOT

    rewrite_hash[u'\u0140'] = "l"  # LATIN SMALL LETTER L WITH MIDDLE DOT
    rewrite_hash[u'\u0141'] = "L"  # LATIN CAPITAL LETTER L WITH STROKE
    rewrite_hash[u'\u0142'] = "l"  # LATIN SMALL LETTER L WITH STROKE
    rewrite_hash[u'\u0143'] = "N"  # LATIN CAPITAL LETTER N WITH ACUTE
    rewrite_hash[u'\u0144'] = "n"  # LATIN SMALL LETTER N WITH ACUTE
    rewrite_hash[u'\u0145'] = "N"  # LATIN CAPITAL LETTER N WITH CEDILLA
    rewrite_hash[u'\u0146'] = "n"  # LATIN SMALL LETTER N WITH CEDILLA
    rewrite_hash[u'\u0147'] = "N"  # LATIN CAPITAL LETTER N WITH CARON
    rewrite_hash[u'\u0148'] = "n"  # LATIN SMALL LETTER N WITH CARON
    rewrite_hash[u'\u0149'] = "n"  # LATIN SMALL LETTER N PRECEDED BY APOSTROPHE
    rewrite_hash[u'\u014A'] = "N"  # LATIN CAPITAL LETTER ENG
    rewrite_hash[u'\u014B'] = "n"  # LATIN SMALL LETTER ENG
    rewrite_hash[u'\u014C'] = "O"  # LATIN CAPITAL LETTER O WITH MACRON
    rewrite_hash[u'\u014D'] = "o"  # LATIN SMALL LETTER O WITH MACRON
    rewrite_hash[u'\u014E'] = "O"  # LATIN CAPITAL LETTER O WITH BREVE
    rewrite_hash[u'\u014F'] = "o"  # LATIN SMALL LETTER O WITH BREVE

    rewrite_hash[u'\u0150'] = "O"  # LATIN CAPITAL LETTER O WITH DOUBLE ACUTE
    rewrite_hash[u'\u0151'] = "o"  # LATIN SMALL LETTER O WITH DOUBLE ACUTE
    rewrite_hash[u'\u0152'] = "oe"  # LATIN CAPITAL LIGATURE OE
    rewrite_hash[u'\u0153'] = "oe"  # LATIN SMALL LIGATURE OE
    rewrite_hash[u'\u0153'] = "R"  # LATIN CAPITAL LETTER R WITH ACUTE
    rewrite_hash[u'\u0154'] = "R"  # LATIN CAPITAL LETTER R WITH ACUTE
    rewrite_hash[u'\u0155'] = "r"  # LATIN SMALL LETTER R WITH ACUTE
    rewrite_hash[u'\u0156'] = "R"  # LATIN CAPITAL LETTER R WITH CEDILLA
    rewrite_hash[u'\u0157'] = "r"  # LATIN SMALL LETTER R WITH CEDILLA
    rewrite_hash[u'\u0158'] = "R"  # LATIN CAPITAL LETTER R WITH CARON
    rewrite_hash[u'\u0159'] = "r"  # LATIN SMALL LETTER R WITH CARON
    rewrite_hash[u'\u015A'] = "S"  # LATIN CAPITAL LETTER S WITH ACUTE
    rewrite_hash[u'\u015B'] = "s"  # LATIN SMALL LETTER S WITH ACUTE
    rewrite_hash[u'\u015C'] = "S"  # LATIN CAPITAL LETTER S WITH CIRCUMFLEX
    rewrite_hash[u'\u015D'] = "s"  # LATIN SMALL LETTER S WITH CIRCUMFLEX
    rewrite_hash[u'\u015E'] = "S"  # LATIN CAPITAL LETTER S WITH CEDILLA
    rewrite_hash[u'\u015F'] = "s"  # LATIN SMALL LETTER S WITH CEDILLA

    rewrite_hash[u'\u0160'] = "S"  # LATIN CAPITAL LETTER S WITH CARON
    rewrite_hash[u'\u0161'] = "s"  # LATIN SMALL LETTER S WITH CARON
    rewrite_hash[u'\u0162'] = "T"  # LATIN CAPITAL LETTER T WITH CEDILLA
    rewrite_hash[u'\u0163'] = "t"  # LATIN SMALL LETTER T WITH CEDILLA
    rewrite_hash[u'\u0164'] = "T"  # LATIN CAPITAL LETTER T WITH CARON
    rewrite_hash[u'\u0165'] = "t"  # LATIN SMALL LETTER T WITH CARON
    rewrite_hash[u'\u0166'] = "T"  # LATIN CAPITAL LETTER T WITH STROKE
    rewrite_hash[u'\u0167'] = "t"  # LATIN SMALL LETTER T WITH STROKE
    rewrite_hash[u'\u0168'] = "U"  # LATIN CAPITAL LETTER U WITH TILDE
    rewrite_hash[u'\u0169'] = "u"  # LATIN SMALL LETTER U WITH TILDE
    rewrite_hash[u'\u016A'] = "U"  # LATIN CAPITAL LETTER U WITH MACRON
    rewrite_hash[u'\u016B'] = "u"  # LATIN SMALL LETTER U WITH MACRON
    rewrite_hash[u'\u016C'] = "U"  # LATIN CAPITAL LETTER U WITH BREVE
    rewrite_hash[u'\u016D'] = "u"  # LATIN SMALL LETTER U WITH BREVE
    rewrite_hash[u'\u016E'] = "U"  # LATIN CAPITAL LETTER U WITH RING ABOVE
    rewrite_hash[u'\u016F'] = "u"  # LATIN SMALL LETTER U WITH RING ABOVE

    rewrite_hash[u'\u0170'] = "U"  # LATIN CAPITAL LETTER U WITH DOUBLE ACUTE
    rewrite_hash[u'\u0171'] = "u"  # LATIN SMALL LETTER U WITH DOUBLE ACUTE
    rewrite_hash[u'\u0172'] = "U"  # LATIN CAPITAL LETTER U WITH OGONEK
    rewrite_hash[u'\u0173'] = "u"  # LATIN SMALL LETTER U WITH OGONEK
    rewrite_hash[u'\u0174'] = "W"  # LATIN CAPITAL LETTER W WITH CIRCUMFLEX
    rewrite_hash[u'\u0175'] = "w"  # LATIN SMALL LETTER W WITH CIRCUMFLEX
    rewrite_hash[u'\u0176'] = "Y"  # LATIN CAPITAL LETTER Y WITH CIRCUMFLEX
    rewrite_hash[u'\u0177'] = "y"  # LATIN SMALL LETTER Y WITH CIRCUMFLEX
    rewrite_hash[u'\u0178'] = "Y"  # LATIN CAPITAL LETTER Y WITH DIAERESIS
    rewrite_hash[u'\u0179'] = "Z"  # LATIN CAPITAL LETTER Z WITH ACUTE
    rewrite_hash[u'\u017A'] = "z"  # LATIN SMALL LETTER Z WITH ACUTE
    rewrite_hash[u'\u017B'] = "Z"  # LATIN CAPITAL LETTER Z WITH DOT ABOVE
    rewrite_hash[u'\u017C'] = "z"  # LATIN SMALL LETTER Z WITH DOT ABOVE
    rewrite_hash[u'\u017D'] = "Z"  # LATIN CAPITAL LETTER Z WITH CARON
    rewrite_hash[u'\u017E'] = "z"  # LATIN SMALL LETTER Z WITH CARON
    rewrite_hash[u'\u017F'] = "s"  # LATIN SMALL LETTER LONG S

    rewrite_hash[u'\u0180'] = "b"  # LATIN SMALL LETTER B WITH STROKE
    rewrite_hash[u'\u0181'] = "B"  # LATIN CAPITAL LETTER B WITH HOOK
    rewrite_hash[u'\u0182'] = "B"  # LATIN CAPITAL LETTER B WITH TOPBAR
    rewrite_hash[u'\u0183'] = "b"  # LATIN SMALL LETTER B WITH TOPBAR
    rewrite_hash[u'\u0184'] = "b"  # LATIN CAPITAL LETTER TONE SIX
    rewrite_hash[u'\u0185'] = "b"  # LATIN SMALL LETTER TONE SIX
    rewrite_hash[u'\u0186'] = "O"  # LATIN CAPITAL LETTER OPEN O
    rewrite_hash[u'\u0187'] = "C"  # LATIN CAPITAL LETTER C WITH HOOK
    rewrite_hash[u'\u0188'] = "c"  # LATIN SMALL LETTER C WITH HOOK
    rewrite_hash[u'\u0189'] = "D"  # LATIN CAPITAL LETTER AFRICAN D
    rewrite_hash[u'\u018A'] = "D"  # LATIN CAPITAL LETTER D WITH HOOK
    rewrite_hash[u'\u018B'] = "d"  # LATIN CAPITAL LETTER D WITH TOPBAR
    rewrite_hash[u'\u018C'] = "d"  # LATIN SMALL LETTER D WITH TOPBAR
    rewrite_hash[u'\u018D'] = " "  # LATIN SMALL LETTER TURNED DELTA
    rewrite_hash[u'\u018E'] = " "  # LATIN CAPITAL LETTER REVERSED E
    rewrite_hash[u'\u018F'] = " "  # LATIN CAPITAL LETTER SCHWA

    rewrite_hash[u'\u0190'] = "E"  # LATIN CAPITAL LETTER OPEN E
    rewrite_hash[u'\u0191'] = "F"  # LATIN CAPITAL LETTER F WITH HOOK
    rewrite_hash[u'\u0192'] = "f"  # LATIN SMALL LETTER F WITH HOOK
    rewrite_hash[u'\u0193'] = "G"  # LATIN CAPITAL LETTER G WITH HOOK
    rewrite_hash[u'\u0194'] = " "  # LATIN CAPITAL LETTER GAMMA
    rewrite_hash[u'\u0195'] = "hv"  # LATIN SMALL LETTER HV
    rewrite_hash[u'\u0196'] = "I"  # LATIN CAPITAL LETTER IOTA
    rewrite_hash[u'\u0197'] = "I"  # LATIN CAPITAL LETTER I WITH STROKE
    rewrite_hash[u'\u0198'] = "K"  # LATIN CAPITAL LETTER K WITH HOOK
    rewrite_hash[u'\u0199'] = "k"  # LATIN SMALL LETTER K WITH HOOK
    rewrite_hash[u'\u019A'] = "l"  # LATIN SMALL LETTER L WITH BAR
    rewrite_hash[u'\u019B'] = " "  # LATIN SMALL LETTER LAMBDA WITH STROKE
    rewrite_hash[u'\u019C'] = " "  # LATIN CAPITAL LETTER TURNED M
    rewrite_hash[u'\u019D'] = "N"  # LATIN CAPITAL LETTER N WITH LEFT HOOK
    rewrite_hash[u'\u019E'] = "n"  # LATIN SMALL LETTER N WITH LONG RIGHT LEG
    rewrite_hash[u'\u019F'] = "O"  # LATIN CAPITAL LETTER O WITH MIDDLE TILDE

    rewrite_hash[u'\u0226'] = "a"  # LATIN CAPITAL LETTER A WITH DOT ABOVE
    rewrite_hash[u'\u0227'] = "a"  # LATIN SMALL LETTER A WITH DOT ABOVE
    rewrite_hash[u'\u02DC'] = " "  # SMALL TILDE

    rewrite_hash[u'\u0336'] = " "  # COMBINING LONG STROKE OVERLAY
    rewrite_hash[u'\u0391'] = "A"  # GREEK CAPITAL LETTER ALPHA
    rewrite_hash[u'\u03A4'] = "T"  # GREEK CAPITAL LETTER TAU
    rewrite_hash[u'\u03A9'] = " omega "  # GREEK CAPITAL LETTER OMEGA
    rewrite_hash[u'\u03B2'] = " beta "  # GREEK SMALL LETTER BETA
    rewrite_hash[u'\u03BC'] = " mu "  # GREEK SMALL LETTER MU
    rewrite_hash[u'\u03C0'] = " pi "  # GREEK SMALL LETTER PI

    rewrite_hash[u'\u0441'] = "c"  # CYRILLIC SMALL LETTER ES

    rewrite_hash[u'\u1F7B'] = "u"  # GREEK SMALL LETTER UPSILON WITH OXIA
    rewrite_hash[u'\u1E25'] = "h"  # LATIN SMALL LETTER H WITH DOT BELOW
    rewrite_hash[u'\u1ECB'] = "i"  # LATIN SMALL LETTER I WITH DOT BELOW

    rewrite_hash[u'\u2000'] = " "  # EN QUAD
    rewrite_hash[u'\u2001'] = " "  # EM QUAD
    rewrite_hash[u'\u2009'] = " "  # THIN SPACE
    rewrite_hash[u'\u200A'] = " "  # HAIR SPACE
    rewrite_hash[u'\u200B'] = " "  # ZERO WIDTH SPACE

    rewrite_hash[u'\u200E'] = " "  # LEFT-TO-RIGHT MARK
    rewrite_hash[u'\u200F'] = " "  # RIGHT-TO-LEFT MARK

    rewrite_hash[u'\u2010'] = "-"  # HYPHEN
    rewrite_hash[u'\u2011'] = "-"  # NON-BREAKING HYPHEN
    rewrite_hash[u'\u2013'] = " "  # EN DASH
    rewrite_hash[u'\u2014'] = " "  # EM DASH
    rewrite_hash[u'\u2015'] = " "  # HORIZONTAL BAR
    rewrite_hash[u'\u2018'] = "'"  # LEFT SINGLE QUOTATION MARK
    rewrite_hash[u'\u2019'] = "'"  # RIGHT SINGLE QUOTATION MARK
    rewrite_hash[u'\u201A'] = " "  # SINGLE LOW-9 QUOTATION MARK
    rewrite_hash[u'\u201C'] = " "  # LEFT DOUBLE QUOTATION MARK
    rewrite_hash[u'\u201D'] = " "  # RIGHT DOUBLE QUOTATION MARK
    rewrite_hash[u'\u201E'] = " "  # DOUBLE LOW-9 QUOTATION MARK
    rewrite_hash[u'\u201F'] = " "  # OUBLE HIGH-REVERSED-9 QUOTATION MARK

    rewrite_hash[u'\u2020'] = " "  # DAGGER
    rewrite_hash[u'\u2021'] = " "  # DOUBLE DAGGER
    rewrite_hash[u'\u2022'] = " "  # BULLET
    rewrite_hash[u'\u2023'] = " "  # TRIANGULAR BULLET
    rewrite_hash[u'\u2024'] = " "  # ONE DOT LEADER
    rewrite_hash[u'\u2025'] = " "  # TWO DOT LEADER
    rewrite_hash[u'\u2026'] = " "  # HORIZONTAL ELLIPSIS
    rewrite_hash[u'\u2027'] = " "  # HYPHENATION POINT
    rewrite_hash[u'\u2028'] = " "  # LINE SEPARATOR
    rewrite_hash[u'\u2029'] = "\n"  # PARAGRAPH SEPARATOR
    rewrite_hash[u'\u202A'] = " "  # LEFT-TO-RIGHT EMBEDDING (???)
    rewrite_hash[u'\u202B'] = " "  # RIGHT-TO-LEFT EMBEDDING (???)
    rewrite_hash[u'\u202C'] = " "  # POP DIRECTIONAL FORMATTING (???)
    rewrite_hash[u'\u202D'] = " "  # LEFT-TO-RIGHT OVERRIDE
    rewrite_hash[u'\u202E'] = " "  # RIGHT-TO-LEFT OVERRIDE
    rewrite_hash[u'\u202F'] = " "  # NARROW NO-BREAK SPACE

    rewrite_hash[u'\u2032'] = "\'"  # PRIME
    rewrite_hash[u'\u2033'] = " "  # DOUBLE PRIME
    rewrite_hash[u'\u203B'] = " "  # REFERENCE MARK

    rewrite_hash[u'\u206B'] = " "  # ACTIVATE SYMMETRIC SWAPPING
    rewrite_hash[u'\u206E'] = " "  # NATIONAL DIGIT SHAPES
    rewrite_hash[u'\u206F'] = " "  # NOMINAL DIGIT SHAPES

    rewrite_hash[u'\u20AC'] = " euros "  # EURO SIGN

    rewrite_hash[u'\u2116'] = " "  # NUMERO SIGN
    rewrite_hash[u'\u2154'] = "2/3"  # VULGAR FRACTION TWO THIRDS
    rewrite_hash[u'\u2192'] = " "  # RIGHTWARDS ARROW
    rewrite_hash[u'\u21FC'] = " "  # LEFT RIGHT ARROW WITH DOUBLE VERTICAL STROKE
    rewrite_hash[u'\u2122'] = " "  # TRADE MARK SIGN

    rewrite_hash[u'\u2212'] = "-"  # MINUS SIGN

    rewrite_hash[u'\u23AF'] = " "  # HORIZONTAL LINE EXTENSION
    rewrite_hash[u'\u25BA'] = " "  # BLACK RIGHT-POINTING POINTER
    rewrite_hash[u'\u2665'] = " "  # BLACK HEART SUIT

    rewrite_hash[u'\uFB01'] = "fi"  # LATIN SMALL LIGATURE FI
    rewrite_hash[u'\uFF00'] = " "  #
    return rewrite_hash

def parse_breitbart_news(soup):
    """
    Give a soup object, parse the article text from HTML.

    This is verified to work for Breitbart News, but not other websites. You need to create a function like this for
    each website.

    :param soup: HTML code as BeautifulSoup object
    :return article_txt: Text from article
    """
    # Parse soup - for lines that are bs4.element.NavigableString/article text
    q = list(soup.children)[2]
    q3 = list(q.children)[3]
    w = q3.find_all('p')
    # Ignogore line if it starts with any of these strings
    nonos = (u'<p><a href', u'<p class="', u'<p id="', u'<p dir', u'<p>advertisement</p>',u'<p><br/>Comment count on this article reflects comments made on Breitbart.com and Facebook. Visit <a href="https://www.facebook.com/Breitbart" target="_blank">Breitbart\'s Facebook Page</a>.</p>')
    # Remove these strings from beginning of "good" lines
    # badstarts = [u'<span style="font-weight: 400;">', u'<p><span>', u'<p>']
    # Replancement dictionary
    rewrite_hash = create_utf8_rewrite_hash()
    # rewrite_hash[u'</p>'] = ' '
    # rewrite_hash[u'<em>'] = ''
    # rewrite_hash[u'</em>'] = ''
    # rewrite_hash[u'<strong>'] = ''
    # rewrite_hash[u'</strong>'] = ''
    # rewrite_hash[u'<span style="font-weight: 400;">'] = ''
    # Parse article txt
    article_txt = u''
    for ii in w:
        print ii
        txt = True
        ii = unicode(ii)
        if ii.startswith(nonos):
            txt = False
        if txt:
            soup2 = BeautifulSoup(ii, 'lxml')
            print soup2.get_text()
            txt = soup2.get_text()
            for kk in rewrite_hash.keys():
                txt.replace(kk, unicode(rewrite_hash[kk]))
            # # Format good lines...
            # for bb in badstarts:
            #     if ii.startswith(bb):
            #         ii = ii[len(bb):]
            #         break
            # # Replace strings with others
            # for kk in rewrite_hash.keys():
            #     ii.replace(kk, unicode(rewrite_hash[kk]))
            # article_txt += ii
            article_txt += txt
    return article_txt


def main():
    ##
    # Set Parameters
    main_dir = '/Users/AR_Magnavox/Documents/RUDSSP FILES'        # Main dir
    save_dir = os.path.join(main_dir,'sample_data' )                # Where you save clean data
    key_param_fn = os.path.join(main_dir,'api_keys.json')           # File of API keys

    # Load API keys
    with open(key_param_fn,'r') as f:
        api_keys = json.load(f)
    newsapi_key = api_keys['newsapi']                               # newsapi.org API key

    # Fake browser header
    hdr  = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

    ##
    # Look up all sources of newsapi.org
    # This is where you can obtain all request options for each publication
    # We make the project easier, we'll only focus on English language documents
    source_req = ' https://newsapi.org/v1/sources?language=en'
    data_str = urllib2.urlopen(source_req).read()
    source_data = json.loads(data_str)
    source_ids = [i['id'] for i in source_data['sources']]      # all IDs

    # Collect data from breitbart
    id = source_ids[7]

    ##
    # Check URL
    # 1. See if it falls into other categories
    # 2. Look at the content, does it seem labeled correctly? If not, please message me so I can fix it.
    # 3. Add note to ourdropbox/url_labels/readme.txt that you are scraping the source. Also state what labels it has
    # Load clean url list
    url_source_dir = '/Users/AR_Magnavox/Documents/RUDSSP FILES'     # local dropbox dir
    clean_urls_fn = os.path.join(url_source_dir, 'all_clean_urls.json')         # clean urls json
    with open(clean_urls_fn, 'r') as f:
        clean_urls = json.load(f)
    # Check site labels:
    for ii in clean_urls.keys():
        for jj in clean_urls[ii]:
            if 'breitbart' in jj.lower():
                print '%s: %s' % (ii, jj)

    ##
    # Get JSONs
    clean_data = []
    for sortBy in source_data[u'sources'][7]['sortBysAvailable']:
        # Make request
        req = 'https://newsapi.org/v1/articles?source=' + id + '&sortBy=' + sortBy + '&apiKey=' + newsapi_key
        q = urllib2.urlopen(req).read()
        data = json.loads(q)
        for ii in data['articles']:
            # Did we already collect URL?
            repeat=False
            for jj in clean_data:
                if ii[u'url'] == jj[u'url']:
                    print u"Already collected URL " + jj[u'url']
                    repeat = True
                    break
            if repeat:
                continue
            # Download article html
            main = ii[u'url']
            req = urllib2.Request(main, headers=hdr)
            res = urllib2.urlopen(req)
            webContent = res.read()
            # Parse out article text
            soup = BeautifulSoup(webContent, 'html.parser')  # Convert it to soup object
            print "\n new try:"
            article_txt = parse_breitbart_news(soup)
            print '\nresults:'
            print article_txt
            # append data
            ii[u'publisher'] = id
            ii[u'article'] = article_txt
            now = datetime.datetime.utcnow()  # Use UTC time zone ONLY
            ii[u'dateOfCollection'] = unicode(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
            clean_data.append(ii)


    # req = 'https://newsapi.org/v1/articles?source=' + id + '&sortBy=top&apiKey=' + newsapi_key
    # q = urllib2.urlopen(req).read()
    # data = json.loads(q)
    #
    # # Get articles
    # clean_data = []
    # for ii in data['articles']:
    #     # Download article html
    #     main = ii[u'url']
    #     req = urllib2.Request(main, headers=hdr)
    #     res = urllib2.urlopen(req)
    #     webContent = res.read()
    #     # Parse out article text
    #     soup = BeautifulSoup(webContent, 'html.parser')     # Convert it to soup object
    #     print "\n new try:"
    #     article_txt = parse_breitbart_news(soup)
    #     print '\nresults:'
    #     print article_txt
    #     # append data
    #     ii[u'publisher'] = id
    #     ii[u'article'] = article_txt
    #     now = datetime.datetime.utcnow()                    # Use UTC time zone ONLY
    #     ii[u'dateOfCollection'] = unicode(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    #     clean_data.append(ii)

    # Save clean data to json file
    uid = str(uuid.uuid4())         # Add this to filename to avoid overwritting files with the same name
    clean_fn = os.path.join(save_dir, 'source_json_' + uid + '.json')
    with open(clean_fn, "w") as json_file:
        json.dump(clean_data, json_file)


if __name__ == '__main__':
    main()