import spacy

#small English model
nlp = spacy.load("en_core_web_sm")

# Process the text
doc = nlp("In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context.")

for token in doc:
    print(f"Token: {token.text}")
    print(f"POS: {token.pos_} - {spacy.explain(token.pos_)}")
    print(f"Tag: {token.tag_} - {spacy.explain(token.tag_)}")
    print() 


'''Output:
$ python posTag.py

Token: In
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition

Token: corpus
POS: X - other
Tag: FW - foreign word

Token: linguistics
POS: NOUN - noun
Tag: NNS - noun, plural

Token: ,
POS: PUNCT - punctuation
Tag: , - punctuation mark, comma

Token: part
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: -
POS: PUNCT - punctuation
Tag: HYPH - punctuation mark, hyphen

Token: of
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition
Token: -
POS: PUNCT - punctuation
Tag: HYPH - punctuation mark, hyphen

Token: speech
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: tagging
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: (
POS: PUNCT - punctuation
Tag: -LRB- - left round bracket

Token: POS
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: tagging
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: or
POS: CCONJ - coordinating conjunction
Tag: CC - conjunction, coordinating

Token: PoS
POS: ADJ - adjective
Tag: JJ - adjective (English), other noun-modifier 
(Chinese)

Token: tagging
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: or
POS: CCONJ - coordinating conjunction
Tag: CC - conjunction, coordinating

Token: POST
POS: PROPN - proper noun
Tag: NNP - noun, proper singular

Token: )
POS: PUNCT - punctuation
Tag: -RRB- - right round bracket

Token: ,
POS: PUNCT - punctuation
Tag: , - punctuation mark, comma

Token: also
POS: ADV - adverb
Tag: RB - adverb

Token: called
POS: VERB - verb
Tag: VBD - verb, past tense

Token: grammatical
POS: ADJ - adjective
Tag: JJ - adjective (English), other noun-modifier 
(Chinese)

Token: tagging
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: is
POS: AUX - auxiliary
Tag: VBZ - verb, 3rd person singular present       

Token: the
POS: DET - determiner
Tag: DT - determiner

Token: process
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: of
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition
Token: marking
POS: VERB - verb
Tag: VBG - verb, gerund or present participle

Token: up
POS: ADP - adposition
Tag: RP - adverb, particle

Token: a
POS: DET - determiner
Tag: DT - determiner

Token: word
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: in
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition

Token: a
POS: DET - determiner
Tag: DT - determiner

Token: text
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: (
POS: PUNCT - punctuation
Tag: -LRB- - left round bracket

Token: corpus
POS: X - other
Tag: FW - foreign word

Token: )
POS: PUNCT - punctuation
Tag: -RRB- - right round bracket

Token: as
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition
Token: corresponding
POS: VERB - verb
Tag: VBG - verb, gerund or present participle      

Token: to
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition

Token: a
POS: DET - determiner
Tag: DT - determiner

Token: particular
POS: ADJ - adjective
Tag: JJ - adjective (English), other noun-modifier 
(Chinese)

Token: part
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: of
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition

Token: speech
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: ,
POS: PUNCT - punctuation
Tag: , - punctuation mark, comma

Token: based
POS: VERB - verb
Tag: VBN - verb, past participle

Token: on
POS: ADP - adposition
Tag: IN - conjunction, subordinating or preposition
Token: both
POS: CCONJ - coordinating conjunction
Tag: CC - conjunction, coordinating

Token: its
POS: PRON - pronoun
Tag: PRP$ - pronoun, possessive

Token: definition
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: and
POS: CCONJ - coordinating conjunction
Tag: CC - conjunction, coordinating

Token: its
POS: PRON - pronoun
Tag: PRP$ - pronoun, possessive

Token: context
POS: NOUN - noun
Tag: NN - noun, singular or mass

Token: .
POS: PUNCT - punctuation
Tag: . - punctuation mark, sentence closer
'''