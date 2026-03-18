#importing libraries
import spacy
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#loading spaCy model
nlp=spacy.load("en_core_web_sm")

text=open("book_pca.txt", encoding="utf8").read()
doc=nlp(text)

#sentence extraction
sentences=[sent.text.strip() for sent in doc.sents]

#filtering sentences (too short/long)
sentences=[s for s in sentences if 5 < len(s.split()) < 40]

#randomizing + selecting 100 sentences
random.shuffle(sentences)
sentences=sentences[:100]

print("Total sentences used:", len(sentences))

###SENTENCE ANALYSIS###
data=[]
for sent in sentences:
    doc=nlp(sent)
    length=len(doc)
    
    nouns=sum(1 for token in doc if token.pos_=="NOUN")
    verbs=sum(1 for token in doc if token.pos_=="VERB")
    adjectives=sum(1 for token in doc if token.pos_=="ADJ")
    adverbs=sum(1 for token in doc if token.pos_=="ADV")
