#import libraries
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#load spaCy model
nlp=spacy.load("en_core_web_sm")
