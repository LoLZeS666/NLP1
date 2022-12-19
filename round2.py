#importing required libraries
import string
from collections import defaultdict, Counter

import matplotlib.pyplot as plt     #allows us to plot graphs in python
import pandas as pd            #allows easy manipulation of large data
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from pattern.text.en import singularize
from wordcloud import STOPWORDS
import seaborn as sns          #used for visualizing

nouns, verbs = [], []
t1 = open("book1.txt", encoding="utf8")     #reading from file
text = []
words = []
final = []
custom = ["chapter", "page", ]
for line in t1:
    text.append(line.strip())

#filtering stopwords and punctuation marks from the read file
for sentence in text:
    punctuationfree = "".join([i for i in sentence if i not in string.punctuation and not i.isdigit()])  #remove punctuation
    word_tokens = word_tokenize(punctuationfree)   #tokenize words
    filtered_sentence = pos_tag(word_tokens)
    filtered_sentence = [(w.casefold(), t.casefold()) for w, t in filtered_sentence if not w.lower() in STOPWORDS and len(w)>1 and w.lower() not in custom] #remove stopwords
    words.append(filtered_sentence)
print(len(words))

#assigning words to their respective categories in the WordNET database
#instead of counting all senses of a given word, we have matched the database with the pos tag and taken the first
#sense of the word in that database
for sentence in words:
    for word, tag in sentence:
        for synset in wn.synsets(word):
            if synset.pos() == "n" and tag[0]=='n':
                nouns.append(synset.lexname()[5:])
                break
            elif synset.pos() == "v" and tag[0]=='v':
                verbs.append(synset.lexname()[5:])
                break

#counting all the tags for plotting
count_nouns = Counter(nouns)
count_verbs = Counter(verbs)
noun_freq = pd.DataFrame(count_nouns.items(), columns=['category', 'frequency']).sort_values(by='frequency', ascending=False)
verb_freq = pd.DataFrame(count_verbs.items(), columns=['category', 'frequency']).sort_values(by='frequency', ascending=False)

#using seaborn to plot bargraphs for the data
fig, ax = plt.subplots(figsize=(8, 6))
plt.title("Noun category frequency")
sns.barplot(y="category", x="frequency", data=noun_freq, ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
plt.title("Verb category frequency")
sns.barplot(y="category", x="frequency", data=verb_freq, ax=ax)
plt.show()