import operator
import nltk
from matplotlib import pyplot as plt
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter, defaultdict
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from pattern.text.en import singularize
import inflect

#read book
t1 = open("book1.txt", encoding="utf8")

text = []
words = []
final = []
lemmatizer = WordNetLemmatizer()
custom = ["chapter", "page", ]
p = inflect.engine()

#helper function
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#read sentences
for line in t1:
    text.append(line.strip())

for sentence in text:
    punctuationfree = "".join([i for i in sentence if i not in string.punctuation and not i.isdigit()])  #remove punctuation
    word_tokens = word_tokenize(punctuationfree)   #tokenize words
    filtered_sentence = nltk.pos_tag(word_tokens)  #part of speech tagging for words based on treebank tagset
    filtered_sentence = [(singularize(w.casefold()), t) for w, t in filtered_sentence if not w.lower() in STOPWORDS and len(w)>1 and w.lower() not in custom] #remove stopwords
    words.append(filtered_sentence)

for sentence in words:
    temp = []
    for word, tag in sentence:
        temp.append((lemmatizer.lemmatize(word, get_wordnet_pos(tag)), tag))   #lemmatize using pos tag
    final.append(temp)


#word frequency plots
to_plot = []
to_plot_tag = []
for sentence in words:
    for word, tag in sentence:
        to_plot.append(word)
        to_plot_tag.append(tag)
counted = Counter(to_plot)
counted_tag = Counter(to_plot_tag)
word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
tag_freq = pd.DataFrame(counted_tag.items(), columns=['tag', 'frequency']).sort_values(by='frequency', ascending=False)
mp = defaultdict(int)
for word in word_freq['word']:
    mp[len(word)] += 1
mp = dict( sorted(mp.items(), key=operator.itemgetter(1),reverse=True))
keys = list(mp.keys())
vals = [mp[k] for k in keys]

keys = keys[:15]
vals = vals[:15]
sns.barplot(x=keys, y=vals)
plt.title("Word length vs frequency")
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.show()

word_freq = word_freq.head(25)
plt.title("Word frequency")
sns.barplot(x='frequency', y='word', data=word_freq)
plt.show()

tag_freq = tag_freq.head(10)
plt.title("Tag Frequency (Treebank Tags)")
sns.barplot(x='frequency', y='tag', data=tag_freq)
plt.show()

#wordcloud
temp = []
for word in word_freq['word']:
    temp.append(word)
temp2 = " ".join(temp)+" "

cloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(temp2)
plt.figure(figsize=(8,8), facecolor=None)
plt.imshow(cloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


