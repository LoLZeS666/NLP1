import nltk
from matplotlib import pyplot as plt
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS

#read book
t1 = open("test.txt", encoding="utf8")

text = []
words = []
final = []
lemmatizer = WordNetLemmatizer()

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

#set stopwords from nltk corpus
stop_words = set(stopwords.words('english'))

for sentence in text:
    punctuationfree = "".join([i for i in sentence if i not in string.punctuation])  #remove punctuation
    word_tokens = word_tokenize(punctuationfree)   #tokenize words
    filtered_sentence = nltk.pos_tag(word_tokens)  #part of speech tagging for words based on treebank tagset
    filtered_sentence = [(w.casefold(), t) for w, t in filtered_sentence if not w.lower() in STOPWORDS] #remove stopwords
    words.append(filtered_sentence)

for sentence in words:
    temp = []
    for word, tag in sentence:
        temp.append((lemmatizer.lemmatize(word, get_wordnet_pos(tag)), tag))   #lemmatize using pos tag
    final.append(temp)



#word frequency plots
to_plot = []
for sentence in words:
    for word, tag in sentence:
        to_plot.append(word)
counted = Counter(to_plot)
word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
word_freq = word_freq.head(30)
sns.barplot(x='frequency', y='word', data=word_freq)
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