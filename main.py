import nltk
from matplotlib import pyplot
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

t1 = open("test.txt", encoding="utf8" )

text = []
words = []
final = []
lemmatizer = WordNetLemmatizer()
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

for line in t1:
    text.append(line.strip())

stop_words = set(stopwords.words('english'))
#
# for sentence in text:
#     print(sentence)

for sentence in text:
    punctuationfree = "".join([i for i in sentence if i not in string.punctuation])
    word_tokens = word_tokenize(punctuationfree)
    filtered_sentence = nltk.pos_tag(word_tokens)
    filtered_sentence = [(w.casefold(), t) for w, t in filtered_sentence if not w in stop_words]
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # filtered_sentence = nltk.pos_tag(filtered_sentence)
    # print(filtered_sentence)
    words.append(filtered_sentence)

for sentence in words:
    temp = []
    for word, tag in sentence:
        temp.append((lemmatizer.lemmatize(word, get_wordnet_pos(tag)), tag))
    final.append(temp)
print(final)