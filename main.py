from collections import defaultdict

import nltk
from matplotlib import pyplot
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

t1 = open("test.txt", encoding="utf8" )

text = []
words = []
lemmatizer = WordNetLemmatizer()

for line in t1:
    text.append(line.strip())

stop_words = set(stopwords.words('english'))
#
# for sentence in text:
#     print(sentence)

for sentence in text:
    sentence = sentence.casefold()
    punctuationfree = "".join([i for i in sentence if i not in string.punctuation])
    word_tokens = word_tokenize(punctuationfree)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = nltk.pos_tag(filtered_sentence)
    print(filtered_sentence)
    # words += [x for x in filtered_sentence]

# print(words)

# mp = defaultdict(int)
#
# corpus = []
# ps = PorterStemmer()
#
# for sentence in text:
#     for line in sentence:
#         test_str = line.translate(str.maketrans('', '', string.punctuation))
#         test_str = test_str.casefold()
#         ps.stem(test_str)
#         corpus.append(test_str)
#         mp[test_str]+=1
#
# # pyplot.plot(corpus)
# # pyplot.show()
# # print(mp)
#
# temp = []
#
# for x in mp:
#     # print(x, mp[x])
#     temp.append((mp[x], x))
#
# temp.sort(reverse=True)
# print(temp)
# print(mp['that'])