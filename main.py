from collections import defaultdict
from matplotlib import pyplot
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

t1 = open("test.txt", encoding="utf8" )

text = []
words = []

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
    print(filtered_sentence)


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