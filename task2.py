#importing required libraries
import spacy
from spacy import displacy
from spacy import tokenizer
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example

#evaluation function of the given examples
def evaluate(examples):
    scorer = Scorer()
    example = []
    nlp = spacy.load('en_core_web_sm')  # for spaCy's pretrained use 'en_core_web_sm'
    for input_, annot in examples:
        pred = nlp.make_doc(input_)
        # print(pred)
        temp = Example.from_dict(pred, annot)
        # temp.predicted = nlp(str(example.predicted))
        example.append(temp)
    scores = nlp.evaluate(example)
    return scores

#here we input the manually labelled data for evaluation against spaCy english NER model
examples = [('In the early 1970s, packet-switched networks began to proliferate, with the', {'entities': [(3, 18, 'DATE')]}),('ARPAnet—the precursor of the Internet—being just one of many networks. Each of',{'entities': [(0, 7, 'ORG')]}), ('these networks had its own protocol. Two researchers, Vinton Cerf and Robert Kahn,', {'entities': [(37, 40, 'CARDINAL'), (54, 65, 'PERSON'), (70, 81, 'PERSON')]}), ('network protocol called TCP/IP, which stands for Transmission Control', {'entities': [(24, 30, 'PRODUCT'), (49, 69, 'PRODUCT')]}), ('Protocol/Internet Protocol. Although Cerf and Kahn began by seeing the protocol as', {'entities': [(0, 26, 'LAW'), (46, 50, 'PERSON')]}), ('a single entity, it was later split into its two parts, TCP and IP, which operated sepa-', {'entities': [(45, 48, 'CARDINAL'), (56, 59, 'ORG'), (64, 66, 'ORG')]}), ('rately. Cerf and Kahn published a paper on TCP/IP in May 1974 in IEEE', {'entities': [(17, 21, 'PERSON'), (43, 49, 'ORG'), (53, 61, 'DATE'), (65, 69, 'ORG')]}), ('Transactions on Communications Technology [Cerf 1974].', {'entities': [(0, 41, 'ORG'), (48, 52, 'DATE')]}), ('     The TCP/IP protocol, which is the bread and butter of today’s Internet, was devised', {'entities': [(13, 15, 'ORG'), (59, 64, 'DATE')]}), ('before PCs, workstations, smartphones, and tablets, before the proliferation of Ethernet,', {'entities': [(80, 88, 'PRODUCT')]}), ('cable, and DSL, WiFi, and other access network technologies, and before the Web,', {'entities': [(11, 14, 'PRODUCT'), (16, 20, 'PRODUCT')]}), ('social media, and streaming video. Cerf and Kahn saw the need for a networking pro-', {'entities': [(44, 48, 'PERSON')]}), ('     In 2004, Cerf and Kahn received the ACM’s Turing Award, considered the', {'entities': [(8, 12, 'DATE'),(14, 18, 'PERSON'), (23, 27, 'PERSON'), (41, 44, 'ORG'), (46, 58, 'ORG')]}), ('“Nobel Prize of Computing” for “pioneering work on internetworking, including the', {'entities': [(1, 25, 'WORK_OF_ART')]}), ('design and implementation of the Internet’s basic communications protocols, TCP/IP,', {'entities': [(76, 82, 'ORG')]})]

#loading the model
nlp = spacy.load('en_core_web_sm')
t1 = open("passage.txt", encoding="utf8") #read data from file
text=""
ex = []
for line in t1:
    text = line
    doc = nlp(text)
    # ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    temp = (line[:len(line)-1], {'entities':[(e.start_char, e.end_char, e.label_) for e in doc.ents]}) #extracting entites from the line
    if len(doc.ents)>0:
        # print(ents)
        ex.append(temp)
        # print(ex)

results = evaluate(examples)    #evaluating the model
print(ex)
print("Precision {:0.4f}\tRecall {:0.4f}\tF-score {:0.4f}".format(results['ents_p'], results['ents_r'], results['ents_f']))
