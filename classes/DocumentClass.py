import spacy
try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Document:
    def __init__(self, doctext, nlp, tfdict=None, tftable=None, ndoc=None, lang="spanish"):
        sentences = re.split('\.|\n', doctext)
        pos = 0
        self.sentenceTable = np.zeros((len(sentences), 2))
        self.sentenceVectors = np.zeros((len(sentences), len(nlp("a").vector)))
        self.text = []
        emptysent = []
        for n, sent in enumerate(sentences):
            sentpos = doctext.find(sent, pos)
            posi = sentpos
            posf = sentpos + len(sent)
            cleansent = re.sub(r'[^\w\s]',' ',sent)
            words = cleansent.split()
            words = [word.lower() for word in words if word not in stopwords.words(lang)]
            if len(words) > 0 and any([len(word)>1 for word in words]):
                self.text.append(cleansent)
                sentweights = []
                try:
                    sentvecs = np.vstack([nlp(word).vector for word in words])
                    for w in words:
                        try:
                            we = tftable[(ndoc, tfdict[w])]
                        except:
                            we = 0
                        sentweights.append(we)
                    sentweights = np.asarray(sentweights)#sentweights = np.asarray([tftable[(ndoc, tfdict[word])] for word in words])
                    sentweights = np.multiply(1/(sum(sentweights)), sentweights)
                    sentvecs = np.vstack([np.multiply(vector, weight) for vector, weight in zip(sentvecs, sentweights)])
                    sentencevector = np.sum(sentvecs, axis=0)
                    self.sentenceTable[n] = (posi, posf)
                    self.sentenceVectors[n] = sentencevector

                #TODO: This is for debugging purposes. Change to logging
                except Exception as e:
                    logger.exception("message")
                    print(e)
            else:
                emptysent.append(n)
        self.sentenceVectors = np.delete(self.sentenceVectors,emptysent,axis=0)
        self.sentenceTable = np.delete(self.sentenceTable,emptysent,axis=0)