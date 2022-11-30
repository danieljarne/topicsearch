import spacy
import numpy as np
import scipy.spatial.distance as vdist
import math
from classes.DocumentClass import Document
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
#from matplotlib import pyplot
import pickle as pkl
from py2neo import Node, Relationship, Subgraph, Graph
import random
import string

logger = logging.getLogger(__name__)

class GraphEngine:

    def __init__(self, mpath, docs=None, impor=False):
        # For now, docs is a folder full of text files.
        self.modpath = mpath
        self.docs = None
        self.docfolder = None
        self.processes = {}
        self.docNumbering = {}
        # This loads spacy model. TAKES LONG TIME. Import option allows to manipulate docs
        self.nlp = spacy.load(mpath)

    @staticmethod
    def _tfidf(docdata):
        docs = []
        for key in docdata.keys():
            docs.append(docdata[key])
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(docs)
        return vectorizer.vocabulary_, x

    @staticmethod
    def cosine_similarity(v1, v2):
        # return 1- vdist.cosine(v1, v2)
        xx, xy, yy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            xx += x * x
            yy += y * y
            xy += x * y
        return xy / math.sqrt(xx * yy) if math.sqrt(xx * yy) > 0 else 0

    def doc_pipeline(self,docpath):
        letters = string.ascii_lowercase
        procid = ''.join(random.choice(letters) for i in range(5))
        try:
            self.processes[procid] = {"docfolder":docpath, "docs":[], "graphid":""}
            self.docfolder = docpath
            docdata = self._process_text(docpath)
            self._build_documents(docdata)
            self._intradoc_sim()
            self.build_graph(procid)
            self.add_similarities_to_graph(procid)
            self.processes[procid]["docs"] = self.docs.keys()
            logger.log(level=1, msg="Subgraph added:"+str(self.processes[procid]))
        except Exception as e:
            logger.exception("Something went wrong (PID="+str(procid)+")" + str(e))
        return self.processes[procid]

    def return_ids(self):
        return self.processes.keys()


    def export_docVec_pkl(self):
        with open(os.path.join(".","pickle","docDB.pkl"),'wb') as file:
            pkl.dump(self.docs,file)
        if self.intradocSim is not None:
            with open(os.path.join(".", "pickle", "docSim.pkl"), 'wb') as file:
                pkl.dump(self.intradocSim, file)

    def import_docVec_pkl(self):
        try:
            with open(os.path.join(".","pickle","docDB.pkl"), 'rb') as file:
                self.docs = pkl.load(file)
        except Exception as e:
            logger.exception("File not found:" + str(e))
        try:
            # if self.intradocSim is not None:
            with open(os.path.join(".", "pickle", "docSim.pkl"), 'rb') as file:
                self.intradocSim = pkl.load(file)
        except Exception as e:
            logger.exception("File not found:" + str(e))

    def _process_text(self, docpath, docdata=None, process_all=True):
        if docdata is None:
            docdata = {}
        if self.nlp is None:
            self.nlp = spacy.load(self.modpath)
        if self.docs is not None and not process_all:
            docs_processed = self.docs.keys()
        else:
            docs_processed = []

        for item in os.listdir(docpath):
            if ".txt" in item and item not in docs_processed:
                ilast = len(self.docNumbering.keys())
                with open(os.path.join(docpath, item), "r", encoding='utf-8') as text:
                    doctext = text.read()
                    docdata[item] = doctext
                    self.docNumbering[item] = ilast+1
        self.tfidfdict, self.tfidftable = self._tfidf(docdata)
        return docdata

    def _build_documents(self, docdata):
        self.docs = {}
        for i, key in enumerate(docdata.keys()):
            text = docdata[key]
            self.docs[key] = Document(text, nlp=self.nlp, tfdict=self.tfidfdict, tftable=self.tfidftable, ndoc=i)

    def _intradoc_sim(self):
        self.intradocSim = {}
        self.doc2docSim = {}
        for k, key1 in enumerate(self.docs.keys()):
            self.intradocSim[k] = {}
            self.doc2docSim[key1] = {}
            for l, key2 in enumerate(self.docs.keys()):
                doc1 = self.docs[key1]
                doc2 = self.docs[key2]
                sentenceSim = np.zeros([len(doc1.sentenceVectors), len(doc2.sentenceVectors)])
                for i, sentence1 in enumerate(doc1.sentenceVectors):
                    for j, sentence2 in enumerate(doc2.sentenceVectors):
                        sentenceSim[i, j] = self.cosine_similarity(sentence1, sentence2)
                self.intradocSim[k][l] = sentenceSim
                self.doc2docSim[key1][key2] = sentenceSim.sum()/np.size(sentenceSim)


    def build_graph(self,id, host="localhost",password="test"):
        g = None # g is the basic subgraph of docs + sentences
        for k, key in enumerate(self.docs.keys()):
            docName = key
            d = Node("Document", name=docName, n_sentences=len(self.docs[key].sentenceTable),pid=id)
            if g is None:
                g = d
            else:
                g = g | d
            for i,s in enumerate(self.docs[key].sentenceVectors):
                sname = docName + "||s_"+str(i)
                ns = Node("Sentence", name=sname, text=self.docs[key].text[i],pid=id)#vector=s.tostring(),
                dns = Relationship(d, "CONTAINS", ns)
                g = g | dns
        graph1 = Graph(host=host, password=password)
        graph1.create(g)


    def add_similarities_to_graph(self, id, host="localhost",password="test", threshold=0.6):
        graph1 = Graph(host=host, password=password)
        nodesgraph = graph1.nodes.match(pid=id)
        sentencenodes = graph1.nodes.match("Sentence", pid=id)
        for nod1 in sentencenodes:
            nodename1 = nod1["name"]
            nodename1 = nodename1.split("||")
            docnumber1 = self.docNumbering[nodename1[0]]
            sentencenumber1 = nodename1[1].split("_")[1]
            for nod2 in sentencenodes:
                nodename2 = nod2["name"]
                nodename2 = nodename2.split("||")
                docnumber2 = self.docNumbering[nodename2[0]]
                sentencenumber2 = nodename2[1].split("_")[1]
                if docnumber1 != docnumber2 and self.intradocSim[docnumber1-1][docnumber2-1][int(sentencenumber1),int(sentencenumber2)] > threshold:
                    r = Relationship(nod1,"SENTENCE_SIMILAR_TO_SENTENCE",nod2)
                    graph1.create(r)
        docnodes = graph1.nodes.match("Document", pid=id)
        for doc1 in docnodes:
            n1 = doc1["name"]
            for doc2 in docnodes:
                n2 = doc2["name"]
                if doc1 != doc2 and self.doc2docSim[n1][n2]> 0.5:
                    r = Relationship(doc1,"DOC_VERY_SIMILAR_TO_DOC",doc2)
                    graph1.create(r)
                elif doc1 != doc2 and self.doc2docSim[n1][n2]> 0.45:
                    r = Relationship(doc1,"DOC_SIMILAR_TO_DOC",doc2)
                    graph1.create(r)

    def query(self, q):
        # Query the similarity between the word q and the documents
        try:
            assert isinstance(q, str)
        except Exception as e:
            logger.error("Query is not a valid string")
        q = q.lower()
        words = q.split()
        results = {}
        qvec = np.mean(np.vstack([self.nlp(word).vector for word in words]), axis=0)
        for key in self.docs.keys():
            sim = [self.cosine_similarity(sentvec, qvec) for sentvec in self.docs[key].sentenceVectors]
            sentences = [i for i,j in enumerate(self.docs[key].sentenceVectors)]
            # Plotting of similarity per sentence in doc
            # pyplot.scatter(sentences,sim)
            # pyplot.title("Similarity of "+q+" "+key)
            # pyplot.show()
            # check similarity with top 10% sentences
            sim.sort(reverse=True)
            top10 = max(int(0.1*len(sim)),1)
            mean = np.mean(sim[0:top10])
            results[key] = {"10_Sim": mean}
        return results

    def scan_new(self, folder=None, process_all=True):
        # Scans and processes new documents. If folder is None, uses old. If process_all=True,
        # ignores documents already processed based on name.
        if folder is None:
            folder = self.docfolder
        assert os.path.exists(folder)

        docdata = self._process_text(folder, process_all=process_all)
        self._build_documents(docdata)
