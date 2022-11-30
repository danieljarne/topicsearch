from classes.GraphEngine import GraphEngine
import os
import logging

try:
    logging.basicConfig(filename=os.path.join("logs", 'dev.log'), level=logging.DEBUG, format='%(asctime)s %(message)s')
except:
    os.mkdir(os.path.join(".","logs"))
    logging.basicConfig(filename=os.path.join("logs", 'dev.log'), level=logging.DEBUG, format='%(asctime)s %(message)s')
modelPath = os.path.join(os.getcwd(), "models", "spacy")#, "crawl-300d-2M")
docPath = os.path.join(os.getcwd(), "docs")


# WE ASSUME WE START FROM SCRATCH. The steps to process documents are:
# 1. Load models in graph engine (done here). The option impor allows you to import already processed documents.
#       Alert: Setting impor=True does not load nlp model (takes long time). Therefore we cant query docs. Just
#       manipulate already generated.
g = GraphEngine(modelPath, impor=False)

# 2. By initializing, we just process the text and convert it to vectors. To compute the intradoc similarity (similarity
#    between each document sentences with eachother) run:
# g._intradoc_sim()


# 3. If we now want to export the vectorized docs and the similarities to pkl files (to not have to run NLP engine) do:
# g.export_docVec_pkl()

# 4. Now we can query the document graph for word similarities. Very simple output; just shows a dictionary with 10 most similar sentences.
# print("word: "+"ferrari   "+str(g.query("ferrari")))
# print("word: "+"reptil   "+str(g.query("reptile")))
# print("word: "+"piramide   "+str(g.query("pyramids")))
# print("word: "+"mecanica   "+str(g.query("mechanics")))


# print(g.intradocSim[1, 1])
# print(g.intradocSim.shape)

# 5. Creates neo4j graph from document similarities. Neo4J has to be running!
# g.build_graph()
# g.add_similarities_to_graph(host='localhost', password='test')

# 6. We can process new documents with
# g.scan_new(folder="new/folder")
g.doc_pipeline(docPath)