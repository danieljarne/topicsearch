from py2neo import Node, Relationship, Subgraph, Graph

g1 = Graph(host="localhost", password="test")
n = g1.nodes.match("Sentence")
for nod in n:
    print(nod)