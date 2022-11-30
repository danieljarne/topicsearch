# topicsearchgraph

Topic search engine and visualisation for text documents.

This small prototype project creates a graph out of documents splitting them and uses semantic vector
representation (FastText) to generate a graph of documents and its topics. It is then searchable by word queries 
(i.e. query the graph with a topic and returns the top documents that include the topic). It also builds the weighted graph using document similarities.

See TopicSearch.py for test script. Instructions are commented out.

IMPORTANT: 
Install fasttext models:
- https://fasttext.cc/docs/en/crawl-vectors.html
- Download wiki-news or crawl model (text)
- Convert model to spacy format: https://spacy.io/usage/vectors-similarity#converting
- Copy all files of the model under ./models/spacy

You need to run a Neo4j server for the graph functionalities to work. See https://github.com/neo4j/neo4j.
- Install neo4j desktop
- Create new project
- Add graph
- Set password "test"
- Start graph


