import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os, os.path
import nltk
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.analysis import FancyAnalyzer
from whoosh import scoring
from whoosh.filedb.filestore import FileStorage
from whoosh import index
from whoosh import qparser
from whoosh.qparser import MultifieldParser
from whoosh.qparser import FuzzyTermPlugin
import re
from operator import attrgetter
from nltk.stem import PorterStemmer
import pyltr

porter_stemmer = PorterStemmer()

# Method to read input from CSV file
def ReadInput():
    global paragraphs, questions, test, trainraw
    paragraphs = pd.read_csv('paragraphs.csv')
    questions = pd.read_csv('questions.csv')
    questions['qid'] = pd.to_numeric(questions['qid'])
    questions['qrownum'] = np.arange(len(questions))
    test = pd.read_csv('test.csv')
    # split the qid and paraid in the test csv file
    test[['qid', 'ParaId']] = test['qpid'].str.split('#', expand=True)
    test['qid'] = pd.to_numeric(test['qid'])
    test['ParaId'] = pd.to_numeric(test['ParaId'])
    trainraw = pd.read_csv('train.csv')

# Method to stem text using porter stemmer
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Method for some basic preprocessing
def preProcessParaTextFields(df):
    df["ParaText"] = df["ParaText"].str.replace("[^0-9a-zA-Z]", " ")
    df["ParaText"] = df["ParaText"].str.lower()
    df["ParaText"] = df["ParaText"].str.strip()
    df["ParaText"] = df.ParaText.apply(stem_sentences)
    df["Title"] = df["Title"].str.replace("[^0-9a-zA-Z]", " ")
    df["Title"] = df["Title"].str.lower()
    df["Title"] = df["Title"].str.strip()
    df["Title"] = df.Title.apply(stem_sentences)

def preProcessQueryTextFields(df):
    df["qtext"] = df["qtext"].str.strip()
    df["qtext"] = df["qtext"].str.replace("[^0-9a-zA-Z]", " ")
    df["qtext"] = df["qtext"].str.lower()
    df["qtext"] = df.qtext.apply(stem_sentences)

# Takes a vector of texts and embeds into Universal Sentence encoder from tensorflow
def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))

# takes two embedding vectors and calculated cosine similarity
def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

# Method for doing cross products on two dfs
def cartesian_product_basic(left, right):
    return (
        left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))

# For BM25 calculation I using a library called Whoosh
# this method creates an index and schema that we will use for search
def CreateIndex():
    global ix
    schema = Schema(ParaId=NUMERIC(stored=True),
                    Title=TEXT(analyzer=FancyAnalyzer(), field_boost=2.0),
                    ParaText=TEXT(analyzer=FancyAnalyzer(), stored=True)
                    )
    if not os.path.exists("AIML_Index"):
        os.mkdir("AIML_Index")
    ix = index.create_in("AIML_Index", schema)
    with ix.writer()as writer:
        for i in paragraphs.index:
            writer.update_document(ParaId=str(paragraphs.loc[i, "ParaId"]),
                                   Title=str(paragraphs.loc[i, "Title"]),
                                   ParaText=str(paragraphs.loc[i, "ParaText"]))

# Creates a query parser for parsing our quesitons
def CreateQueryParser():
    global qp
    og = qparser.OrGroup.factory(0.9)
    qp = qparser.MultifieldParser(["Title", "ParaText"], schema=ix.schema, group=og)
    qp.add_plugin(FuzzyTermPlugin())


ReadInput()
print("Input read")

paragraphs.fillna('', inplace=True)
questions.fillna('', inplace=True)
preProcessParaTextFields(paragraphs)
preProcessQueryTextFields(questions)

# joining the questions set with trainset
trainQuery = pd.merge(trainraw, questions, how="inner", on="qid")
# also adding some non relevant paragraphs for each question to help in training
# as we need negative examples also
totalTrain = cartesian_product_basic(trainQuery, paragraphs)
# setting relevance label as zero
totalTrain["Relevance"] = 0
# setting relevance label as one only for query and question pairs that are present in original train set
totalTrain.loc[totalTrain["ParaId_x"] == totalTrain["ParaId_y"], "Relevance"] = 1
relevantTrain = totalTrain[totalTrain["ParaId_x"] == totalTrain["ParaId_y"]]
nonrelevantTrain = totalTrain[totalTrain["ParaId_x"] != totalTrain["ParaId_y"]]
# taking only 30 non relevant paragraphs for each query to keep train set size small
nonrelevantTrain = nonrelevantTrain.groupby("qid").head(30)
trainSet = relevantTrain.append(nonrelevantTrain, ignore_index=True)
trainSet = trainSet.sort_values(by="qid")
del trainSet["ParaId_x"]
trainSet.rename(columns={'ParaId_y': 'ParaId'}, inplace=True)

# preparing test set by mergin test data with correponding questions and para texts
testQuery = pd.merge(test, questions, how="inner", on="qid")
testSet = pd.merge(testQuery, paragraphs, how="inner", on="ParaId")

print("load module")
# tensroflow hub module for Universal sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

# Embeding paragraphs using Universal sentence encoder
print("generate features for paragraphs")
BASE_VECTORS = get_features(paragraphs["ParaText"].tolist())
print(BASE_VECTORS.shape)

# Embeding questions using Universal sentence encoder
print("generate features for questions")
QUERY_VECTORS = get_features(questions["qtext"].tolist())
print(QUERY_VECTORS.shape)

# for each question paragraph pair in test set calculate similarity
print("generate similarity for testset")
uniqueqids = pd.unique(testSet["qid"])
# queryresults = []
for qid in uniqueqids:
    queryrownumber = questions[questions['qid'] == qid].iloc[0]["qrownum"]
    print("Extracting features..." + str(qid))
    query_vec = QUERY_VECTORS[queryrownumber]
    parasforquestion = testSet[testSet['qid'] == qid]
    for i in parasforquestion.index:
        paraid = parasforquestion.loc[i, "ParaId"]
        paravec = BASE_VECTORS[paraid]
        sim = cosine_similarity(query_vec, paravec)
        rowindex = parasforquestion.loc[i, "id"]
        # queryresults.append([rowindex, sim,qid, paraid])
        testSet.loc[testSet["id"] == rowindex, 'similarity'] = sim

print("generate similarity for trainset")
uniquetrainqids = pd.unique(trainSet["qid"])
# queryresults = []
for qid in uniquetrainqids:
    queryrownumber = questions[questions['qid'] == qid].iloc[0]["qrownum"]
    print("Extracting features..." + str(qid))
    query_vec = QUERY_VECTORS[queryrownumber]
    parasforquestion = trainSet[trainSet['qid'] == qid]
    for i in parasforquestion.index:
        paraid = parasforquestion.loc[i, "ParaId"]
        paravec = BASE_VECTORS[paraid]
        sim = cosine_similarity(query_vec, paravec)
        # queryresults.append([rowindex, sim,qid, paraid])
        trainSet.loc[(trainSet['ParaId'] == paraid) & (trainSet['qid'] == qid), 'similarity'] = sim

CreateIndex()
CreateQueryParser()

print("index created")

# Get BM25 for train set on para and title
truniqueqids = pd.unique(trainSet["qid"])
trainSet["BM25Para"] = 0;
trpararesults = []
with ix.searcher() as s:
    for qid in truniqueqids:
        query = questions[questions['qid'] == qid].iloc[0]["qtext"]
        q = qp.parse(query)
        results = s.search(q, limit=20)
        parasforquestion = trainSet[trainSet['qid'] == qid]
        for i in parasforquestion.index:
            paraid = parasforquestion.loc[i, "ParaId"]
            matches = [x for x in results if int(x["ParaId"]) == paraid]
            for match in matches:
                trpararesults.append([qid, paraid, match.score])
for para in trpararesults:
    trainSet.loc[(trainSet['ParaId'] == para[1]) & (trainSet['qid'] == para[0]), "BM25Para"] = para[2]
print("train set BM25 calculated")

# Get BM25 for test set on para and title
tuniqueqids = pd.unique(testSet["qid"])
testSet["BM25Para"] = 0;
testresults = []
with ix.searcher() as s:
    for qid in tuniqueqids:
        query = questions[questions['qid'] == qid].iloc[0]["qtext"]
        q = qp.parse(query)
        results = s.search(q, limit=1)
        parasforquestion = testSet[testSet['qid'] == qid]
        for i in parasforquestion.index:
            paraid = parasforquestion.loc[i, "ParaId"]
            matches = [x for x in results if int(x["ParaId"]) == paraid]
            for match in matches:
                testresults.append([qid, paraid, match.score])
for para in testresults:
    testSet.loc[(testSet['ParaId'] == para[1]) & (testSet['qid'] == para[0]), "BM25Para"] = para[2]

print("test set BM25 calculated")

# replace nan's with 0
trainSet["BM25Para"].fillna(0, inplace=True)
testSet["BM25Para"].fillna(0, inplace=True)
trainSet["similarity"].fillna(0, inplace=True)
testSet["similarity"].fillna(0, inplace=True)

#trainSet.to_csv('trainSetwithfeatures.csv', sep=',', encoding='utf-8', index=False)

# Using Pyltr to train a lambda mart model for ranking the results
# Relvance labels for train set
TY = trainSet["Relevance"].to_numpy()
Tqids = trainSet["qid"]
# Features to use for training using BM25Para and similarity
TX = trainSet[["BM25Para", "similarity"]].to_numpy()
# We are using NDGC. This is metric the model will maximize
metric = pyltr.metrics.NDCG(k=1)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    max_depth = 6,
    learning_rate=0.02,
    # max_features=0.5,
    # query_subsample=0.5,
    # max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)
# Fit the model
model.fit(TX, TY, Tqids)
# test features same as train set
TestX = testSet[["BM25Para", "similarity"]].to_numpy()
# predict the ranks using model
Tpred = model.predict(TestX)
Testqids = testSet["qid"]
TestParas = testSet["ParaId"]
testSet["Rank"] = Tpred
testSet["target"] = 0
# take largest ranked para for each query and set target as 1 for that
testSet.loc[testSet.groupby('qid').Rank.nlargest(1).index.get_level_values(1), "target"] = 1
#testSet.to_csv('testSetPrediction.csv', sep=',', encoding='utf-8', index=False)

# write the index and target to result file that we will submit
bestmatchingresultset = pd.DataFrame(testSet, columns=['id', 'target'])
bestmatchingresultset.to_csv('resultUSIC.csv', sep=',', encoding='utf-8', index=False)

# print feature importance. BM25 most significant feature
print(model.feature_importances_)
