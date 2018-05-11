
import re
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from learning2rank.rank import RankNet
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from cvxpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix



#******************************************************
"""
This function is used to get the sub-corpus from the combination data set, which is combined 
from descriptor.txt and supplement.txt. The combination.txt serves as our dictionary resouce. 
In this funtion, each entity and its corresponding synonyms have been put together as a single 
vector
"""
def get_new_dic_corpus_from_Raw():
	
	corpus = []

	fo = open("combination.txt", "r")
	line = fo.readlines()

	tempVector = ""
	for i in line:
		if i.startswith("*NEWREC"):
			tempVector = ""
		if i.startswith("MH = ") or i.startswith("NM = "):	
			index = i.index("=")
			tempVector = i[index+2: len(i)-1]
		
		if i.startswith("SY = "):
			idx1 = i.index("=")
			idx2 = i.index("|")	
			tempVector = tempVector + " " + i[idx1+2: idx2]
					
		if i.startswith("UI = "):
			corpus.append(tempVector)

	return corpus

"""
This function is used to get the sub-corpus from the training data set. Each entity in the training 
data has been expressed as a single vector 
"""
def get_train_corpus_from_Raw():
	corpus = []

	fo = open("CDR_CN_train.json", "r") 	
	line = fo.readlines()
	KEY = []
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
		
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
				
			if pair[index+3:index+5]== "-1":
				KEY.append(pair[index+19: len(pair)-2])
				
			else:
				KEY.append(pair[index+24: len(pair)-2])
				
	for i in range(len(KEY)):
		KEY[i] = KEY[i].lower()
		corpus.append(KEY[i])


	return corpus

"""
This function is used to get the sub-corpus from the developing data set, which is a supplementation 
of the training data set.Each entity in the training data has been expressed as a single vector 
"""

def get_devel_corpus_from_Raw():
	corpus = []
	fo = open("CDR_CN_devel.json", "r") 	
	line = fo.readlines()
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
			
			if pair[index+3:index+5]== "-1":
				temp = pair[index+19: len(pair)-2]
				corpus.append(temp)
	
			else:
				temp = pair[index+24: len(pair)-2]
				corpus.append(temp)
				
	for i in range(len(corpus)):
		corpus[i] = corpus[i].lower()
	return corpus	

"""
This function is used to get the sub-corpus from the test data set.
Each entity in the training data has been expressed as a single vector 
"""

def get_test_corpus_from_Raw():
	corpus = []

	fo = open("CDR_CN_test.json", "r") 	
	line = fo.readlines()
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
			
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
			
			
			if pair[index+3:index+5]== "-1":
				temp = pair[index+19: len(pair)-2]
				corpus.append(temp)
				
			else:
				temp = pair[index+24: len(pair)-2]
				corpus.append(temp)
				
	for i in range(len(corpus)):
		corpus[i] = corpus[i].lower()
	return corpus	


"""
This function is used to get the corresponding label for the combination.txt, and it serves as
out dictionary. The sequence of the label and the sequence of the vector in get_new_dic_corpus_from_Raw()
funtion are correspondent.
"""
def get_new_dic_label_from_Raw():
	UI = []
	fo = open("combination.txt", "r")
	line = fo.readlines()

	for i in line:	
		if i.startswith("UI = "):
			index = i.index("=")
			UI.append(i[index+2: len(i)-1])
	
	return UI

"""
This function is used to get the corresponding label for the training data set. 
The sequence of the label and the sequence of the vector in get_train_corpus_from_Raw()
funtion are correspondent.
"""
def get_train_label_from_Raw():
	fo = open("CDR_CN_train.json", "r") 	
	line = fo.readlines()
	UI = []
	train_dict = dict()
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
			
			if pair[index+3:index+5]== "-1":
				UI.append(pair[index+3:index+5])
			else:
				UI.append(pair[index+3:index+10])

	return UI

"""
This function is used to get the corresponding label for the developing data set. It serves as
the supplementation of the training data set. 
The sequence of the label and the sequence of the vector in get_devel_corpus_from_Raw()
funtion are correspondent.
"""
def get_devel_label_from_Raw():
	fo = open("CDR_CN_devel.json", "r") 	

	line = fo.readlines()
	UI = []
	train_dict = dict()
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
			
			if pair[index+3:index+5]== "-1":
				UI.append(pair[index+3:index+5])
			else:
				UI.append(pair[index+3:index+10])

	return UI

"""
This function is used to get the corresponding label for the test data set. 
The sequence of the label and the sequence of the vector in get_test_corpus_from_Raw()
funtion are correspondent.
"""

def get_test_label_from_Raw():
	fo = open("CDR_CN_test.json", "r") 	
	line = fo.readlines()
	UI = []
	train_dict = dict()
	pair = ""
	
	for i in line:
		searchObj = re.search( r'{"concept"(.*?)}', i, re.M|re.I)
		if searchObj:
			pair = searchObj.group()
	
		if pair.startswith("concept", 2, 9):			
			index = pair.index(":")
			if pair[index+3:index+5]== "-1":
				UI.append(pair[index+3:index+5])
			else:
				UI.append(pair[index+3:index+10])

	return UI



"""
This function is used to get the entire corpus from all the source document, and then 
get the TF-IDF expression for text features.
Word-of-Bags text feature are used, where each word counts for one dimension.
"""
def get_entire_corpus_counts():
	
	new_dic_corpus = get_new_dic_corpus_from_Raw()
	train_corpus = get_train_corpus_from_Raw()
	devel_corpus = get_devel_corpus_from_Raw()
	test_corpus = get_test_corpus_from_Raw()

	
	vectorizer = TfidfVectorizer()
	
	entire_corpus = new_dic_corpus
	entire_corpus+=(train_corpus)
	entire_corpus+=(devel_corpus)
	entire_corpus+=(test_corpus)
	
	X = vectorizer.fit_transform(entire_corpus)

	counts = X.toarray()

	return counts

"""
This function is used to get the corresponding label for each of the record 
from all the source document,
"""
def get_entire_corpus_label():
	
	new_dic_label = get_new_dic_label_from_Raw()
	train_label = get_train_label_from_Raw()
	devel_label = get_devel_label_from_Raw()
	test_label = get_test_label_from_Raw()	
	
	entire_label = new_dic_label
	entire_label+=(train_label)
	entire_label+=(devel_label)
	entire_label+=(test_label)
	

	return entire_label

"""
This function is used to reduce the dimension of text feature
In order to increase the running time, a for loop is used for the 
dimension reduction task. A package from SKlearn is used. This transformer 
performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). 
"""
def dense():
	counts = get_entire_corpus_counts()

	svd = TruncatedSVD(n_components=100, n_iter=1)
	
	np_counts = np.asarray(counts)

	dense_counts = np.empty(shape=[0, 100])
	for i in range (0, 2764):
		part_i = svd.fit_transform(np_counts[i:i+100, :])
		dense_counts = np.concatenate((dense_counts, part_i), axis = 0)

	np.savetxt("counts.txt", dense_counts)

	return dense_counts


"""
This function is used to find the optimizaed W matrix, CVXPY package is used. 
Our model has been trained in this function.
"""
def optimization():
	counts = np.loadtxt("counts_copy.txt")
	label = get_entire_corpus_label()
	

	new_dic_corpus = get_new_dic_corpus_from_Raw()
	train_corpus = get_train_corpus_from_Raw()
	devel_corpus = get_devel_corpus_from_Raw()
	test_corpus = get_test_corpus_from_Raw()

	size_1 = len(new_dic_corpus)
	size_2 = len(new_dic_corpus) + len(train_corpus)
	size_3 = len(new_dic_corpus) + len(train_corpus) + len(devel_corpus)
	size_4 = len(new_dic_corpus) + len(train_corpus) + len(devel_corpus) + len(test_corpus)
	
	
	dic = counts[:size_1]
	train = counts[size_1:size_3]
	test = counts[size_3:]

	new_dic_label = label[:size_1]
	train_label = label[size_1:size_3] 
	test_label = label[size_3:size_4]

	reversed_dic = {}
	for i in range (len(dic)):
		if new_dic_label[i] not in reversed_dic:
			reversed_dic[new_dic_label[i]] = i

	
	W = Variable(100, 100)
	expr_list = []
	W = np.identity (100)
	for i in range (len(train)):
		label = train_label[i]
		idx = reversed_dic.get(label, 1)
		vec = dic[idx]
		np_vec = np.asarray(vec)
		np_vec = np_vec.reshape(100, 1)
		train_i = np.asarray(train[i])
		train_i = train_i.reshape(100, 1)
		sT = sparse.csr_matrix(train_i)
		STT = csr_matrix.transpose(sT)
		score_1_1 = STT.dot(W)
		score_1 = np.dot (score_1_1, np_vec)


		for con in range (0, 25):
			idx_d = random.randint(0, 264850)
			if idx_d != idx:
				np_data = np.asarray(dic[idx_d])
				np_data = np_data.reshape(100, 1)
				score_1_1s = sparse.csr_matrix(score_1_1)
				score_2 = score_1_1s.dot(np_data)
				expr_list.append( max (0, 1 - score_1 + score_2) )

				if (1 - score_1 + score_2 > 0):
					W = [[W[x][y] + 0.0001* sT.dot(np_vec.T)[x][y]- 0.0001*sT.dot(np_data.T)[x][y] for y in range(0, 100)] for x in range(0, 100)]					
	
		print (i)

	np.savetxt("W_matrix_before copy 2.txt", W)
	prob = Problem (Minimize(sum(expr_list)))
	result = prob.solve()
	
	np.savetxt("W_matrix copy 2.txt", W)


"""
This function is used to evaluate the precision, recall, and F1 score for 
our model
"""
def score():
	counts = np.loadtxt("counts_copy.txt")
	label = get_entire_corpus_label()
	


	new_dic_corpus = get_new_dic_corpus_from_Raw()
	train_corpus = get_train_corpus_from_Raw()
	devel_corpus = get_devel_corpus_from_Raw()
	test_corpus = get_test_corpus_from_Raw()

	size_1 = len(new_dic_corpus)
	size_2 = len(new_dic_corpus) + len(train_corpus)
	size_3 = len(new_dic_corpus) + len(train_corpus) + len(devel_corpus)
	size_4 = len(new_dic_corpus) + len(train_corpus) + len(devel_corpus) + len(test_corpus)
	

	dic = counts[:size_1]
	train = counts[size_1:size_3]
	devel = counts[size_2:size_3]
	test = counts[size_3:]
	

	new_dic_label = label[:size_1]
	train_label = label[size_1:size_3] 
	devel_label = label[size_2:size_3]
	test_label = label[size_3:size_4]


	W = np.loadtxt("W_matrix.txt")

	y_true =[]
	y_pre =[]

	for i in range (len(test)):
		
		label = test_label[i]
		mini_score = 0;

		test_i = np.asarray(test[i])
		test_i = test_i.reshape(100, 1)		
		sT = sparse.csr_matrix(test_i)		
		STT = csr_matrix.transpose(sT)	
		score_1_1 = STT.dot(W)

		for idx_s in range (len(dic)):
			vec = dic[idx_s]
			np_vec = np.asarray(vec)
			np_vec = np_vec.reshape(100, 1)
			score_1 = np.dot (score_1_1, np_vec)
			if (score_1 < mini_score):
				mini_score = score_1
				pre_label = new_dic_label[idx_s]
			
		y_pre.append(pre_label)
		y_true.append(label)

		print (i)
		

	precision = precision_score(y_true, y_pre, average='micro')  
	recall = recall_score(y_true, y_pre, average='micro')
	f1 = f1_score(y_true, y_pre, average='micro')

	print("precision")
	print (precision)
	print("recall")
	print (recall)
	print("f1")
	print (f1)



"""
The following functions are useless, they were used at beginning but were replaced by
other functions of other methods. Keep them for record.
"""
#*****************************************
def get_wordVector():
	new_dic = {}
	train_dic = {}
	test_dic = {}
	devel_dic = {}


	new_dic = newDic()
	train_dic = trainDic()
	test_dic = testDic()
	devel_dic = develDic()

	

	vectorizer = CountVectorizer()
	corpus = []
	
	for k in new_dic.keys():
		corpus.append(k)


	for k in train_dic.keys():
		corpus.append(k)

	for k in test_dic.keys():
		corpus.append(k)

	for k in devel_dic.keys():
		corpus.append(k)

	
	X = vectorizer.fit_transform(corpus)

	counts = X.toarray()

	print (counts)
	return counts

	
def get_tfIdf():
	counts = get_wordVector()

	transformer = TfidfTransformer(smooth_idf=False)

	tfidf = transformer.fit_transform(counts)

	vector = tfidf.toarray() 

	return vector
	
def get_recordBinaryVecotor():	
	wordVector = get_wordVector()

	fo = open("combination.txt", "r")
	line = fo.readlines()
	
	recordVectorAll = []
	
	for i in line:
		if i.startswith("*NEWREC"):
			tempVector = []
		if i.startswith("MH = ") or i.startswith("NM = "):	
			index = i.index("=")
			tempList = i[index+2: len(i)-1].split(" ")

			for j in tempList:
				j = j.lower()
				j = j.replace(',','')
				tempVector.append(j)
			
		if i.startswith("SY = "):
			idx1 = i.index("=")
			idx2 = i.index("|")	
			tempList = i[idx1+2: idx2-1].split(" ")
			
			for j in tempList:
				j = j.lower()
				j = j.replace(',','')
				tempVector.append(j)
			
		if i.startswith("UI = "):
			recordVectorAll.append(tempVector)


	recordBinaryVectorAll = []
	
	for r in recordVectorAll:
		recordBinaryVector = [0] * len(wordVector)
		for w in r:
			index = wordVector.index(w)
			recordBinaryVector[index] = recordBinaryVector[index] + 1
		recordBinaryVectorAll.append(recordBinaryVector)

	print (recordBinaryVectorAll)


def get_tfIDF_representation():
	new_dic = {}
	train_dic = {}
	test_dic = {}
	devel_dic = {}
	new_dic = newDic()
	train_dic = trainDic()
	test_dic = testDic()
	devel_dic = develDic()


	vectorizer = TfidfVectorizer()
	corpus = []
	
	for k in new_dic.keys():
		corpus.append(k)


	for k in train_dic.keys():
		corpus.append(k)

	for k in test_dic.keys():
		corpus.append(k)

	for k in devel_dic.keys():
		corpus.append(k)

	X = vectorizer.fit_transform(corpus)

	counts = X.toarray()

	print (counts)
	
	return counts

def split_new_dic_tfIDF_representation():
	new_dic = {}
	new_dic = newDic()

	size = len(new_dic)

	counts = get_tfIDF_representation()
	split_new_dic_tfIDF_representation = counts[:size]

	return split_new_dic_tfIDF_representation


def split_train_tfIDF_representation():
	new_dic = {}
	new_dic = newDic()
	train_dic = {}
	train_dic = trainDic()

	sizestart = len(new_dic)
	size = len(new_dic) + len(train_dic)

	counts = get_tfIDF_representation()
	split_train_tfIDF_representation = counts[sizestart:size]

	return split_train_tfIDF_representation



def split_test_tfIDF_representation():
	new_dic = {}
	new_dic = newDic()
	train_dic = {}
	train_dic = trainDic()
	test_dic = {}
	test_dic = test_dic()

	sizestart = len(new_dic) + len(train_dic)
	size = len(new_dic) + len(train_dic) + len(test_dic)

	counts = get_tfIDF_representation()
	split_test_tfIDF_representation = counts[sizestart:size]

	return split_test_tfIDF_representation


def split_devel_tfIDF_representation():
	new_dic = {}
	train_dic = {}
	test_dic = {}
	devel_dic = {}
	new_dic = newDic()
	train_dic = trainDic()
	test_dic = testDic()
	devel_dic = develDic()

	sizestart = len(new_dic) + len(train_dic) + len(test_dic)
	size = len(new_dic) + len(train_dic) + len(test_dic) + len(devel_dic)

	counts = get_tfIDF_representation()
	split_devel_tfIDF_representation = counts[sizestart:size]

	return split_devel_tfIDF_representation

def get_counts():
	new_dic = {}
	train_dic = {}
	test_dic = {}
	devel_dic = {}
	new_dic = newDic()
	train_dic = trainDic()
	test_dic = testDic()
	devel_dic = develDic()


	vectorizer = TfidfVectorizer()
	corpus = []
	
	for k, v in new_dic.items():
		corpus.append(k)
		

	for k, v in train_dic.items():
		corpus.append(k)
		
	
	for k, v in devel_dic.items():
		corpus.append(k)
		

	for k, v in test_dic.items():
		corpus.append(k)
		

	
	print (type(corpus))
	X = vectorizer.fit_transform(corpus)

	counts = X.toarray()

	print (counts.shape)

	return counts
#*****************************************

"""
Uncomment one of the following function to run 
"""
#get_new_dic_corpus_from_Raw()
#get_train_corpus_from_Raw()
#get_test_corpus_from_Raw()
#get_devel_corpus_from_Raw()
#get_entire_corpus_counts()
#get_entire_corpus_label()
#dense()

optimization()
#score()





#*****************************************
"""
The following functions are useless, they were used at beginning but were replaced by
other functions of other methods. Keep them for record.
"""
#get_wordVector()
#get_recordBinaryVecotor()
#get_tfIdf()
#get_tfIDF_representation()
#get_counts()


