
import re
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from cvxpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


def newDic():
	fo = open("combination.txt", "r")
	line = fo.readlines()
	KEY = []
	UI = []
	new_dict = {}
	count = 0
	for i in line:
		if i.startswith("*NEWREC"):
			count = 0
		if i.startswith("MH = ") or i.startswith("NM = "):	
			index = i.index("=")
			KEY.append(i[index+2: len(i)-1])
			count = count + 1	
		if i.startswith("SY = "):
			idx1 = i.index("=")
			idx2 = i.index("|")
			KEY.append(i[idx1+2: idx2])
			count = count + 1
		if i.startswith("UI = "):
			index = i.index("=")
			for j in range (count):
				UI.append(i[index+2: len(i)-1])
			

	assert(len(KEY) == len(UI))
	
	for i in range(len(KEY)):
		KEY[i] = KEY[i].lower()
		if KEY[i]in new_dict:
		 	new_dict[KEY[i]] = (new_dict[KEY[i]])+(UI[i])
		else:
		 	new_dict[KEY[i]] = UI[i]  


	return new_dict
	


def develDic():
	fo = open("CDR_CN_devel.json", "r")
	line = fo.readlines()
	KEY = []
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
				KEY.append(pair[index+19: len(pair)-2])
				UI.append(pair[index+3:index+5])
			else:
				KEY.append(pair[index+24: len(pair)-2])
				UI.append(pair[index+3:index+10])

	for i in range(len(KEY)):
		KEY[i] = KEY[i].lower()
		if KEY[i]in train_dict:
			if 	train_dict[KEY[i]] != UI[i]:
				train_dict[KEY[i]] = (train_dict[KEY[i]])+(UI[i])
		else:
			train_dict[KEY[i]] = UI[i]	 

	return train_dict



def testDic():
	fo = open("CDR_CN_test.json", "r") 	

	line = fo.readlines()
	KEY = []
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
				KEY.append(pair[index+19: len(pair)-2])
				UI.append(pair[index+3:index+5])
			else:
				KEY.append(pair[index+24: len(pair)-2])
				UI.append(pair[index+3:index+10])


	for i in range(len(KEY)):
		KEY[i] = KEY[i].lower()		
		if KEY[i]in train_dict:	
			if 	train_dict[KEY[i]] != UI[i]:
				train_dict[KEY[i]] = (train_dict[KEY[i]])+(UI[i]) 
		else:
			train_dict[KEY[i]] = UI[i]	 

	return train_dict


def trainDic():
	fo = open("CDR_CN_train.json", "r") 	

	line = fo.readlines()
	KEY = []
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
				KEY.append(pair[index+19: len(pair)-2])
				UI.append(pair[index+3:index+5])
			else:
				KEY.append(pair[index+24: len(pair)-2])
				UI.append(pair[index+3:index+10])


	for i in range(len(KEY)):
		KEY[i] = KEY[i].lower()		
		if KEY[i]in train_dict:	
			if 	train_dict[KEY[i]] != UI[i]:
				train_dict[KEY[i]] = (train_dict[KEY[i]])+(UI[i]) 
		else:
			train_dict[KEY[i]] = UI[i]	 

	return train_dict


def trainScore():
	new_dic = {}
	train_dic = {}

	new_dic = newDic()
	train_dic = trainDic()
	
	y_true =[]
	y_pre =[]

	for k, v in train_dic.items():
		value = new_dic.get(k, "NA")
		y_true.append(v)
		y_pre.append(value)
		

	assert(len(y_true) == len(y_pre))

	precision = precision_score(y_true, y_pre, average='micro')  
	recall = recall_score(y_true, y_pre, average='micro')
	f1 = f1_score(y_true, y_pre, average='micro')

	print("precision")
	print (precision)
	print("recall")
	print (recall)
	print("f1")
	print (f1)

def testScore():
	new_dic = {}
	train_dic = {}

	new_dic = newDic()
	train_dic = testDic()

	y_true =[]
	y_pre =[]

	for k, v in train_dic.items():
		value = new_dic.get(k, "NA")
		y_true.append(v)
		y_pre.append(value)
		

	assert(len(y_true) == len(y_pre))

	precision = precision_score(y_true, y_pre, average='micro')  
	recall = recall_score(y_true, y_pre, average='micro')
	f1 = f1_score(y_true, y_pre, average='micro')

	print("precision")
	print (precision)
	print("recall")
	print (recall)
	print("f1")
	print (f1)

def develScore():
	new_dic = {}
	train_dic = {}

	new_dic = newDic()
	train_dic = develDic()
	

	y_true =[]
	y_pre =[]

	for k, v in train_dic.items():
		value = new_dic.get(k, "NA")
		y_true.append(v)
		y_pre.append(value)
		

	assert(len(y_true) == len(y_pre))

	precision = precision_score(y_true, y_pre, average='micro')  
	recall = recall_score(y_true, y_pre, average='micro')
	f1 = f1_score(y_true, y_pre, average='micro')

	print("precision")
	print (precision)
	print("recall")
	print (recall)
	print("f1")
	print (f1)

######
def API():
	print ("Setting up the system ...")
	new_dic = {}
	new_dic = newDic()
	print ("System setup finished!")

	key = input("Please enter an entity name: ")
	value = new_dic.get(key, "NA")
	if value != "NA":
		print ("This is the corresponding MeSH concept: ")
		print (value)
	else:
		print ("Sorry, cannot find the entity")
####



		

	




#newDic()
#develDic()
#trainDic()
#testDic()

#trainScore()
#testScore()
#develScore()
API()





