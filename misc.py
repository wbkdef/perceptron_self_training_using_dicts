import operator
import random
import itertools as it
import math

def argmax(values):
	max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
	return max_index

def sign(num):
	if num == 0:
		assert False
	if num>0:
		return 1
	else:
		return 0

def pause():
    raw_input("Press enter to continue")

def sortedInd(x,reverse=False):
	return [i[0] for i in sorted(enumerate(x),reverse=reverse,key=lambda x:x[1])]

def random_round(num):
	return int(num)+(random.random()<num%1)

def gen_corrupted_features1(orig_feats,masking_rate,feats_to_sample_from=set(['a','b','c']),num_to_add=0):
	num_feats_to_keep=random_round((1-masking_rate)*len(orig_feats))
	num_to_add=random_round(num_to_add)
	corrupted_feats = set(random.sample(orig_feats,num_feats_to_keep))
	corrupted_feats|=set(random.sample(feats_to_sample_from,num_to_add))
	return corrupted_feats	

def gen_corrupted_features(list_of_orig_feats,labels,masking_rate,
	num_corruptions,all_feats=set(['a','b','c']),num_to_add=0):
    assert(masking_rate>=0 and masking_rate<1)
    assert(num_to_add>=0)
    sample_size_to_sub_sample_from=int(math.sqrt(len(all_feats)))
    print "sample_size_to_sub_sample_from", sample_size_to_sub_sample_from
    num_sampled=1000000 #A big # so will sample immediately
    new_feats=[]
    new_labels=[]
    for i in range(num_corruptions):
    	for feats, label in it.izip(list_of_orig_feats,labels):
    		if num_sampled > sample_size_to_sub_sample_from:
    			sub_sample=random.sample(all_feats,sample_size_to_sub_sample_from)
    			num_sampled=0
    		corrupted_feats=gen_corrupted_features1(feats,masking_rate,sub_sample,num_to_add)
    		num_sampled+=num_to_add
    		new_feats.append(corrupted_feats)
    		new_labels.append(label)
    return new_feats, new_labels


def get_all_features(list_of_sets_of_features):
	all_feats=set([])
	print "Generating set of all features from ", len(list_of_sets_of_features), "sets"
	for i in range(len(list_of_sets_of_features)):
		feats=list_of_sets_of_features[i]
		all_feats.update(feats)
		if i%1000==0:
			print i,
	return all_feats


def test_gen_corrupted_features():
	digits=range(10)
	list_of_orig_feats=[random.sample(digits,4) for i in range(3)]
	labels=[random.randint(0,3) for i in range(3)]
	masking_rate=.3
	num_corruptions=4
	all_feats=get_all_features(list_of_orig_feats)
	num_to_add=1.8
	new_feats, new_labels = gen_corrupted_features(list_of_orig_feats,labels,masking_rate,num_corruptions,
		all_feats,num_to_add)
	print "all_feats",all_feats
	print "Original Data"
	for label, feats in it.izip(labels,list_of_orig_feats):
		print label, feats
	print "Corrupted Data"
	for label, feats in it.izip(new_labels,new_feats):
		print label, feats
test_gen_corrupted_features()
