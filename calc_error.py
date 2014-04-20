from __future__ import division
import itertools

def error(predicted_labels,correct_labels):
	pl,cl,n=predicted_labels,correct_labels,len(predicted_labels)
	num_errors=sum([pl[i]!=cl[i] for i in range(n)])
	return num_errors, num_errors/n

def calc_error(list_of_sets_of_features,classifier,correct_labels):
	"""classifier should have a function 'def predict_label(self,feats):'"""
	predicted_labels=classifier.predict_labels(list_of_sets_of_features)
	#[classifier.predict_label(feats) for feats in list_of_features]
	return error(predicted_labels,correct_labels)
