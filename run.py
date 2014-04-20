import dataImport
import DL
import misc
import Yarowsky
import perceptron_dropout_averaging as ped
import calc_error
import string
import operator
#from pylab import *
import itertools


reload(Yarowsky)
data = 'namedentity'

train,test,gold,nLabels,rules = dataImport.getData(data)

print "gold length:",len(gold)

# Label training data based on initial seed rules
labels = DL.label(train,rules,nLabels)
train123=[]
labels123=[]
train_unlabelled=[]
for features, label in zip(train, labels):
	if label==1 or label==2 or label==3:
		train123.append(features)
		labels123.append(label)
	else:
		train_unlabelled.append(features)



# n=len(labels123)
#print labels123[0:n]
#print train123[0:n]

# print "NUM EXAMPLES TO TRAIN ON: ", n

# '''TRAIN A SINGLE PERCEPTRON'''
# perceptron = perceptron_enssemble_dropout.perceptron()
# #perceptron.print_training_data(train123[0:n],labels123[0:n])
# perceptron.train(train123[0:n],labels123[0:n])

# print "Perceptron trained on ", n, " examples"
# print "training error: ", calc_error.calc_error(train123[0:n],perceptron,labels123[0:n])
# print "test error: ", calc_error.calc_error(test,perceptron,gold)

# '''TRAIN AN ENSSEMBLE OF PERCEPTRONS'''
# perceptrons = perceptron_enssemble_dropout.perceptron_enssemble(5)
# #perceptron.print_training_data(train123[0:n],labels123[0:n])
# perceptrons.train(train123[0:n],labels123[0:n])

# print "Perceptron Enssemble trained on ", n, " examples"
# print "training error: ", calc_error.calc_error(train123[0:n],perceptrons,labels123[0:n])
# print "test error: ", calc_error.calc_error(test,perceptrons,gold)

# def print_labels_and_feats(labels,list_of_sets_of_feats):
# 	print string.ljust("label",6)+string.ljust("features",6)
# 	for i in xrange(len(labels)):
# 		l=labels[i]
# 		feats=list_of_sets_of_feats[i]
# 		print string.ljust(`l`,6)+string.ljust(`feats`,6)

# def print_labels_feats_and_confidence(labels_with_confidence,list_of_sets_of_feats):
# 	print string.ljust("label",6)+string.ljust("confidence",6)+string.ljust("features",6)
# 	for i in xrange(len(labels_with_confidence)):
# 		l,c=labels_with_confidence[i]
# 		feats=list_of_sets_of_feats[i]
# 		print string.ljust(`l`,6)+string.ljust(`c`,6)+string.ljust(`feats`,6)
	
#train_unlabelled=train_unlabelled[1:25]#DEBUG

def do_semi_supervised_iteration(num_to_add, classifier, labeled_features, labels, unlabelled_features, averaged):
	labels_and_confidence=classifier.predict_labels_with_confidence(unlabelled_features,averaged)
	n=len(unlabelled_features)
	assert len(labels_and_confidence)==n 
	# def check_consistency_of_predictions():
	# 	for lc,feats in itertools.izip(labels_and_confidence,unlabelled_features):
	# 		assert classifier.predict_label_with_confidence(feats)==lc 
	# check_consistency_of_predictions()
	# def print_ilc(ilcf):
	# 	for i,l,c,f in ilcf[:10]:
	# 		print "i,l,c, pl: ", i,l,c,classifier.predict_label_with_confidence(f,averaged)
	# 	print "..."
	# 	for i,l,c,f in ilcf[-10:]:
	# 		print "i,l,c, pl: ", i,l,c,classifier.predict_label_with_confidence(f,averaged)
	ilcf=[]
	for i in xrange(n):
		l,c=labels_and_confidence[i]
		f=unlabelled_features[i]
		# if(classifier.predict_label_with_confidence(f,averaged)!=(l,c)): #DEBUG
		# 	print "i,l,c,pl,pc: ",i,l,c,classifier.predict_label_with_confidence(f)

		ilcf.append((i,l,c,f))
	# def check_consistency(ilcf):
	# 	for i,l,c,f in ilcf:
	# 		if (classifier.predict_label_with_confidence(f)!=(l,c)):
	# 			print "i,l,c,pl,pc: ",i,l,c,classifier.predict_label_with_confidence(f)
	# 			assert False #DEBUG
	# check_consistency(ilcf)#DEBUG

	#print_ilc(ilcf) #DEBUG
	ilcf=sorted(ilcf, reverse=True,key=operator.itemgetter(2))
	# check_consistency(ilcf)#DEBUG

	#print_ilc(ilcf) #DEBUG
	count=0
	for i,l,c,f in ilcf[0:num_to_add]:
		labels.append(l)
		labeled_features.append(f)
		count+=1
		if(classifier.predict_label(f,averaged)!=l):
			print "i,l,c,classifier.predict_label(f),count,f"
			print i,l,c,classifier.predict_label(f),count,f
			assert(False) #DEBUG
		#print "added: ", l, c, f
	new_unlabelled = [f for i,l,c,f in ilcf[num_to_add:]]
	return new_unlabelled

def do_semi_supervised_learning(num_iterations,num_to_add_each_iteration, perceptron, labeled_features, labels, unlabelled_features,iterations):
	test_errors=[]	
	test_errors_averaged=[]
	for i in xrange(num_iterations):
		print "\n"		
		print "Starting iteration ", i, "of ", num_iterations,"runtype = ", run_type,
		perceptron.train(labeled_features,labels,iterations/3) #to get close to equilibrium first
		perceptron.train(labeled_features,labels,iterations)
		def get_errors(feats,labels,averaged):
			predicted_labels=perceptron.predict_labels(feats,averaged)
			test_num_errors, test_error=calc_error.error(predicted_labels,labels)
			return test_num_errors, test_error
		test_num_errors, test_error = get_errors(test,gold,False)
		test_errors.append(test_error)
		test_num_errors_averaged, test_error_averaged = get_errors(test,gold,True)
		test_errors_averaged.append(test_error_averaged)
		if i%5==1:
			print "training error: ", get_errors(labeled_features,labels,False)
			print "training error, averaged perceptron: ", get_errors(labeled_features,labels,True)
			print "iteration ", i, " test error: ", test_error,"num errors:",test_num_errors
			print "labelled and unlabelled feature vector lengths: ", len(labeled_features), len(unlabelled_features)
		print "iteration ", i, " test error: ", test_error_averaged,"num errors:",test_num_errors_averaged
		if test_error>0.65:
			print "error condition met\n"
			break
		if i>2:
			print "i>2"
			print "variation in last 3 iterations: ",max(test_errors[-3:])-min(test_errors[-3:])
			if max(test_errors[-3:])-min(test_errors[-3:])<0.0000001:
				print "Not Changing\n"
				break
			if (sum(test_errors[-3:])/3>min(test_errors)+0.15):
				print "relativeerror condition met\n"
				break
			if (sum(test_errors[-3:])/3>60):
				print "error condition met\n"
				break

		unlabelled_features=do_semi_supervised_iteration(num_to_add_each_iteration,
			perceptron, labeled_features, labels, unlabelled_features, True)
		
	return test_errors,test_errors_averaged

# num_perceptrons_in_enssemble=[1,3,10]
# # drop_out_rate=[0,.2,.4]
# num_perceptrons_in_enssemble=[1]
# #drop_out_rate=[0,.2,.4,.6,.8,.9]
# drop_out_rate=[.5]
# num_to_add=20000
# max_iterations_if_0=3
# #num_to_add_each_iteration=[4000,1000,250,0]
# num_to_add_each_iteration=[1000]
# repetitions=10

run_type="standard"

def set_runtime_parameters(run_type_in):
	global run_type
	print run_type
	run_type=run_type_in
	global drop_out_rate
	global num_to_add
	global max_iterations_if_0
	global num_to_add_each_iteration
	global repetitions
	global iterations
	global train123
	global labels123
	global train_unlabelled
	if run_type=="dropout":
		drop_out_rate=[0,.2,.4,.6,.8]
		num_to_add=60000
		max_iterations_if_0=3
		num_to_add_each_iteration=[4000]
		repetitions=1
		iterations=[3]
	if run_type=="iterations":
		drop_out_rate=[.4]
		num_to_add=40000
		max_iterations_if_0=3
		num_to_add_each_iteration=[2000]
		repetitions=1
		iterations=[.5,1,2,3,4,8]
	elif run_type=="cautiousness":
		drop_out_rate=[.5] 
		num_to_add=60000
		max_iterations_if_0=10
		num_to_add_each_iteration=[8000, 2000, 500, 125,0]
		repetitions=1
		iterations=[5]
	elif run_type=="standard":
		drop_out_rate=[.5]
		num_to_add=40000
		max_iterations_if_0=3
		num_to_add_each_iteration=[1000]
		repetitions=1
		iterations=[3]
	elif run_type=="quick":
		drop_out_rate=[.5] 
		num_to_add=3000
		max_iterations_if_0=3
		num_to_add_each_iteration=[1000]
		repetitions=1
		iterations=[.5]
		train123=train123[1:2000]#DEBUG
		labels123=labels123[1:2000]
		train_unlabelled=train_unlabelled[1:10000]


# perceptron_enssembles = [perceptron_enssemble.perceptron_enssemble(x) for x in num_perceptrons_in_enssemble]
# for pe in perceptron_enssembles:
import time
def do_all_runs():
	file_to_write_to=run_type+str(time.time())
	with open(file_to_write_to,'w') as f:
		f.write("Results:\n")
	for i in xrange(repetitions):
		for d in drop_out_rate:
			for a in num_to_add_each_iteration:
				for its in iterations:
					pe=ped.perceptron(d)
					copy_of_labelled_features=[x for x in train123]
					copy_of_labels=[x for x in labels123]
					copy_of_unlabelled_features=[x for x in train_unlabelled]
					pe.train(train123,labels123,its)
					print '\n Semi-supervised learning with a perceptron.	Feature dropout rate:',d,'and adding', a, 'each iteration'
					num_iterations=int(num_to_add/(a+1))+1
					if a==0:
						num_iterations=max_iterations_if_0
					test_errors, test_errors_averaged=\
						do_semi_supervised_learning(num_iterations,a, pe, 
							copy_of_labelled_features, copy_of_labels, 
							copy_of_unlabelled_features,its)
					with open(file_to_write_to,'a') as f:
						f.write("Not Averaged:\tdropout:"+str(d)+"\tadded_per_iteration"+str(a)+"\titerations:"+str(its))
						for err in test_errors:
							f.write("\t"+str(err))
						f.write("\n\n")

						f.write("Averaged:\tdropout:"+str(d)+"\tadded_per_iteration"+str(a)+"\titerations:"+str(its))
						f.write("na: "+str(d)+" "+str(a))
						for err in test_errors_averaged:
							f.write("\t"+str(err))
						f.write("\n")
set_runtime_parameters("dropout")
do_all_runs()
# set_runtime_parameters("iterations")
# do_all_runs()
# set_runtime_parameters("cautiousness")
# do_all_runs()




# import cProfile
# cProfile.runctx("do_all_runs()",globals(),locals())

# #perceptron.print_training_data(train123[0:n],labels123[0:n])
# perceptrons.train(train123[0:n],labels123[0:n])

# num_to_add_each_iteration=100
# labels_and_confidence=perceptron.predict_labels_with_confidence(train_unlabelled)

# def print_data():
# 	print "\nLABELLED DATA:"
# 	print_labels_and_feats(labels123,train123)
# 	print "\nUNLABELLED DATA:"
# 	print_labels_feats_and_confidence(labels_and_confidence,train_unlabelled)
# #print_data()

# n=len(labels_and_confidence)
# assert(n==len(train_unlabelled))


# '''Semi Supervised Learning'''
# ilcf=[]
# for i in xrange(n):
# 	l,c=labels_and_confidence[i]
# 	f=train_unlabelled[i]
# 	ilcf.append((i,l,c,f))
# ilcf=sorted(ilcf, reverse=True,key=operator.itemgetter(2))
# print "\nilcf"
# print ilcf #DEBUG
# for i,l,c,f in ilcf[0:num_to_add_each_iteration]:
# 	labels123.append(l)
# 	train123.append(f)
# 	#print "added: ", l, c, f
# train_unlabelled = [f for i,l,c,f in ilcf[num_to_add_each_iteration:]]

# print "\nLABELLED DATA - AFTER ADDING FROM UNLABELLED SET:"
# print_labels_and_feats(labels123,train123)

# labels_and_confidence=perceptron.predict_labels_with_confidence(train_unlabelled)
# print "\nUNLABELLED DATA:"
# print_labels_feats_and_confidence(labels_and_confidence,train_unlabelled)





