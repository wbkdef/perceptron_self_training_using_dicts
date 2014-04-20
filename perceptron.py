from __future__ import division
import itertools
import misc
import string
import random
import math
import classifier

def arg_top_two(list_):
    """TESTED - returns the indices of the max and the second biggest element"""
    list_copy = [x for x in list_]
    max_arg=misc.argmax(list_copy)
    list_copy[max_arg]=min(list_copy)-1
    second_biggest=misc.argmax(list_copy)
    return max_arg, second_biggest

def dict_add_to(v,vect_to_add):
    """Modifies v so that
    1) If a key originally occurs in only one of 'v' and 'vect_to_add'
    then that will be the value in v for that key.
    2) If a key originally occurs in both 'v' and 'vect_to_add'
    then the final value for key in v will be the sum of the values 
    from both dictionaries"""
    for key,value in vect_to_add.iteritems():
        if key not in v:
            v[key]=value
        else:
            v[key]+=value

def dict_add_key_value_to(d,key,value):
    """Modifies v so that
    1) If a key originally occurs in only one of 'v' and 'vect_to_add'
    then that will be the value in v for that key.
    2) If a key originally occurs in both 'v' and 'vect_to_add'
    then the final value for key in v will be the sum of the values 
    from both dictionaries"""
    if key not in d:
        d[key]=value
    else:
        d[key]+=value

def sum_values(my_dict,keys):
    return sum([my_dict[key] for key in keys if key in my_dict])


class perceptron_vector: #Has been tested
    """A class to manage a single perceptron vector

    This class will manage the updating of perceptron vectors
    and keep a cumulative sum (for making predictions using 
    the averaged perceptron) """
    def __init__(self):
        self.w=dict() #A dictionary with keys=features and values=(time of last update, vector value)
        self.t=dict() #The time that self.w was last updated for each feature
        self.w_sum=dict() #A dictionary with only values
        self.w_sum_up_to_date=False
        self.latest_time=0
    def update_w_sum_and_time(self,feats,time):
        for key in feats:
            if key not in self.w: #No need to update self.w_sum
                assert(key not in self.w_sum)
                assert(key not in self.t)
                self.t[key]=time
            else:
                delta_t=time-self.t[key]
                dict_add_key_value_to(self.w_sum,key,self.w[key]*delta_t)
                self.t[key]=time
    def update_all_w_sum_and_time(self,time):
        self.update_w_sum_and_time(self.w.keys(),time)
        self.w_sum_up_to_date=True
    def update_w(self,feats,value,time):
        self.w_sum_up_to_date=False
        if time<=self.latest_time:
            print "time, self.latest_time", time, self.latest_time
            assert(time>self.latest_time)
        self.latest_time=time
        self.update_w_sum_and_time(feats,time)
        for key in feats:
            self.t[key]=time
            if key not in self.w:
                self.w[key]=value
            else:
                self.w[key]+=value
    def dot_product(self,feats):
        return sum_values(self.w,feats)
    def dot_product_averaged(self,feats):
        assert(self.w_sum_up_to_date==True)
        return sum_values(self.w_sum,feats)
    def set_t_to_0_and_w_sum_to_empty(self):
        self.w_sum=dict()
        for key in self.t.keys():
            self.t[key]=0
        self.latest_time=0
    def print_self(self):
        print "Printing perceptron vector"
        print "latest_time: ", self.latest_time
        print "feature\ttime\tw\tw_sum"
        for key in self.w.keys():
            print key,"\t",self.t[key],"\t", self.w[key],"\t",
            if key in self.w_sum:
                print self.w_sum[key]
            else:
                print "-"

class perceptron(classifier.Classifier):
    """A Class for training a perceptron and make predictions using it."""
    def __init__(self,training_passes,drop_out_rate=0,averaged=False):
        # self.ws = [perceptron_vector() for x in xrange(3)]
        self.ws = None #Perceptron Vectors - to be initialized with training!
        self.keys=set() #record the keys0    
        assert(drop_out_rate>=0 and drop_out_rate<=1)
        assert isinstance(averaged,bool)
        assert 100>training_passes>0 
        self.training_passes=training_passes
        self.drop_out_rate=drop_out_rate #For Training Only
        self.averaged=averaged #For predictions only, not for training
    def __str__(self):
        s=""
        if averaged: 
            s+="Averaged perceptron"
        else:
            s+="Perceptron"

        s+=", dr=", str(self.drop_out_rate)
        s+=", training_passes=", str(self.training_passes)

    def __dot_products__(self,feats,averaged):
        if averaged:
            return [p.dot_product_averaged(feats) for l,p in self.ws.iteritems()]
        return [p.dot_product(feats) for l,p in self.ws.iteritems()]        
    def __predict_label_with_confidence__(self,x,averaged):
        """does w dot x and returns this as the confidence.  If the dot product is +ve it returns label 2, else label 1"""
        dot_products=self.__dot_products__(x, averaged)
        biggest, second_biggest = arg_top_two(dot_products)
        confidence=dot_products[biggest]-dot_products[second_biggest]
        assert(confidence>=0)
        return biggest+1, confidence        
    def predict_label_with_confidence(self,x):
        return self.__predict_label_with_confidence__(x,self.averaged)
    def __predict_label__(self,x,averaged):
        return self.__predict_label_with_confidence__(x,averaged)[0]
    def predict_label(self,x):
        return self.__predict_label__(x,self.averaged)[0]
    def update_w(self, feats, correct_label, predicted_label,time):
        """To be called when something was classified incorrectly to update the perceptron vector."""
        assert(correct_label!=predicted_label)
        correct_w=self.ws[correct_label]
        predicted_w=self.ws[predicted_label]
        for feat in feats:
            if feat not in self.keys: #Make them all be defined on the same features
                self.keys.add(feat) #record the keys in the order they were added.
        correct_w.update_w(feats,1,time)
        predicted_w.update_w(feats,-1,time)
    def __calculate_averaged_vectors__(self,time):
        for l,w in self.ws.iteritems():
            w.update_all_w_sum_and_time(time)
    def train(self, X, Y, X_unlabelled):
        print "\nstarting training",
        time=0
        self.ws = {l:perceptron_vector() for l in np.unique(Y)}
        n=len(X)
        assert n==len(Y)
        for j in xrange(int(self.training_passes*n)):
            ind=random.randint(0,n-1)
            y=Y[ind]
            assert y!=0: #this will just be points that neither should train on.
            assert(y in [1,2,3]) #For the first application, this should be the case
            time+=1                
            if time<2 or time%10000==0:
                print time,", ",
            if self.drop_out_rate>0:
                feats=set([])
                for f in X[ind]:
                    if(random.random()>self.drop_out_rate):
                        feats.add(f)
                #feats=random.rand(feats,max(1,int(self.drop_out_rate*len(feats)+.5)))
            else:
                feats=X[ind]
            feats.add("const_term") #This never drops out.  Perhaps its updates should be smaller.
            #print ''
            #print y, ": ", feats
            predicted_label=self.__predict_label__(feats,False)
            if predicted_label==y:
                continue
            assert(predicted_label in [1,2,3])
            self.update_w(feats, y, predicted_label, time) #If they are different, update the predicted label
            #self.print_self()
        self.__calculate_averaged_vectors__(time)
    def print_self(self):
        print "Printing components of perceptron vectors"
        print string.ljust("feature",20),"\tw1\tw2\tw3"
        assert(self.ws[1].keys()==self.ws[2].keys())
        assert(self.ws[2].keys()==self.ws[3].keys())
        for feat in self.keys:
            print string.ljust(feat,20), "\t",self.ws[1][feat],"\t",self.ws[2][feat],"\t",self.ws[3][feat]
    def print_training_data(self, features, labels):
        print "Training Data:"
        for feats, label in itertools.izip(features, labels):
            print label, ": ", feats
        print ''







