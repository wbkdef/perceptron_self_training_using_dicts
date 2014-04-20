from __future__ import division
import itertools
import misc
import string
import random
import math

def arg_top_two(list_):
    """TESTED - returns the indices of the max and the second biggest element"""
    list_copy = [x for x in list_]
    max_arg=misc.argmax(list_copy)
    list_copy[max_arg]=min(list_copy)-1
    second_biggest=misc.argmax(list_copy)
    return max_arg, second_biggest

class perceptron_enssemble:
    """A class meant to encapsulate an enssemble of perceptrons and interface like a single perceptron"""
    def __init__(self, num_perceptrons,drop_out_rate=0):
        self.n=num_perceptrons
        self.perceptrons=[perceptron(drop_out_rate) for i in xrange(self.n)]
    def dot_products_sum(self,feats):
        dps=[0,0,0]
        for p in self.perceptrons:
            dp=p.dot_products(feats)
            dps=[x+y for x,y in zip(dps,dp)]
        return dps
    def predict_label_with_confidence(self,feats):
        """does w dot feats and returns this as the confidence.  If the dot product is +ve it returns label 2, else label 1"""
        dot_products=self.dot_products_sum(feats)
        biggest, second_biggest = arg_top_two(dot_products)
        confidence=dot_products[biggest]-dot_products[second_biggest]
        assert(confidence>=0)
        return biggest+1, confidence
    def predict_labels_with_confidence(self,list_of_sets_of_feats):
        return [self.predict_label_with_confidence(feats) for feats in list_of_sets_of_feats]
    def predict_label(self,feats):
        label, confidence = self.predict_label_with_confidence(feats)
        return label
    def predict_labels(self,list_of_sets_of_feats):
        return [self.predict_label(feats) for feats in list_of_sets_of_feats]
    def train(self, features, labels):
        for perceptron in self.perceptrons:
            perceptron.train(features,labels)

    def print_self(self):
        def print_ws(ws):
            print string.ljust("feature",20),"\tw1\tw2\tw3"
            assert(ws[0].keys()==ws[1].keys())
            assert(ws[1].keys()==ws[2].keys())
            for feat in ws[0].keys():
                print string.ljust(feat,20), "\t",ws[0][feat],"\t",ws[1][feat],"\t",ws[2][feat]
        print "Printing components of sum of perceptron's vectors"
        print_ws(self.get_ws_sum())
        print "Printing components of first perceptron's vectors"
        print_ws(self.perceptrons[0].get_ws())

    def get_ws_sum(self):
        def merge_ws(w_to_merge_into,w_to_merge):
            for key in w_to_merge:
                if key not in w_to_merge_into:
                    w_to_merge_into[key]=w_to_merge[key]
                else:
                    w_to_merge_into[key]+=w_to_merge[key]
        ws=[{},{},{}]
        for p in self.perceptrons:
            for w_to_merge_into, w_to_merge in zip(ws,p.get_ws()):
                merge_ws(w_to_merge_into,w_to_merge)
        return ws

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
    #print "key",key,d
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

class perceptron:
    """A Class for training a perceptron and make predictions using it."""
    def __init__(self,drop_out_rate=0):
        self.ws = [perceptron_vector() for x in xrange(3)]
        self.keys=set() #record the keys0    
        self.drop_out_rate=drop_out_rate #For Training Only
    def dot_products(self,feats,averaged):
        if averaged:
            return [p.dot_product_averaged(feats) for p in self.ws]
        return [p.dot_product(feats) for p in self.ws]        
    def predict_label_with_confidence(self,feats,averaged):
        """does w dot feats and returns this as the confidence.  If the dot product is +ve it returns label 2, else label 1"""
        dot_products=self.dot_products(feats,averaged)
        biggest, second_biggest = arg_top_two(dot_products)
        confidence=dot_products[biggest]-dot_products[second_biggest]
        assert(confidence>=0)
        return biggest+1, confidence        
    def predict_labels_with_confidence(self,list_of_sets_of_feats,averaged):
        return [self.predict_label_with_confidence(feats,averaged) for feats in list_of_sets_of_feats]
    def predict_label(self,feats,averaged):
        label, confidence = self.predict_label_with_confidence(feats,averaged)
        return label
    def predict_labels(self,list_of_sets_of_feats,averaged):
        return [self.predict_label(feats,averaged) for feats in list_of_sets_of_feats]
    def update_w(self, feats, correct_label, predicted_label,time):
        """To be called when something was classified incorrectly to update the perceptron vector."""
        assert(correct_label!=predicted_label)
        correct_w=self.ws[correct_label-1]
        predicted_w=self.ws[predicted_label-1]
        for feat in feats:
            if feat not in self.keys: #Make them all be defined on the same features
                self.keys.add(feat) #record the keys in the order they were added.
        correct_w.update_w(feats,1,time)
        predicted_w.update_w(feats,-1,time)
    # def get_ws(self): !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     return [w for w in self.ws] #Make a copy and return it.
    def calculate_averaged_vectors(self,time):
        for w in self.ws:
            w.update_all_w_sum_and_time(time)
    def train(self, sets_of_features, labels, iterations):
        print "\nstarting training",
        time=0
        assert(self.drop_out_rate>=0 and self.drop_out_rate<=1)
        for w in self.ws:
            w.set_t_to_0_and_w_sum_to_empty()
        n=len(sets_of_features)
        for j in xrange(int(n*iterations*math.sqrt(1/(1-self.drop_out_rate)))):
            ind=random.randint(0,n-1)
            label=labels[ind]
            if label==0: #this will just be points that neither should train on.
                continue
            assert(label in [1,2,3])
            time+=1                
            if time<3 or time%10000==0:
                print time,", ",
            if self.drop_out_rate>0:
                feats=set([])
                for f in sets_of_features[ind]:
                    if(random.random()>self.drop_out_rate):
                        feats.add(f)
                #feats=random.rand(feats,max(1,int(self.drop_out_rate*len(feats)+.5)))
            else:
                feats=sets_of_features[ind]
            feats.add("const_term") #This never drops out.  Perhaps its updates should be smaller.
            #print ''
            #print label, ": ", feats
            predicted_label=self.predict_label(feats,False)
            if predicted_label==label:
                continue
            assert(predicted_label in [1,2,3])
            self.update_w(feats, label, predicted_label,time) #If they are different, update the predicted label
            #self.print_self()
        self.calculate_averaged_vectors(time)
    def print_self(self):
        print "Printing components of perceptron vectors"
        print string.ljust("feature",20),"\tw1\tw2\tw3"
        assert(self.ws[0].keys()==self.ws[1].keys())
        assert(self.ws[1].keys()==self.ws[2].keys())
        for feat in self.keys:
            print string.ljust(feat,20), "\t",self.ws[0][feat],"\t",self.ws[1][feat],"\t",self.ws[2][feat]

    def print_training_data(self, features, labels):
        print "Training Data:"
        for feats, label in itertools.izip(features, labels):
            print label, ": ", feats
        print ''







