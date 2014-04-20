from __future__ import division

class Classifier:
  def __init__(self):
    assert False
  def train(self,X,Y,X_unlabelled):
    assert False
  def predict_label(self,x):
    assert False
  def predict_label_with_confidence(self,x):
    """return (label,confidence)"""
    assert False
  def predict_labels(self,X):
    """return [labels] (as a list)"""
    return [self.predict_label(x) for x in X]
  def predict_labels_with_confidences(self,X):
    """return ([labels],[confidences]) (as a tuple of lists)"""
    labels, confidences = [],[]
    for x in X:
      l,c=self.predict_label_with_confidence(x)
      labels.append(l)
      confidences.append(c)
    return labels, confidences

  def __str__(self): 
    """Return a short description that suffices to explain the 
    classifier being used and its parameters."""
    assert False



def create_synthetic_data(num_labels,num_feats,num_train_labelled,num_unlabelled,num_test,sparsity=3,skew=2,rand_seed=None):
  """Returns Synthetic Data in a dictionary with keys: 
     "X_train", "y_train", "X_unlabelled", "X_test", "y_test" """
  import random
  #results=[X_train,y_train,X_unlabelled,X_test,y_test]
  assert num_feats <= 26
  feats=set('abcdefghijklmnopqrstuvwxyz'[:num_feats])  
  labels=range(1,num_labels+1)
  assert sparsity>=skew
  if rand_seed != None:
    random.seed(rand_seed)

  feat_probs={}
  for f in feats:
    feat_probs[f]=random.random()**(sparsity-skew)
  feat_label_probs={l:{} for l in labels}
  for l in labels:
    for f in feats:
      feat_label_probs[l][f]=random.random()**skew*feat_probs[f]

  def generate_X_Y(n):
    Y=[random.randint(1,num_labels) for x in range(n)]
    X=[]
    for i in range(n):
      X.append(set())
      for f in feats:
        if random.random()<feat_label_probs[Y[i]][f]:
          X[-1].add(f)
    return X, Y
  
  data={}
  data["X_train"], data["y_train"] = generate_X_Y(num_train_labelled)
  data["X_unlabelled"], y = generate_X_Y(num_unlabelled)
  data["X_test"], data["y_test"] = generate_X_Y(num_train_labelled)

  return data

if __name__ == '__main__':
  import operator
  import misc
  data = create_synthetic_data(num_labels=3,num_feats=20,num_train_labelled=10,num_unlabelled=5,num_test=5)
  def print_Y_X(X,Y):
    ind = misc.sortedInd(Y)
    # [i[0] for i in sorted(enumerate(Y),key=operator.itemgetter(1))]
    assert len(X)==len(Y)
    for i in ind:
      print Y[i],":  ",
      for f in X[i]:
        print f,
      print
  print "Training_Data"
  print_Y_X(data["X_train"], data["y_train"])
  print "Test_Data"
  print_Y_X(data["X_test"], data["y_test"])
  print "Unlabelled_Data"
  print_Y_X(data["X_unlabelled"], [0]*len(data["X_unlabelled"]))