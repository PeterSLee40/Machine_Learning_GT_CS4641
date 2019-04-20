import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

#takes in breast data
X_train,y_train  = np.load('../BreastData/X_trainingSet.npy'), np.load('../BreastData/y_trainingSet.npy')
X_test, y_test = np.load('../BreastData/X_testSet.npy'), np.load('../BreastData/y_testSet.npy')
#edit this to desired levels.
n_components = 426


gmm = mixture.GaussianMixture(n_components)

X_train_gmm = gmm.fit_predict(X_train)
X_train_gmm = np.reshape(X_train_gmm,(np.size(X_train_gmm[:]),1))
X_test_gmm = gmm.predict(X_test)
X_test_gmm = np.reshape(X_test_gmm,(np.size(X_test_gmm[:]),1))

#gets the bic of the test set,# Plot the winner

#https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix  

errorTrain = []
errorValid = []
errorTraingmm = []
errorValidgmm = []
x = range(100)
for index in x:
    index = index + 1
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping = False,
           epsilon=1e-08, hidden_layer_sizes=(10, 10), learning_rate='constant',
           learning_rate_init=0.001, max_iter=index, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=False,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=1,
           warm_start=False)
    clfnn = clf
    clfnn.fit(X_train,y_train)
    hypothesis = clfnn.predict(X_test)
    errorValid.append(1 - np.mean(np.abs(hypothesis - y_test)))
    hypothesis = clfnn.predict(X_train)
    errorTrain.append(1 - np.mean(np.abs(hypothesis - y_train)))
    
    clf_gmm = clf
    Xcomp = np.concatenate((X_train, X_train_gmm/n_components*2-1), axis = 1)
    Xtest = np.concatenate((X_test, X_test_gmm/n_components*2-1), axis = 1)
    clf_gmm.fit(Xcomp, y_train)
    hypothesis_gmm = clf_gmm.predict(Xtest)
    errorValidgmm.append(1 - np.mean(np.abs(hypothesis_gmm - y_test)))
    hypothesis_gmm = clf_gmm.predict(Xcomp)
    errorTraingmm.append(1 - np.mean(np.abs(hypothesis_gmm - y_train)))
    
    
    
    
plt.figure(2)
plt.plot(x ,errorValid, label = "testing")
plt.plot(x ,errorTrain, label = "training")
plt.title('Training Performance vs. time')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.plot(x ,errorValidgmm, label = "testing gmm")
plt.plot(x ,errorTraingmm, label = "training gmm")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    
    
#input in the best time point.
#print(confusion_matrix(y_testSet, hypothesis))  
#print(classification_report(y_testSet, hypothesis))  


#model = clf
#learningCurve = skplt.estimators.plot_learning_curve(model, X_trainingSet, y_trainingSet, title='ANN Learning Curve', cv=None, shuffle=False, random_state=None, train_sizes=None, n_jobs=1, scoring=None, ax=None, figsize=None, title_fontsize='large', text_fontsize='medium')
