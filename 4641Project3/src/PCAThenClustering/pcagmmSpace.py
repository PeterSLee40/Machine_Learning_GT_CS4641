import itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
from sklearn import mixture
from kneed import KneeLocator
#takes in breast data


X_train,y_training  = np.load('../SpaceData/X_train.npy'), np.load('../SpaceData/y_train.npy')
X_test, y_test = np.load('../SpaceData/X_test.npy'), np.load('../SpaceData/y_test.npy')

pca = decomp.PCA(n_components=len(X_train[0]))
pca.fit(X_train)   
print (pca.explained_variance_ratio_ )
#pick the top 80% that gets most the data.
#plot cumulative pca
pcaVarRatioArray = pca.explained_variance_ratio_
cumVar = list()
for i in range(len(pca.explained_variance_ratio_)):
    total = pcaVarRatioArray[i]
    if (len(cumVar) > 0):
        total += cumVar[-1]
    cumVar.append(total)
x = range(len(pca.explained_variance_ratio_))
#plt.plot(x,cumVar)

knee = KneeLocator(x,cumVar,curve='convex', direction='increasing').knee
#plt.axvline(knee, c = "r")
print("optimal knee is a k size of:"  , knee)
pca = decomp.PCA(n_components=knee)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)

lowest_bic = np.infty
bic = []
n_components_range = range(1, 25)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X_train_PCA)
        #gets the bic of the test set,=
        bic.append(gmm.bic(X_test_PCA))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min()*1.1 - bic.max()*.1, bic.max()*1.1])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner

#https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html