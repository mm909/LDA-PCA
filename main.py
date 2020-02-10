import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dims = 2
n_components = 1

def model():
    # data = np.genfromtxt( "inputOverlap0.txt", delimiter = ',' )
    # data = np.genfromtxt( "inputOverlap.txt", delimiter = ',' )
    data = np.genfromtxt( "inputCluster.txt", delimiter = ',' )

    labels = data[:,dims]
    data = data[:,:dims]

    fig, axs = plt.subplots(3, 1)

    if dims == 3:
        axs[0].scatter(data[:,0], data[:,1], data[:,2], c = labels)
    if dims == 2:
        axs[0].scatter(data[:,0], data[:,1], c = labels)
    axs[0].title.set_text('Original Data')

    pca = PCA(n_components)
    pca.fit(data)
    pcaData = pca.transform(data)

    explained = np.cumsum(pca.explained_variance_ratio_)
    formatedExplained = math.floor(explained[len(explained) - 1] * 10000) / 100
    print("PCA: ", n_components, "Componets", str(formatedExplained) + "% of the variance is explained.")

    if dims == 3:
        axs[1].scatter(pcaData[:,0], pcaData[:,1], c = labels)
    if dims == 2:
        axs[1].scatter(pcaData[:,0], np.zeros(len(pcaData)), c = labels)

    axs[1].title.set_text('PCA')

    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(data, labels)

    if dims == 3:
        axs[2].scatter(X_lda[:,0], X_lda[:,1], c = labels)
    if dims == 2:
        axs[2].scatter(X_lda[:,0], np.zeros(len(X_lda)), c = labels)

    axs[2].title.set_text('LDA')

    plt.show()


model()
