import numpy as np
import random
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot

import numpy as np
import time
import sys




def minkowski_mat(x,Y,p=2.0):
    dist = (np.abs(x-Y)**p).sum(1)**(1.0/p)
    return dist


def kppv(x, data, k, n_classes, p=2):

    les_comptes = np.zeros(n_classes)
    distances = minkowski_mat(x, data[:, 1:])
    # on prend les indices jusqu'à k du vecteurs de distance trié
    ind_voisins = np.argsort(distances)[:k]
    cl_voisins = data[ind_voisins, 0].astype(int)
    for j in range(min(k, data.shape[0])):
        les_comptes[cl_voisins[j]-1] += 1
    return np.argmax(les_comptes)+1


def conf_matrix(etiquettesTest, etiquettesPred):
    n_classes = int(max(etiquettesTest+1))
    matrix = np.zeros((n_classes, n_classes))

    for (test, pred) in zip(etiquettesTest, etiquettesPred):
        matrix[int(test), int(pred)] += 1

    return matrix






UTD_dataset = np.loadtxt(open("UTD_MHAD_Labeled_Descriptors.csv", "rb"), delimiter=",", skiprows=0)


# Commenter pour avoir des resultats non-deterministes
random.seed(5)

##########################  Make validation set with 1/3 of training set ########################

n_train = int(UTD_dataset.shape[1]*0.66)

# Determiner au hasard des indices pour les exemples d'entrainement et de test
inds = list(range(UTD_dataset.shape[1]))



random.shuffle(inds)
knn_train_for_val_inds = inds[:n_train]
knn_valid_inds = inds[n_train:]



TRAIN_labled_set_for_valid = UTD_dataset[:,knn_train_for_val_inds].T
test_labled_set_for_valid = UTD_dataset[:,knn_valid_inds].T




train_labels = TRAIN_labled_set_for_valid[:,0]

train_descriptors = TRAIN_labled_set_for_valid[:,1:]


test_labels = test_labled_set_for_valid[:,0]

test_descriptors = test_labled_set_for_valid[:,1:]

test_predictions = np.zeros(test_labled_set_for_valid.shape[0])
#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                         K-PPV CLASSIFIER                                          #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################




K = 1

ks = range(1, K + 1)

results = np.zeros(len(ks))
for i, k in enumerate(ks):

    test_predictions = np.zeros(test_labled_set_for_valid.shape[0])

    for j in range(test_labled_set_for_valid.shape[0]):
        test_predictions[j] = kppv(test_labled_set_for_valid[j, 1:], TRAIN_labled_set_for_valid, 1, 27)

    results[i] = (1.0 - np.equal(test_labels, test_predictions)).mean() * 100.0

print("\n \n \n")
print("FIND THE BEST VALUE OF K ON VALIDATION SUB-SET")
print("Meilleur résultat avec k =", ks[np.argmin(results)])
print("Taux d'erreur sur l'ensemble de test", np.min(results))




# trythis = kppv(test_descriptors, TRAIN_labled_set_for_valid, 1, 27)


conf_mat_knn = conf_matrix(test_labels, test_predictions)

error_rate = (1.0 - np.equal(test_labels, test_predictions)).mean() * 100.0



print(conf_mat_knn)

print(error_rate)



#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                           SVM CLASSIFIER                                          #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################



lin_spa = np.arange(0.1, 0.3, 0.02)

#Define parameters
gamma = 1
c = 1



error = np.zeros((len(lin_spa), len(lin_spa)))


for gamma in lin_spa:

    j = 0

    for c in lin_spa:

        #Initialize model
        rbf_svc = svm.SVC(C=c*150, kernel='rbf', gamma = gamma/25)

        rbf_svc.fit(train_descriptors, train_labels)

        print("\n \n \n")
        print("SVM confusion matrix:")

        test_predictions = rbf_svc.predict(test_descriptors)

        error[i][j] = (1.0 - np.equal(test_predictions, test_labels)).mean() * 100.0

        j += 1
    i += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


X, Y = np.meshgrid(lin_spa/25, lin_spa*150)


ax.plot_surface(X, Y, error, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
    linewidth=0, antialiased=False)

ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_zlabel('Taux erreur')


plt.savefig('SVM_hyperparam_optimization.png', dpi=100)
plt.show()






bob = 0