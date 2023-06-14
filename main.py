import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("general.csv")
dataset = dataset.iloc[:, 1:13]
print(dataset)

x = dataset.drop(["is_cheating"], axis=1)
y = dataset.is_cheating.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random.randint(80, 110))

'''Naive bayes     machine learning'''

nb = GaussianNB()
nb.fit(x_train, y_train)
preds_train = nb.predict(x_test)
print('predict ',y_test)
mc_naive_bayes = confusion_matrix(y_test, preds_train)
precision_global_naive_bayes = np.sum(mc_naive_bayes.diagonal()) / np.sum(mc_naive_bayes)
print('bayes   ',preds_train,precision_global_naive_bayes)
error_global_naive_bayes = 1 - precision_global_naive_bayes

'''SVM      machine learning'''

karnel = ["linear", "poly", "rbf", "sigmoid"]
gamma = ['scale', 'auto']
kar = ''
gamm = ''
maxscore = 0

for k in karnel:
    for g in gamma:
        classifier = SVC(kernel=k, random_state=1, C=1, gamma=g)
        classifier.fit(x_train, y_train)
        score = classifier.score(x_test, y_test)

        if score > maxscore:
            maxscore = score
            kar = k
            gamm = g

print('optimo    ',kar,"--", gamm)

classifier = SVC(kernel=kar, random_state=1, C=1, gamma=gamm)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
mc_SVM = confusion_matrix(y_test, y_pred)
precision_global_SVM = np.sum(mc_SVM.diagonal()) / np.sum(mc_SVM)
print('SVM     ',y_pred, precision_global_SVM)
error_global_SVM = 1 - precision_global_SVM



'''KMEANS     machine learning'''
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train, y_train)
kmp = kmeans.predict(x_test)
mc_KM = confusion_matrix(y_test, kmp)
precision_global_KM = np.sum(mc_KM.diagonal()) / np.sum(mc_KM)
print('kmeans  ',kmp, precision_global_KM)



'''KNN        machine learning'''

k_range = range(3, 20)
listK = []
scores = []
del scores[:]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))
    listK.append(k)
""" plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks(listK) """
#plt.show()

maxK = scores.index(max(scores)) + 1
print('maximos vacinos     ',maxK)

knn = KNeighborsClassifier(n_neighbors=maxK)
knn.fit(x_train, y_train)
prediccion = knn.predict(x_test)
mc_KNN = confusion_matrix(y_test, prediccion)
precision_global_KNN = np.sum(mc_KNN.diagonal()) / np.sum(mc_KNN)
print('knn     ',prediccion, precision_global_KNN)
error_global_KNN = 1 - precision_global_KNN



'''LOGISTIC REGRESION     modelo lineal'''
clf = LogisticRegression(random_state=0, max_iter= 700).fit(x_train, y_train)
lrp = clf.predict(x_test)
mc_LR = confusion_matrix(y_test, lrp)
precision_global_LR = np.sum(mc_LR.diagonal()) / np.sum(mc_LR)
print('RL      ',lrp, precision_global_LR)



'''Tree       machine learning'''
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(x_train, y_train)
prediccion_tree = tree.predict(x_test)
mc_AD = confusion_matrix(y_test, prediccion_tree)
precision_global_AD = np.sum(mc_AD.diagonal()) / np.sum(mc_AD)
print('Arbol   ',prediccion_tree, precision_global_AD)
error_global_AD = 1 - precision_global_AD



'''forest'''

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(x_train, y_train)
prediccion_rf = rf.predict(x_test)
mc_rf = confusion_matrix(y_test, prediccion_tree)
precision_global_rf = np.sum(mc_rf.diagonal()) / np.sum(mc_rf)
print('bosque  ',prediccion_rf, precision_global_rf)
error_global_rf = 1 - precision_global_rf

print('\n')

'''Prediccion Final '''

predicts = []
db0 = pd.read_csv('general.csv', sep=',', decimal='.')
db = db0.iloc[:, 1:]

score_global=0

for x in range(len(db)):
    del predicts[:]

    """ print('Sujeto: ',db0.iloc[x:x + 1, :1].values[0][0],'\n') """

    sujeto_x = db.iloc[x:x + 1, :].drop(["is_cheating"], axis=1)
    sujeto_y = db.iloc[x:x + 1, :].is_cheating.values

    """ print('Predict        ', sujeto_y[0]) """

    '''bayes'''

    predict = nb.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict bayes  ',predict[0]) """

    '''SVM'''

    predict = classifier.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict SVM    ', predict[0]) """

    '''kmeans'''

    predict = kmeans.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict kmeans ', predict[0]) """

    '''knn'''

    predict = knn.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict knn    ', predict[0]) """

    '''RL'''

    predict = clf.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict RL     ', predict[0]) """

    '''Arbol'''

    predict = tree.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict Arbol  ', predict[0]) """

    '''bosque'''

    predict = rf.predict(sujeto_x)
    predicts.append(predict[0])
    """ print('Predict bosque ', predict[0]) """

    cont = 0
    for x in predicts:
        if x>0.65:
            cont+=1

    is_cheating=0
    if cont>=4:
        is_cheating=1

    """ print('is_cheating    ',is_cheating,'\n') """

    if sujeto_y[0] == is_cheating:
        score_global+=1

score_global=score_global/len(db)

print("La presicion global es de ",score_global)