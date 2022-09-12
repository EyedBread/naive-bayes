from collections import defaultdict
from email.policy import default
from pdb import post_mortem
from re import X
from xml.sax.handler import all_properties
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

data_path = 'processed.cleveland.data'
n_features = 14
n_patients = 303

def get_avg_of_features(data_path):
    avg = np.zeros(n_features, dtype=np.float32)
    missing = np.zeros(n_features, dtype=np.float32)
    with open(data_path, 'r') as file:
        next(file)
        for line in file.readlines():
            x = line.split(',')
            for index, val in enumerate(x):
                if val == '?':
                    missing[index] += 1
                else:
                    avg[index] += float(val)
    for i in range(n_features):
        avg[i] = avg[i] / (n_patients - missing[i])
    return avg

avg = get_avg_of_features(data_path)

def load_features(data_path):
    data = np.zeros([n_patients, n_features], dtype =np.float32)
    with open(data_path, 'r') as file:
        next(file)
        for count, line in enumerate(file.readlines()):
            x = line.split(',')
            for index, val in enumerate(x):
                if val == '?':
                    data[count, index] = avg[index] #imputing missing values with that columns average
                else:
                    data[count, index] = val
    return data

def model_tuner():
    """
    Fine tune the model by calculating the AUC score for several models, varying the smoothing factor and fit prior to find the best fit.
    smoothing factor = 1, fit prior = True
    """
    k = 5
    k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    smoothing_factor_option = [1,2,3,4,5,6]
    fit_prior_option = [True, False]
    auc_record = {}
    for train_indices, test_indices in k_fold.split(X,Y):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        for alpha in smoothing_factor_option:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train, Y_train)
                prediction_prob = clf.predict_proba(X_test)
                pos_prob = prediction_prob[:,1]
                auc = roc_auc_score(Y_test, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
    for smoothing, smoothing_record in auc_record.items():
        for fit_prior, auc in smoothing_record.items():
            print(f'    {smoothing}     {fit_prior}     {auc/k:.5f}')
    return

data = load_features(data_path)

#print(data)

X = np.delete(data, n_features-1, axis=1)
Y = data[:, n_features-1]

#print(X)
#Values greater or equal to 1 indicate heart disease
Y[Y >= 1] = 1
#print(Y)

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positive samples(disease), {n_neg} negative samples(no disease)')

#Evaluation of our classifier's performance, randomly split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#print(len(Y_train), len(Y_test))
model = MultinomialNB(alpha=1.0, fit_prior=True)
model.fit(X_train, Y_train)
prediction_prob = model.predict_proba(X_test)
#print(prediction_prob)
prediction = model.predict(X_test)
#print(prediction)
accuracy = model.score(X_test, Y_test)
print(f'The Accuracy is: {accuracy*100:.1f}%')
f1_score(Y_test, prediction, pos_label=1)
model_tuner()


