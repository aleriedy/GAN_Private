from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim


from dp_autoencoder import Autoencoder
from dp_wgan import Generator


# Deterministic output
torch.manual_seed(1234)
np.random.seed(1234)


def isolate_column(df, column):
    """Splits a given dataframe into a column and the rest of the dataframe."""
    return df.drop(columns=[column]).values.astype('float32'), df[column].values.astype('float32')


def get_prediction_score(X_train, y_train, X_test, y_test, Model, score):
    """Trains a model on training data and returns the quality of the predictions given test data and expected output."""
    if len(np.unique(y_train)) == 0 or len(np.unique(y_test)) == 0:
        raise

    # Train model on synthetic data, see how well it predicts holdout real data
    clf = Model()
    try:
        clf.fit(X_train, y_train)
        prediction = clf.predict_proba(X_test)[:,1]
    except:
        prediction = np.full(y_test.shape, y_train[0])

    """
    except ValueError as e:
        # Only one label present, so simply predict all examples as that label
        prediction = np.full(y_test.shape, val)
    """

    return score(y_test, prediction)

def balance(data, label):
    '''balance a data according to its label'''
    index_0 = np.where(label == 0)[0] # index of label that is equal to 0
    index_1 = np.where(label == 1)[0]
    data_0 =  np.array([data[i] for i in index_0]) # index of data that with label 0
    data_1 = np.array([data[i] for i in index_1])
    if len(index_0) > len(index_1):
        data_0_disc, temp_remain = train_test_split(data_0, test_size=len(data_1), random_state=0)
        data = np.concatenate((temp_remain, data_1), axis=0)
    elif len(index_0) < len(index_1):
        data_1_disc, temp_remain = train_test_split(data_1, test_size=len(data_0), random_state=0)
        data = np.concatenate((data_0, temp_remain), axis=0)
    else:
        return data, label

    label_new = [0]*(len(data)//2)
    label_new.extend([1]*(len(data)//2))

    return data, np.array(label_new)

def feature_prediction_evaluation(
        train,
        test,
        synthetic,
        Model=lambda: LogisticRegression(solver='lbfgs', max_iter=300),
        score=f1_score,
        training_proportion=0.8,
        plot=False
    ):
  
    if set(train.columns) != set(synthetic.columns):
        raise Exception('Columns of given datasets are not identical.')

    real_classifier_scores, synthetic_classifier_scores = [], []
    for i, column in enumerate(train.columns):
        X_train_real, y_train_real = isolate_column(train, column)
        X_train_synthetic, y_train_synthetic = isolate_column(synthetic, column)
        X_test, y_test = isolate_column(test, column)

    
        try:
            real_score = get_prediction_score(X_train_real, y_train_real, X_test, y_test, Model, score)
            synthetic_score = get_prediction_score(X_train_synthetic, y_train_synthetic, X_test, y_test, Model, score)

            real_classifier_scores.append(real_score)
            synthetic_classifier_scores.append(synthetic_score)

            print(i, real_score, synthetic_score)
        except Exception as e:
            print(e)
            print(i, 'Failed.')

    if plot:
        plt.scatter(real_classifier_scores, synthetic_classifier_scores, s=2, c='blue')
        plt.title('')
        plt.xlabel('Real Data')
        plt.ylabel('Generated Data')
        plt.axis((0., 1., 0., 1.))
        plt.plot((0, 1), (0, 1))
        plt.savefig('dw-pred.png')
        pd.DataFrame(data={'real': real_classifier_scores, 'synthetic': synthetic_classifier_scores}).to_csv('dw-pred.csv')

    return sum(map(lambda pair: (pair[0] - pair[1]) ** 2, zip(real_classifier_scores, synthetic_classifier_scores)))


def feature_probabilities_evaluation(real, synthetic, plot=False):
    if set(real.columns) != set(synthetic.columns):
        raise Exception('Columns of given datasets are not identical.')

    props = pd.DataFrame(columns=['real', 'synthetic'], index=real.columns)

    for column in real.columns:
        _, counts_real = np.unique(real[column], return_counts=True)
        _, counts_synthetic = np.unique(synthetic[column], return_counts=True)

        props.at[column, 'real'] = (0 if len(counts_real) <= 1 else counts_real[1]) / len(real[column])
        props.at[column, 'synthetic'] = (0 if len(counts_synthetic) <= 1 else counts_synthetic[1]) / len(synthetic[column])

    if plot:
        plt.scatter(props['real'], props['synthetic'], s=10, c='blue')
        plt.title('')
        plt.xlabel('Real Data')
        plt.ylabel('Generated Data')
        plt.axis((-0.1, 1.1, -0.1, 1.1))
        plt.plot((0, 1), (0, 1))
        plt.savefig('dw-prob.png')
        props.to_csv('dw-prob.csv')

    return sum(map(lambda pair: (pair[0] - pair[1]) ** 2, zip(props['real'], props['synthetic'])))


def pca_evaluation(real, synthetic):
    pca = PCA(n_components=2)
    pca.fit(real.values)

    real_projection = pca.transform(real.values)
    synthetic_projection = pca.transform(synthetic.values)

    ax = pd.DataFrame(data=real_projection).plot(x=0, y=1, c='red', kind='scatter', s=0.5)
    pd.DataFrame(data=synthetic_projection).plot(x=0, y=1, c='blue', kind='scatter', s=0.5, ax=ax)
    plt.savefig('out.png')


