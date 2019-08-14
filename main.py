import sklearn.datasets
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from scipy import sparse
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_top_cat_indices(target_matrix, num_cats):
    '''Returns column indices of the top n categories by number of hits'''
    cat_counts = target_matrix.sum(axis=0)
    # cat_counts = cat_counts.reshape((1,103)).tolist()[0]
    cat_counts = cat_counts.reshape((103,))

    ind_temp = np.argsort(-cat_counts).tolist()[0]

    ind = [ind_temp[i] for i in range(num_cats)]
    return ind


def prepare_data(x, y, top_cat_indices, sample_size):
    '''Returns list with train x and train y for categories specified by top_cat_indices'''
    res_lst = []

    for i in top_cat_indices:
        # get column of indices with relevant cat
        temp = y.tocsc()[:, i]

        bool_cat = np.where(temp.sum(axis=1) > 0)[0]
        bool_noncat = np.where(temp.sum(axis=1) == 0)[0]

        # all docs with labeled category
        cat_present = x.tocsr()[bool_cat, :]
        # all docs other than labelled category
        cat_notpresent = x.tocsr()[bool_noncat, :]
        # get indices equal to 1/2 of sample size
        idx_cat = np.random.randint(cat_present.shape[0], size=int(sample_size / 2))
        idx_nocat = np.random.randint(cat_notpresent.shape[0], size=int(sample_size / 2))
        # concatenate the ids

        sampled_x_pos = cat_present.tocsr()[idx_cat, :]
        sampled_x_neg = cat_notpresent.tocsr()[idx_nocat, :]
        sampled_x = sparse.vstack((sampled_x_pos, sampled_x_neg))

        sampled_y_pos = temp.tocsr()[bool_cat][idx_cat, :]
        sampled_y_neg = temp.tocsr()[bool_noncat][idx_nocat, :]
        sampled_y = sparse.vstack((sampled_y_pos, sampled_y_neg))

        res_lst.append((sampled_x, sampled_y))

    return res_lst


def prep_validation_cats(y, top_cat_indices):
    '''Returns list with arrays to be used as y argument for comparing with results.
        Leaves only the relevant category from the 103 that are in the target'''
    res_lst = []
    for i in top_cat_indices:
        temp = y.tocsc()[:, i]
        res_lst.append(temp)
    return res_lst


def build_log_models(input_list, cat_names):
    '''Builds logistic regression models from list containing tuples of x and y variables for every category being predicted'''
    model_list = []
    for i in range(len(input_list)):
        x, y = input_list[i]
        cat = cat_names[i]
        print("Building models for category {}".format(cat))
        model = LogisticRegression().fit(x, y.A)
        model_list.append(model)
    return model_list

def build_nn_models(input_list, cat_names):

    model_list = []
    for i in range(len(input_list)):
        x, y = input_list[i]
        cat = cat_names[i]
        print("Building models for category {}".format(cat))

        model = Sequential()
        model.add(Dense(20, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        model.fit(x, y, epochs=5, batch_size=32)
        model_list.append(model)
    return model_list

def predict_data_orig(model_list, test_y):
    '''Predicts using list of models and outputs tuple of precision and recall lists for every model'''
    prec_lst = []
    rec_lst = []
    for i in range(len(model_res)):
        predictions = model_list[i].predict(valid_x)
        precision = precision_score(test_y[i].A, predictions)
        prec_lst.append(precision)

        recall = recall_score(test_y[i].A, predictions)
        rec_lst.append(recall)

    return prec_lst, rec_lst


def predict_data(model_list, val_x, test_y):
    '''Predicts using list of models and outputs tuple of precision and recall lists for every model'''
    prec_lst = []
    rec_lst = []
    for i in range(len(model_list)):
        predictions = model_list[i].predict_classes(val_x)
        precision = precision_score(test_y[i].A, predictions)
        prec_lst.append(precision)

        recall = recall_score(test_y[i].A, predictions)
        rec_lst.append(recall)

    return prec_lst, rec_lst

def plot_metrics(x_axis, prec_res, recall_res):
    plt.plot(x_axis, prec_res, label='Precision')
    plt.plot(x_axis, recall_res, label='Recall')
    plt.title("Stats for most prevalent categories")
    plt.legend()
    plt.plot()
    plt.savefig('nn_res.png')

def main():
    rcv1 = sklearn.datasets.fetch_rcv1()
    res = rcv1.target

    # split into train and test
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(rcv1.data, res)

    ind = get_top_cat_indices(rcv1.target, 5)
    train_res = prepare_data(train_x, train_y, ind, 20000)
    val_res = prep_validation_cats(valid_y, ind)

    nn_models = build_nn_models(train_res, rcv1.target_names[ind])

    precision_res, recall_res = predict_data(nn_models, valid_x, val_res)

    plot_metrics(rcv1.target_names[ind], precision_res, recall_res)

if __name__=="__main__":
    main()

