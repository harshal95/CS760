import json
import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt


def computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric,train_size, test_size, labels, k):

    train_df_numeric_values = train_df_numeric.values
    train_df_categoric_values = train_df_categoric.values

    test_df_numeric_values = test_df_numeric.values
    test_df_categoric_values = test_df_categoric.values

    dist_matrix_numeric = np.sum(abs(train_df_numeric_values[:, np.newaxis, :] - test_df_numeric_values), axis=2)
    dist_matrix_categoric = np.sum(train_df_categoric_values[:, np.newaxis, :] != test_df_categoric_values, axis=2)


    final_dist_matrix = np.zeros((train_size, test_size))

    if(dist_matrix_numeric.size!= 0):
        final_dist_matrix+= dist_matrix_numeric
    if(dist_matrix_categoric.size!= 0):
        final_dist_matrix+= dist_matrix_categoric

    confidence_scores = []


    for distances in final_dist_matrix.T:

        dist_with_labels = np.concatenate((distances[:,None],labels[:,None]),axis=1)
        distance_df = pd.DataFrame.from_records(dist_with_labels,columns = ["distance", "labels"])
        k_nearest = distance_df.nsmallest(k, ["distance"])
        k_nearest_values = k_nearest.values
        weights = 1/ (np.square(k_nearest_values[:,0]) + math.pow(10, -5))
        confidence_score = np.dot(weights, k_nearest_values[:,1]) / np.sum(weights, axis= 0)
        confidence_scores.append(confidence_score)

    return confidence_scores


def computeRoc(scores_with_labels):
    # ROC logic
    num_neg = np.sum(scores_with_labels[:, 1] == 0)
    num_pos = np.sum(scores_with_labels[:, 1] == 1)
    tp = 0
    fp = 0
    last_tp = 0
    fpr = 0
    tpr = 0

    fpr_list = []
    tpr_list = []

    for i, cur_row in enumerate(scores_with_labels):
        if (i > 0 and scores_with_labels[i - 1][0] != cur_row[0] and cur_row[1] == 0 and tp > last_tp):
            fpr = fp / num_neg
            tpr = tp / num_pos
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            last_tp = tp

        if (cur_row[1] == 1):
            tp = tp + 1
        else:
            fp = fp + 1

    fpr = fp / num_neg
    tpr = tp / num_pos

    fpr_list.append(fpr)
    tpr_list.append(tpr)

    return fpr_list, tpr_list


def computeGraph(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size, train_labels, k_list):

    for k in k_list:
        confidence_scores = computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric,
                                            train_size, test_size, train_labels, k)
        confidence_scores = np.asarray(confidence_scores)

        scores_with_labels = np.concatenate((confidence_scores[:, np.newaxis], test_labels[:, np.newaxis]), axis=1)
        scores_with_labels = scores_with_labels[scores_with_labels[:, 0].argsort()[::-1]]

        fpr_list, tpr_list = computeRoc(scores_with_labels)
        plt.plot(fpr_list, tpr_list, label="k="+str(k))
        plt.legend()

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.title("ROC curve")
    plt.savefig("roc_curve.pdf")
    plt.show()





k = int(sys.argv[1])
train_file_path = sys.argv[2]
test_file_path = sys.argv[3]

with open(train_file_path) as f1, open(test_file_path) as f2:

    train_json_data = json.load(f1)
    test_json_data = json.load(f2)


    train_columns = np.array(train_json_data["metadata"]["features"])


    train_datatypes = train_columns[:, 1]
    train_datatypes = train_datatypes[: -1]

    dataset_labels = train_columns[-1, 1]
    positive_class = dataset_labels[0]

    train_columns = train_columns[:,0]


    train_df = pd.DataFrame.from_records(train_json_data["data"], columns= train_columns)

    train_labels = train_df.iloc[:, -1]

    positive_train_labels = train_labels == positive_class
    negative_train_labels = train_labels != positive_class
    train_labels[positive_train_labels] = 1
    train_labels[negative_train_labels] = 0


    train_df = train_df.iloc[:, : -1]


    train_df_numeric = pd.DataFrame()
    train_df_categoric = pd.DataFrame()

    test_df_numeric = pd.DataFrame()
    test_df_categoric = pd.DataFrame()




    test_columns = np.array(test_json_data["metadata"]["features"])
    test_datatypes = test_columns[:, 1]

    test_columns = test_columns[:, 0]


    test_df = pd.DataFrame.from_records(test_json_data["data"], columns= test_columns)
    test_labels = test_df.iloc[:, -1]

    positive_test_labels = test_labels == positive_class
    negative_test_labels = test_labels != positive_class
    test_labels[positive_test_labels] = 1
    test_labels[negative_test_labels] = 0

    test_df = test_df.iloc[:, : -1]


    for i,column in enumerate(train_df):
        if(train_datatypes[i] == "numeric"):
            train_df_numeric[column] = train_df[column]
            test_df_numeric[column] = test_df[column]
        else:
            train_df_categoric[column] = train_df[column]
            test_df_categoric[column] = test_df[column]


    for column in train_df_numeric:

        mean_var = np.sum(train_df_numeric[column]) / train_df_numeric[column].size

        stddev_var = np.sqrt(np.sum(np.square(train_df_numeric[column] - mean_var))/ train_df_numeric[column].size)
        if(stddev_var == 0):
            stddev_var = 1

        train_df_numeric[column] = (train_df_numeric[column] - mean_var)/stddev_var
        test_df_numeric[column] = (test_df_numeric[column] - mean_var)/stddev_var


    train_size = train_df.shape[0]
    test_size = test_df.shape[0]

    confidence_scores = computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size, train_labels, k)
    test_labels = np.asarray(test_labels)
    confidence_scores = np.asarray(confidence_scores)


    scores_with_labels = np.concatenate((confidence_scores[:,np.newaxis], test_labels[:, np.newaxis]), axis=1)
    scores_with_labels = scores_with_labels[scores_with_labels[:,0].argsort()[::-1]]

    fpr_list, tpr_list = computeRoc(scores_with_labels)

    for fpr_item,tpr_item in zip(fpr_list, tpr_list):
        print(fpr_item, end=",")
        print(tpr_item)


    k_list = [10,20,30]

    #Uncomment below code to compute graph for various values of k
    '''
    computeGraph(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size, train_labels, k_list)
    '''









