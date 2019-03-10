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


    n_counts = np.argsort(final_dist_matrix, axis=0, kind='mergesort')

    k_counts = n_counts[0:k, :]

    train_labels = labels.values



    predicted_matrix = train_labels[k_counts[:, :]]

    predicted_matrix = np.sort(predicted_matrix, axis=0, kind='mergesort')
    predicted_matrix_df = pd.DataFrame(predicted_matrix)

    value_counts = predicted_matrix_df.apply(lambda x: x.value_counts()).fillna(0).astype('int64')
    predictions = value_counts.apply(lambda x : x.values.argmax())

    predictions = predictions.apply(lambda x : dataset_labels[x])

    predictions = predictions.values

    '''
    predictions = []
    for distances in final_dist_matrix.T:

        dist_with_labels = np.concatenate((distances[:,None],labels[:,None]),axis=1)
        distance_df = pd.DataFrame.from_records(dist_with_labels,columns = ["distance", "labels"])
        k_distances = distance_df.nsmallest(k, ["distance"])

        label_counts = k_distances["labels"].value_counts()

        votes = label_counts.reindex(dataset_labels)
        votes = votes.fillna(0)
        votes = votes.astype('int64')
        predicted_label = votes.values.argmax()

        predictions.append(dataset_labels[predicted_label])


    predictions = np.asarray(predictions)
    '''
    correct_predictions = sum(test_labels == predictions)
    acc = correct_predictions / predictions.shape[0]
    #print(train_size, end = ",")
    #print(acc)
    return acc

def splitAndCompute(k):
    training_sizes = []
    accuracy_list = []
    for i in range(0,10):
        if(i == 9):
            train_df = global_train_df
            train_labels = global_train_labels
        else:
            train_df = global_train_df.iloc[0 : math.floor((i+1)*step_size),]
            train_labels = global_train_labels.iloc[0 : math.floor((i+1)*step_size),]

        train_df_numeric = pd.DataFrame()
        train_df_categoric = pd.DataFrame()

        test_df_numeric = pd.DataFrame()
        test_df_categoric = pd.DataFrame()

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
        acc = computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size, train_labels, k)
        training_sizes.append(train_size)
        accuracy_list.append(acc)
    return training_sizes, accuracy_list

def computeGraph():
    k_list = [10,20,30]

    for k in k_list:
        training_sizes, accuracy_list = splitAndCompute(k)
        plt.plot(training_sizes, accuracy_list, label="k="+str(k))
        plt.legend()
    plt.title("training size vs accuracies")
    plt.xlabel("training size")
    plt.ylabel("accuracy")
    plt.savefig("learning_curve.pdf")
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
    train_columns = train_columns[:,0]


    global_train_df = pd.DataFrame.from_records(train_json_data["data"], columns= train_columns)

    global_train_labels = global_train_df.iloc[:, -1]
    global_train_df = global_train_df.iloc[:, : -1]

    total_size = global_train_df.shape[0]
    step_size = (total_size / 10)




    test_columns = np.array(test_json_data["metadata"]["features"])
    test_datatypes = test_columns[:, 1]

    test_columns = test_columns[:, 0]


    test_df = pd.DataFrame.from_records(test_json_data["data"], columns= test_columns)
    test_labels = test_df.iloc[:, -1]
    test_df = test_df.iloc[:, : -1]

    training_sizes,accuracy_list = splitAndCompute(k)

    for training_size, accuracy in zip(training_sizes, accuracy_list):
        print(training_size, end=",")
        print(accuracy)


    # Uncomment below code to perform computation for graph
    '''
    #Computing for graph
    computeGraph()
    '''
























