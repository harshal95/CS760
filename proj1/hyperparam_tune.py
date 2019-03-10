import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


k = int(sys.argv[1])
train_file_path = sys.argv[2]
valid_file_path = sys.argv[3]
test_file_path = sys.argv[4]



def computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size):
    train_df_numeric_values = train_df_numeric.values
    train_df_categoric_values = train_df_categoric.values

    test_df_numeric_values = test_df_numeric.values
    test_df_categoric_values = test_df_categoric.values

    dist_matrix_numeric = np.sum(abs(train_df_numeric_values[:, np.newaxis, :] - test_df_numeric_values), axis=2)
    dist_matrix_categoric = np.sum(train_df_categoric_values[:, np.newaxis, :] != test_df_categoric_values, axis=2)

    final_dist_matrix = np.zeros((train_size, test_size))

    if (dist_matrix_numeric.size != 0):
        final_dist_matrix += dist_matrix_numeric
    if (dist_matrix_categoric.size != 0):
        final_dist_matrix += dist_matrix_categoric

    return final_dist_matrix

def computePredictions(dist_matrix, train_labels, i):



    n_counts = np.argsort(dist_matrix, axis=0, kind='mergesort')

    k_counts = n_counts[0:i, :]

    train_labels = train_labels.values



    predicted_matrix = train_labels[k_counts[:, :]]

    predicted_matrix = np.sort(predicted_matrix, axis=0, kind='mergesort')
    predicted_matrix_df = pd.DataFrame(predicted_matrix)

    value_counts = predicted_matrix_df.apply(lambda x: x.value_counts()).fillna(0).astype('int64')
    predictions = value_counts.apply(lambda x : x.values.argmax())

    predictions = predictions.apply(lambda x : dataset_labels[x])

    predictions = predictions.values


    return predictions



    '''
    predictions = []

    for distances in dist_matrix.T:
        dist_with_labels = np.concatenate((distances[:, None], train_labels[:, None]), axis=1)
        distance_df = pd.DataFrame.from_records(dist_with_labels, columns=["distance", "labels"])
        k_distances = distance_df.nsmallest(i, ["distance"])

        label_counts = k_distances["labels"].value_counts()

        votes = label_counts.reindex(dataset_labels)
        votes = votes.fillna(0)
        votes = votes.astype('int64')
        predicted_label = votes.values.argmax()
        predictions.append(dataset_labels[predicted_label])

    predictions = np.asarray(predictions)
    return predictions
    '''




with open(train_file_path) as f1, open(test_file_path) as f2, open(valid_file_path) as f3:
    train_json_data = json.load(f1)
    test_json_data = json.load(f2)
    valid_json_data = json.load(f3)



    train_columns = np.array(train_json_data["metadata"]["features"])


    train_datatypes = train_columns[:, 1]
    train_datatypes = train_datatypes[: -1]

    dataset_labels = train_columns[-1, 1]
    train_columns = train_columns[:,0]


    train_df = pd.DataFrame.from_records(train_json_data["data"], columns= train_columns)
    train_labels = train_df.iloc[:, -1]
    train_df = train_df.iloc[:, : -1]

    valid_df = pd.DataFrame.from_records(valid_json_data["data"], columns= train_columns)
    valid_labels = valid_df.iloc[:, -1]
    valid_df = valid_df.iloc[:, : -1]

    train_df_numeric = pd.DataFrame()
    train_df_categoric = pd.DataFrame()

    valid_df_numeric = pd.DataFrame()
    valid_df_categoric = pd.DataFrame()

    test_columns = np.array(test_json_data["metadata"]["features"])
    test_datatypes = test_columns[:, 1]

    test_columns = test_columns[:, 0]


    test_df = pd.DataFrame.from_records(test_json_data["data"], columns= test_columns)
    test_labels = test_df.iloc[:, -1]
    test_df = test_df.iloc[:, : -1]

    test_df_numeric = pd.DataFrame()
    test_df_categoric = pd.DataFrame()

    for i,column in enumerate(train_df):
        if(train_datatypes[i] == "numeric"):
            train_df_numeric[column] = train_df[column]
            valid_df_numeric[column] = valid_df[column]
            test_df_numeric[column] = test_df[column]
        else:
            train_df_categoric[column] = train_df[column]
            valid_df_categoric[column] = valid_df[column]
            test_df_categoric[column] = test_df[column]


    frames_numeric = [train_df_numeric, valid_df_numeric]
    tuned_df_numeric = pd.concat(frames_numeric)

    frames_categoric = [train_df_categoric, valid_df_categoric]
    tuned_df_categoric = pd.concat(frames_categoric)

    for column in train_df_numeric:

        mean_var = np.sum(train_df_numeric[column]) / train_df_numeric[column].size

        stddev_var = np.sqrt(np.sum(np.square(train_df_numeric[column] - mean_var))/ train_df_numeric[column].size)
        if(stddev_var == 0):
            stddev_var = 1

        train_df_numeric[column] = (train_df_numeric[column] - mean_var)/stddev_var
        valid_df_numeric[column] = (valid_df_numeric[column] - mean_var)/stddev_var


    train_size = train_df.shape[0]
    valid_size = valid_df.shape[0]

    dist_matrix = computeDistance(train_df_numeric, train_df_categoric, valid_df_numeric, valid_df_categoric, train_size, valid_size)

    valid_labels_values = valid_labels.values

    valid_acc= np.zeros(k)

    for i in range(1,k+1):

        predictions = computePredictions(dist_matrix,train_labels,i)
        correct_predictions = sum(valid_labels_values == predictions)
        acc = correct_predictions/predictions.shape[0]
        print(i,end=",")
        print(acc)
        valid_acc[i-1] = acc

    optimal_k = np.argmax(valid_acc) + 1
    print(optimal_k)





    for column in tuned_df_numeric:

        mean_var = np.sum(tuned_df_numeric[column]) / tuned_df_numeric[column].size

        stddev_var = np.sqrt(np.sum(np.square(tuned_df_numeric[column] - mean_var))/ tuned_df_numeric[column].size)
        if(stddev_var == 0):
            stddev_var = 1

        tuned_df_numeric[column] = (tuned_df_numeric[column] - mean_var)/stddev_var
        test_df_numeric[column] = (test_df_numeric[column] - mean_var)/stddev_var

    tune_size = train_size + valid_size
    test_size = test_df.shape[0]


    dist_matrix = computeDistance(tuned_df_numeric, tuned_df_categoric, test_df_numeric, test_df_categoric, tune_size, test_size)



    label_frames = [train_labels, valid_labels]
    tuned_labels = pd.concat(label_frames)
    predictions = computePredictions(dist_matrix, tuned_labels, optimal_k)
    correct_predictions = sum(test_labels.values == predictions)
    final_acc = correct_predictions / predictions.shape[0]
    print(final_acc)



    '''
    plt.plot(np.arange(1, k+1), valid_acc)


    plt.xlabel("k-values")
    plt.ylabel("Validation Accuracy")

    plt.title("k vs Validation Accuracy ")
    plt.savefig("tune_k.pdf")
    plt.show()
    '''





























