import json
import numpy as np
import pandas as pd
import sys



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

        for vote in votes:
            print(vote, end=",")

        print(dataset_labels[predicted_label])

    predictions = np.asarray(predictions)
    correct_predictions = sum(test_labels == predictions)
    acc = correct_predictions / predictions.shape[0]
    #print("Accuracy: ",acc)



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


    train_df = pd.DataFrame.from_records(train_json_data["data"], columns= train_columns)

    train_labels = train_df.iloc[:, -1]
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
    computeDistance(train_df_numeric, train_df_categoric, test_df_numeric, test_df_categoric, train_size, test_size, train_labels, k)


























