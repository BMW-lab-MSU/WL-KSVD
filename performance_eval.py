import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

datasets = ["MUTAG",  "NCI1", "NCI109", "PTC_FM", "PTC_MR","PROTEINS"]
datasets = ["MUTAG"]

clf_path = "./Classifier_data/"

clf_list = [ "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"]

G_emb_list = ["G2V", "GL2V", "SF", "WL_KSVD"]
ndims_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

KFolds = 5

# clf_list =["Linear SVM"]

for ds_name in datasets:
    acc ={}
    score_path = clf_path + ds_name + '/Score/'

    score_test = pickle.load(open(score_path + 'score_test_' + ds_name + '.pkl', 'rb'))
    score_val = pickle.load(open(score_path + 'score_val_' + ds_name + '.pkl', 'rb'))

    for emb_name in G_emb_list:
        acc[emb_name] ={}
        for clf_name in clf_list:
            acc[emb_name][clf_name] ={}

            acc_val_fold = pd.DataFrame(np.zeros([KFolds, len(ndims_list)]), index=range(KFolds), columns=ndims_list)
            for n_KFold in range(KFolds):
                for n_dims in ndims_list:
                    acc_val_fold.at[n_KFold, n_dims] = score_val[str(n_KFold)][n_dims][emb_name][clf_name]

            acc[emb_name][clf_name]['results'] =acc_val_fold
            acc[emb_name][clf_name]['mean'] = acc_val_fold.mean(axis = 0)
            acc[emb_name][clf_name]['std'] = acc_val_fold.std(axis=0)

    acc_mean = {}
    acc_std = {}
    for clf_name in clf_list:
        print(clf_name)

        n_rows = len(G_emb_list)
        n_cols = len(ndims_list)

        score_val_avg_df = pd.DataFrame(np.zeros([n_rows, n_cols]), index=G_emb_list, columns=ndims_list)
        score_val_std_df = pd.DataFrame(np.zeros([n_rows, n_cols]), index=G_emb_list, columns=ndims_list)

        for emb_name in G_emb_list:
            print(emb_name)
            m = acc[emb_name][clf_name]['mean']
            score_val_avg_df.at[emb_name] = m
            score_val_std_df.at[emb_name] = acc[emb_name][clf_name]['std']

        acc_mean[clf_name] =score_val_avg_df
        acc_std[clf_name] = score_val_std_df

        acc_val_file = open(score_path + "acc_val_" + ds_name + ".pkl", "wb")
        pickle.dump(acc, acc_val_file)
        acc_val_file.close()

        acc_mean_val_file = open(score_path + "acc_mean_val_" + ds_name + ".pkl", "wb")
        pickle.dump(acc_mean, acc_mean_val_file)
        acc_mean_val_file.close()

        acc_std_val_file = open(score_path + "acc_std_val_" + ds_name + ".pkl", "wb")
        pickle.dump(acc_mean, acc_std_val_file)
        acc_std_val_file.close()
