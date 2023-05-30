from reader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from karateclub.graph_embedding import Graph2Vec, GL2Vec, SF, IGE
import timeit

from WL_KSVD import WL_KSVD

import pandas as pd
import numpy as np
import os
from joblib import dump, load
import pickle


def tsne_2d_plot(vector, labels, n_vec, plt_title='TSNE visualization', return_plot=False):
    two_dim_embedding = TSNE(n_components=2).fit_transform(vector)

    unique_labels = set(train_labels)

    for label in unique_labels:
        label_idx = [i for i in range(n_vec) if labels[i] == label]

        plt.plot(two_dim_embedding[label_idx, 0], two_dim_embedding[label_idx, 1], 'o', label=label, alpha=0.2)

    plt.suptitle(plt_title)
    plt.legend(bbox_to_anchor=(0.8, 1))
    plt.legend(loc='upper left')

    fig1 = plt.gcf()
    plt.show()

    if return_plot:
        return two_dim_embedding, fig1
    else:
        plt.clf()
        return two_dim_embedding


if __name__ == "__main__":
    print('Starting!')

    # Path for the graph datasets
    ds_path = "/media/kaveen/D/Datasets/graph_datasets"

    # list of embedding methods
    # ds_list = ["ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "NCI1", "NCI109", "PTC_FM", "PROTEINS", "REDDIT-BINARY",
    #            "YEAST", "YEASTH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H", "ZINC_full", "alchemy_full", "QM9"]

    # ds_list = ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]
    # ds_list = [  "NCI109", "PTC_MR", "PROTEINS"]
    ds_list = ["ENZYMES"]

    # G_emb_list = ["G2V", "GL2V", "SF", "WL_KSVD"]
    G_emb_list = ["GL2V"]
    # G_emb_list = ["WL_KSVD"]

    # ndims_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ndims_list = [2]

    # number of K-Folds
    K_folds = 2

    # path for the output vector embeddings
    emb_path = "./Embeddings/"

    save_emb = True
    plot_emb = True
    plot_save = True

    time_ds = {}
    for ds_name in ds_list:

        print('Dataset: ' + ds_name)

        # load graph dataset
        graph_DB, graph_DB_labels = tud_to_networkx(ds_path=ds_path, ds_name=ds_name)
        print("Data Loaded")

        graph_DB_names = [i for i in range(0, len(graph_DB_labels))]

        skf = StratifiedKFold(n_splits=K_folds)

        k = 0
        time_fold = {}
        for train_index, test_index in skf.split(graph_DB, graph_DB_labels):

            n_KFold = str(k)

            k += 1

            graph_DB_train = []
            graph_DB_train_labels = []
            graph_DB_train_names = []
            for idx in train_index:
                graph_DB_train = graph_DB_train + [graph_DB[idx]]
                graph_DB_train_labels = graph_DB_train_labels + [graph_DB_labels[idx]]
                graph_DB_train_names = graph_DB_train_names + [graph_DB_names[idx]]

            test_graphs = []
            test_labels = []
            test_names = []
            for idx in np.nditer(test_index):
                test_graphs = test_graphs + [graph_DB[idx]]
                test_labels = test_labels + [graph_DB_labels[idx]]
                test_names = test_names + [graph_DB_names[idx]]

            # graph_DB_train = list(graph_DB[idx] for idx in np.nditer(train_index))
            # test_graph = list(graph_DB[idx] for idx in np.nditer(test_index))
            #
            # graph_DB_train_labels = list(graph_DB_labels[idx] for idx in np.nditer(train_index))
            # test_labels = list(graph_DB_labels[idx] for idx in np.nditer(test_index))
            #
            # # graph_DB_train, test_graphs = graph_DB(train_index], graph_DB[test_index]
            # graph_DB_train_labels, test_labels = graph_DB_labels[train_index], graph_DB_labels[test_index]
            # graph_DB_train_names, test_names = graph_DB_names[train_index], graph_DB_names[test_index]

            train_graphs, train_clf_graphs, train_labels, train_clf_labels, train_names, train_clf_names \
                = train_test_split(graph_DB_train, graph_DB_train_labels, graph_DB_train_names,
                                   test_size=0.5, random_state=42)

            n_train_clf = len(train_clf_labels)
            n_test = len(test_labels)

            time_emb = {}
            for G_emb_name in G_emb_list:

                time_dim = {}
                for ndims in ndims_list:
                    print(ds_name)
                    print("K-Fold :" + n_KFold)
                    print(G_emb_name)
                    dimensions = ndims
                    print("Dimension(K) = ", dimensions)

                    if G_emb_name == "G2V":
                        model = Graph2Vec(dimensions=dimensions)

                    elif G_emb_name == "GL2V":
                        model = GL2Vec(dimensions=dimensions)

                    elif G_emb_name == "SF":
                        model = SF(dimensions=dimensions)

                    elif G_emb_name == "WL_KSVD":
                        model = WL_KSVD(dimensions=dimensions, n_vocab=10000,
                                        n_non_zero_coefs=int(np.ceil(dimensions / 10)))
                    else:
                        print("Embedding method not incorporated")
                        break

                    try:
                        print("model training")
                        tic = timeit.default_timer()
                        model.fit(train_graphs)
                        toc = timeit.default_timer()
                        time_dim[dimensions] = toc - tic

                        train_emb_embeddings = model.get_embedding()

                        print('Model infer')
                        train_clf_embeddings = model.infer(train_clf_graphs)
                        test_embeddings = model.infer(test_graphs)

                        if save_emb:
                            train_emb_path = emb_path + ds_name + '/Fold_' + n_KFold + '/' + G_emb_name + '/train/'
                            test_emb_path = emb_path + ds_name + '/Fold_' + n_KFold + '/' + G_emb_name + '/test/'

                            os.makedirs(train_emb_path, exist_ok=True)
                            os.makedirs(test_emb_path, exist_ok=True)

                            fig_path = emb_path + ds_name + '/Fold_' + n_KFold + '/' + G_emb_name + '/Figures/'
                            model_path = emb_path + ds_name + '/Fold_' + n_KFold + '/' + G_emb_name + '/Models/'

                            os.makedirs(fig_path, exist_ok=True)
                            os.makedirs(model_path, exist_ok=True)

                            model_name = (model_path + '/' + 'model_' + str(dimensions) + 'model.joblib')
                            dump(model, model_name)
                            # model2 = load('saved_model.joblib')

                            pd.DataFrame(train_clf_embeddings).to_csv(train_emb_path + 'emb_vectors_' + str(dimensions)
                                                                      + '.csv', header=None, index=None)
                            pd.DataFrame(test_embeddings).to_csv(
                                test_emb_path + 'emb_vectors_' + str(dimensions) + '.csv',
                                header=None, index=None)

                            train_clf_df = pd.DataFrame({'name': train_clf_names, 'Label': train_clf_labels})
                            train_clf_df.to_csv(train_emb_path + 'labels_' + str(dimensions) + '.csv')

                            test_df = pd.DataFrame({'name': test_names, 'Label': test_labels})
                            test_df.to_csv(test_emb_path + 'labels_' + str(dimensions) + '.csv')

                            if plot_emb:
                                print('Visualizations training')
                                title = ds_name + ' ' + G_emb_name + ' embedding \n(' \
                                        + str(dimensions) + ' dims) train graphs'
                                tsne_2d_vector_train, fig_train = tsne_2d_plot(train_clf_embeddings, train_clf_labels,
                                                                           n_train_clf, title, return_plot=True)
                                if plot_save:
                                    fig_name = fig_path + 'train_vector_' + str(dimensions) + '-dims.png'
                                    fig_train.savefig(fig_name)
                                plt.clf()
                                plt.close(fig_train)

                                title = ds_name + ' ' + G_emb_name + ' embedding \n(' + str(
                                    dimensions) + ' dims) test graphs'
                                tsne_2d_vector_test, fig_test = tsne_2d_plot(test_embeddings, test_labels, n_test, title,
                                                                          return_plot=True)

                                if plot_save:
                                    fig_name = fig_path + 'test_vector_' + str(dimensions) + '-dims.png'
                                    fig_test.savefig(fig_name)
                                plt.clf()
                                plt.close(fig_test)
                    except Exception as e:
                        print("ERROR!!!!!!! : %s" % e)

                    # deleting the varibales to save space
                    if 'model' in os.environ:
                        del os.environ['model']

                    if 'train_embeddings' in os.environ:
                        del os.environ['train_embeddings']

                    if 'test_embeddings' in os.environ:
                        del os.environ['test_embeddings']

                    if 'train_df' in os.environ:
                        del os.environ['train_df']

                    if 'test_df' in os.environ:
                        del os.environ['test_df']

                    time_emb[G_emb_name] = time_dim

                time_fold[n_KFold] = time_emb

            if 'train_embeddings' in os.environ:
                del os.environ['train_embeddings']

            if 'test_embeddings' in os.environ:
                del os.environ['test_embeddings']

        time_ds_path = emb_path + ds_name + '/Timing/'
        os.makedirs(time_ds_path, exist_ok=True)
        time_file_folds = open(time_ds_path + "time_file_" + ds_name + ".pkl", "wb")
        # write the python object (dict) to pickle file
        pickle.dump(time_fold, time_file_folds)
        # close file
        time_file_folds.close()
        time_ds[ds_name] = time_fold

    time_path = emb_path+'time_file_.pkl'
    time_file_ds = open(time_path, "wb")
    # write the python object (dict) to pickle file
    pickle.dump(time_ds, time_file_ds)
    # close file
    time_file_ds.close()
    print('done!')
