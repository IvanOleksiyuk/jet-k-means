from k_means_train import k_means_process
from train_MoG_sphere import MoG_sphere_process
from properties_table import create_table_of_kmeans_cluster_properties
from ROCcompare import contamination_ROC_compare
import reprocessing
import shutil
import os

#%% Chose a location to store the results. Create all directories if needed
output_folder="paper_results/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder+"ROC_pickles", exist_ok=True)

#%% Choose datasets, direction of tagging (direct/reverse), reweighting (preprocessing), etc
dataset_list=["1h5", "1h5", "2f", "2f", "4f", "4f"]
REVERSE_list=[False, True, False, False, False, False]
preproc_list=[None, None, reprocessing.reproc_4rt, None, reprocessing.reproc_4rt, None]
dataset_label_list=["QCDh5+toph5", "toph5+QCDh5", "QCDf+DMlf", "QCDf+DMlf", "QCDf+Q50M10lf", "QCDf+Q50M10lf"]
reproc_label_list=["none", "none", "4rt", "none", "4rt", "none"]

#%% run all k-means based training and evaluations
SCORE_TYPE_list=["MinD", "KNC", "logLds", "logLrhn", "logLrh0"]
SCORE_TYPE_true_names_list=["MinD",  "KNC5", "logLds", "logLrhn", "logLrh0"]
knc=5 
crop=100000
k_clusters=100
cont=0
SIGMA=3
train_mode="d"

for SCORE_TYPE in SCORE_TYPE_list:

    for i in range(len(dataset_list)):
        k_means_process(dataset=dataset_list[i],
                         k_clusters=k_clusters,
                         SIGMA=SIGMA,
                         crop=crop,
                         cont=cont,
                         preproc=preproc_list[i],
                         Id=0,
                         knc=knc,
                         characterise=True,
                         SAVE_CHAR=True,
                         REVERSE=REVERSE_list[i],
                         train_mode=train_mode, 
                         data=None,
                         SCORE_TYPE=SCORE_TYPE,
                         MINI_BATCH=False,
                         plot_dim=True)


#%% run k-means based training and evaluations on 5dim gaussian dataset
k_means_process(dataset=8.1,
            k_clusters=100,
            SIGMA=0,
            crop=100000,
            preproc=None, 
            Id=0,
            knc=5,                                 
            characterise=True,
            SAVE_CHAR=True,
            REVERSE=False,
            SCORE_TYPE="MinD",
            data=None,
            plot_dim=True,
            density="Gaussianlog")

k_means_process(dataset=8.1,
            k_clusters=100,
            SIGMA=0,
            crop=100000,
            preproc=None,
            Id=0,
            knc=5,                                 
            characterise=True,
            SAVE_CHAR=True,
            REVERSE=False,
            SCORE_TYPE="KNC",
            data=None,
            plot_dim=True,
            density="Gaussianlog")

#%% run all GMM evaluations
train_mode="bl"
reg_covar=-11
for i in range(len(dataset_list)):
        MoG_sphere_process(dataset=dataset_list[i],
                            k_clusters=k_clusters,
                            SIGMA=SIGMA,
                            crop=crop,
                            cont=cont,
                            preproc=preproc_list[i], 
                            Id=0,
                            reg_covar=reg_covar,
                            REVERSE=REVERSE_list[i],                                 
                            characterise=True,
                            SAVE_CHAR=True, 
                            train_mode=train_mode, 
                            data=None)

#%% produce all ROC plots 
box_labels=["direct top\n tagging", "reverse top\n tagging",  "Aachen $\sqrt[4]{p_T}$", "Aachen",  "Heidelberg $\sqrt[4]{p_T}$", "Heidelberg"]
i=0
for dataset_label, reproc_label in zip(dataset_label_list, reproc_label_list):
    names=[]
    for SCORE_TYPE in SCORE_TYPE_true_names_list:
        names.append("char/{:}m100s3c100r{:}KId0{:}".format(dataset_label, reproc_label, SCORE_TYPE))
    names.append("char/MoG{:}m100s3c100r{:}KIbl0".format(dataset_label, reproc_label))
    contamination_ROC_compare(names,
                              methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLLN", "GMMLL"], 
                              plot_name=dataset_label+reproc_label,
                              box_label=box_labels[i],
                              output_folder=output_folder)
    i+=1
#%% store all ROC pickles
for SCORE_TYPE in SCORE_TYPE_true_names_list:
    for i in range(len(dataset_list)):
        shutil.copy("char/{:}m100s3c100r{:}KId0{:}/".format(dataset_label_list[i], reproc_label_list[i], SCORE_TYPE)+"res.pickle", output_folder+"ROC_pickles/ROC_{:}_{:}".format(dataset_label_list[i], SCORE_TYPE))
    shutil.copy("char/MoG{:}m100s3c100r{:}KIbl0/".format(dataset_label_list[i], reproc_label_list[i])+"res.pickle", output_folder+"ROC_pickles/ROC_{:}_{:}".format(dataset_label_list[i], "GMM"))

#%% copy dimension analysis plot
shutil.copy("char/QCDh5+toph5m100s3c100rnoneKId0logLds/dimensions.png", output_folder+"dimensions.png")

#%% copy loss-density correlation plots
shutil.copy("char/5d1s+tor5d4u1sm100s0c100rnoneKId0KNC5/correlation.png", output_folder+"correlation_KNC5.png")
shutil.copy("char/5d1s+tor5d4u1sm100s0c100rnoneKId0MinD/correlation.png", output_folder+"correlation_MinD.png")

#%% produce a table with k-means cluster parameters
datasets_list=["QCDh5+toph5", "toph5+QCDh5", "QCDf+DMlf", "QCDf+DMlf"]
dataset_labels=["QCD hpt", "top hpt", "QCD lpt", "QCD lpt 4rt"]
preproc_list=["none", "none", "none", "4rt"]
preproc_write=["", "", " 4rt",  ""]
metric_list=["mu_min", "mu_median", "mu_max", "sig/mu_max", "d_med"]
labels_list=["$min(\\rho_i)$", "$med(\\rho_i)$", "$max(\\rho_i)$", "$max(\\sigma_i/\\rho_i)$", "$med(d_i(\\rho_i))$"]
create_table_of_kmeans_cluster_properties(datasets_list, preproc_list, preproc_write, metric_list, labels_list, savefile=output_folder+"dim_table.tex", dataset_labels=dataset_labels)
