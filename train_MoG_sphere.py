import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import pickle 
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from dataset_path_and_pref import dataset_path_and_pref, prepare_data
import copy

def train_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode, data=None):
    if data is None:
        X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
        if cont>0:
            X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
            X_tr=np.concatenate((X_tr, X_cont))
    else:
        X_tr=data
    if train_mode=="s":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1, max_iter=10, init_params="random").fit(X_tr)
    if train_mode=="d":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1).fit(X_tr)
    if train_mode=="dia":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="diag", random_state=Id, n_init=1, reg_covar=10**reg_covar).fit(X_tr)
    if train_mode=="bl":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1, reg_covar=10**reg_covar).fit(X_tr)
    model_path="models_MoG_sphere/{:}m{:}reg{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, reg_covar, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    pickle.dump(MoG_sphere, open(model_path, "wb"))
    if not (train_mode in ["d", "s", "dia", "bl"]):
        print("invalid train mode!")
    return X_tr, MoG_sphere
    
def train_or_load_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode, data=None, return_data=False):
    model_path="models_MoG_sphere/{:}m{:}reg{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, reg_covar, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("loading trained model", model_path)
        MoG_sphere=pickle.load(open(model_path, "rb"))
        if data is None:
            X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
            if cont>0:
                X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
                X_tr=np.concatenate((X_tr, X_cont))
            return X_tr, MoG_sphere
        else:
            return data, MoG_sphere
    else:
        print("training a new model", model_path)
        X_tr, MoG_sphere =train_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode=train_mode, data=data)
        return X_tr, MoG_sphere

def MoG_sphere_process(dataset=1,
                    k_clusters=10,
                    SIGMA=3,
                    crop=10000,
                    cont=0,
                    reg_covar=-6,
                    preproc=None,
                    Id=0,                         
                    characterise=False,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    train_mode="d", #explanation below
                    data=None, #training data that was alredy loaded and preprocessed by the prevous iteration of the algorythm and so can be reused
                    full_mean_diffs=False,
                    non_smeared_mean=False): 
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont 
# really require the best and fully convergent clustering, repeating clustering 
# with 10 initialisations is thus a waist of resources (it may be better to 
# build an ensemble out of such 10 instead of picking one of 10 with best clustering) 
# "d" - default as it is default in scikit
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation 
    plt.close("all")
    random.seed(a=10, version=2)
    
    DI = dataset_path_and_pref(dataset, REVERSE)
        
    if cont>0:
        DI["pref"]=DI["pref"]+"+"+DI["pref2"]+"{:}".format(cont)+"_"
    
    hyp=(DI["pref"], k_clusters, reg_covar, SIGMA, crop, cont, preproc, Id)
    
    X_tr, MoG_sphere = train_or_load_MoG_sphere(DI, *hyp, train_mode=train_mode, data=data)
    
    cov=copy.deepcopy(MoG_sphere.covariances_)
    cov=np.array(cov)
    cov.sort()
    print(np.sqrt(cov))
    print(cov)
    print(np.median(cov))
    
    if characterise:
        X_bg_val=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        X_sg_val=prepare_data(DI["sg_val_data_path"], field=DI["sg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        tr_scores = -MoG_sphere.score_samples(X_tr)
        bg_scores = -MoG_sphere.score_samples(X_bg_val)
        sg_scores = -MoG_sphere.score_samples(X_sg_val)
        #centroid_scores = -MoG_sphere.score_samples(MoG_sphere.means_)
        
    if characterise:
        labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
        auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
        
        fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
        plt.figure()
        plt.grid()
        plt.plot(tpr, 1/fpr)
        plt.ylim(ymin=1, ymax=1000)
        plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
        plt.yscale("log")
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure(figsize=(10, 10))
        _, bins, _, = plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
        plt.hist(sg_scores, histtype='step', label='sig', bins=bins, density=True)
        #plt.hist(centroid_scores, histtype='step', label='centr', bins=bins, density=True)
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure()
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        plt.legend()
        
        if SAVE_CHAR:
            path="char/MoG{:}+{:}m{:}s{:}c{:}r{:}KI{:}{:}/".format(DI["pref"], DI["pref2"], k_clusters, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
            os.makedirs(path, exist_ok=True)
            k=0
            k+=1
            plt.figure(k)
            plt.savefig(path+"ROC.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"dist.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"SIC.png", bbox_inches="tight")
            
            res={}
            res["fpr"]=fpr
            res["tpr"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            print(path+"res.pickle")
        
if __name__ == "__main__":
    train_mode="bl"
    reg_covar=-11
    MoG_sphere_process(dataset=1,
                k_clusters=100,
                cont=0,
                SIGMA=3,
                crop=100000,
                preproc=None, 
                Id=0,
                reg_covar=reg_covar,
                REVERSE=False,                                 
                characterise=True,
                SAVE_CHAR=True,
                train_mode=train_mode, 
                data=None)
    
    