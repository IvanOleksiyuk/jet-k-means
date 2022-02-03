from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from functools import partial
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from dataset_path_and_pref import dataset_path_and_pref, prepare_data
from utilities import d_ball_volume, dm1_sphere_area
from likelyhood_estimation import likelyhood_estimation_dim_Uniform
import set_matplotlib_default

def gaussian(x, mean, sigma, weight, d=1):
    if d==1:
        return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))
    else:
        return weight/(sigma*(2*np.pi)**0.5)**d*np.exp(-np.sum((x-mean)**2, axis=1)/(2*sigma**2))
    
def d_slopes_norm(x, mean, sigma, weight, d=1, dont_use_weights=False, dont_use_volume=False):
    N_in=d_ball_volume(d, mean)
    N_sl=dm1_sphere_area(d, mean)*sigma*np.sqrt(np.pi/2)
    N=N_in+N_sl
    out=np.zeros(x.shape)
    out[x>=mean] =((mean/x[x>=mean])**(d-1))* np.exp(-(x[x>=mean]-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    if dont_use_volume==False:
        out/=N
    if dont_use_weights:
        return out
    else:
        return out*weight
    
def mean_knc_mins(dists, knc):
    dists_cop=np.copy(dists)
    dists_cop.sort(1)
    return np.mean(dists_cop[:, :knc], 1)


def train_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, MINI_BATCH=False):
    if data is None:
        X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
        if cont>0:
            X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
            X_tr=np.concatenate((X_tr, X_cont))
    else:
        X_tr=data
    if train_mode=="s":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(n_clusters=k, random_state=Id, n_init=1, max_iter=10, init="random").fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1, max_iter=10, init="random").fit(X_tr)
    if train_mode=="f":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(
                n_clusters=k, 
                random_state=Id, 
                n_init=1).fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1).fit(X_tr)
    if train_mode=="d":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(n_clusters=k, random_state=Id).fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id).fit(X_tr)
    if train_mode=="med":
        kmeans=KMedoids(n_clusters=k, random_state=Id).fit(X_tr)  
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    pickle.dump(kmeans, open(model_path, "wb"))
    if not (train_mode in ["d", "f", "s"]):
        print("invalid train mode!")
    return X_tr, kmeans
    
def train_or_load_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, return_data=False, MINI_BATCH=False):
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("##loading trained model", model_path)
        kmeans=pickle.load(open(model_path, "rb"))
        if data is None:
            X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
            if cont>0:
                X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
                X_tr=np.concatenate((X_tr, X_cont))
            return X_tr, kmeans
        else:
            return data, kmeans
    else:
        print("##training a new model", model_path)
        X_tr, kmeans = train_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
        return X_tr, kmeans

def density_slopex1(x):
            d=np.zeros(shape=x.shape)
            d=(1-x[:, 0])*2
            for i in range(len(x[0])):
                d[x[:, i]<0]=0
                d[x[:, i]>1]=0
            return d

def k_means_process(dataset, #index/name of the dataset 
                    k_clusters=10, #number of clusters (k in k-means) 
                    SIGMA=3, #sigma for the smearing kernel
                    crop=100000, #crop the background training set (take first "crop" images)
                    cont=0, #add "cont" signal images into training set for contamination
                    preproc=None, #reweighting
                    REVERSE=False, # set true if signal and backgrouns should switch roles in the dataset
                    Id=0, #An Id o the training run, has an effect on random initialisation 
                    SCORE_TYPE="KNC", #Anomaly score 
                    knc=5, #number of nearest clusters if SCORE_TYPE="KNC"                                
                    characterise=False, #set true if characterisation should be done after training
                    SAVE_CHAR=True, #set True if the plots and characterisation data should be saved
                    train_mode="d",#explanation below
                    data=None, #training data that was alredy loaded and preprocessed by the prevous iteration of the algorythm and so can be reused
                    MINI_BATCH=False,
                    return_data=True, #return loaded training data for the use by the next iteration
                    density="", 
                    plot_dim=False):
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont
# really require the best and fully convergent clustering, repeating clustering
# with 10 initialisations is thus a waist of resources (it may be better to
# build an ensemble out of such 10 instead of picking one of 10 with best clustering)
# "d" - default as it is default in scikit
# "f" - fast (train only with one initialisation)
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation
    
    res={}  #a dictonary with most of the evaluation results
    plt.close("all") #close all figures of the previous run
    random.seed(a=10, version=2) #a random seed for all runs
    
    DI = dataset_path_and_pref(dataset, REVERSE) #DI stores the dataset info, such as path to the dataset and prefix associated with this dataaset
    
    if MINI_BATCH:
        DI["pref"]="MB"+DI["pref"]
        
    if cont>0:
        DI["pref"]=DI["pref"]+"+"+DI["pref2"]+"{:}".format(cont)+"_"
    
    hyp=(DI["pref"], k_clusters, SIGMA, crop, cont, preproc, Id) #hyperparameters for the training 
    
    X_tr, kmeans = train_or_load_k_means(DI, *hyp, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
    
    if characterise:
        X_bg_val=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        X_sg_val=prepare_data(DI["sg_val_data_path"], field=DI["sg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        
        if SCORE_TYPE=="MinD": #minimal distance
            postf="MinD"
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = np.min(bg_losses, 1)
            sg_scores = np.min(sg_losses, 1)
            tr_scores = np.min(tr_losses, 1)
    
        if SCORE_TYPE=="KNC": #KNC score
            postf="KNC"+str(knc)
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = mean_knc_mins(bg_losses, knc)
            sg_scores = mean_knc_mins(sg_losses, knc)
            tr_scores = mean_knc_mins(tr_losses, knc)
            
        elif SCORE_TYPE=="logLds":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, d_slopes_norm, log_likelyhood=True, res=res, plot_dim=plot_dim)
        
        elif SCORE_TYPE=="Lds":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, d_slopes_norm, log_likelyhood=False, res=res, plot_dim=plot_dim)

        elif SCORE_TYPE=="logLrh0":
            postf=SCORE_TYPE
            new_dist=partial(d_slopes_norm, dont_use_volume=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, new_dist, log_likelyhood=True, res=res, d=1)
        
        elif SCORE_TYPE=="Lrh0":
            postf=SCORE_TYPE
            new_dist=partial(d_slopes_norm, dont_use_volume=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, new_dist, log_likelyhood=False, res=res, d=1)
            
        elif SCORE_TYPE=="Lrhn":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, d_slopes_norm, log_likelyhood=False, res=res, d=1)
            
        elif SCORE_TYPE=="logLrhn":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k_clusters, X_tr, X_bg_val, X_sg_val, d_slopes_norm, log_likelyhood=True, res=res, d=1)
    
    if characterise:
        
        # plot a loss_density corellation 
        if density!="":
            plt.figure(figsize=(5, 5))
            if density=="slopex1":
                d_tr=density_slopex1(X_tr)
                plt.ylabel("p(x)")

            if density=="slopex1log":
                d_tr=np.log(density_slopex1(X_tr))
                plt.ylabel("log(p(x))")
            
            if density=="Gaussian":
                d_tr=gaussian(X_tr, 0, 1, 1, 2)
                plt.ylabel("p(x)")
                
            if density=="Gaussianlog":
                d_tr=np.log(gaussian(X_tr, 0, 1, 1, 2))
                plt.ylabel("log(p(x))")
                
            
            corr=np.corrcoef(tr_scores, d_tr)
            plt.scatter(tr_scores, d_tr, s=0.5, label="Pearson correlation {:.3f}".format(corr[0, 1]))
            plt.xlabel(SCORE_TYPE)
            ax=plt.gca()
            leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False) 
        
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
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure()
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        plt.legend()
        
        counts=np.array([np.sum(kmeans.labels_==i) for i in range(k_clusters)])
        counts.sort()
                
        #%% Save results:
        if SAVE_CHAR:
            path="char/{:}+{:}m{:}s{:}c{:}r{:}KI{:}{:}{:}/".format(DI["pref"], DI["pref2"], k_clusters, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id, postf)
            os.makedirs(path, exist_ok=True)
            k=0
            if plot_dim and (SCORE_TYPE in ["logLds"]):
                k+=1
                plt.figure(k)
                plt.savefig(path+"dimensions.png", bbox_inches="tight")

            if density!="":
                k+=1
                plt.figure(k)
                plt.savefig(path+"correlation.png", bbox_inches="tight")

            k+=1
            plt.figure(k)
            plt.savefig(path+"ROC.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"dist.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"SIC.png", bbox_inches="tight")
        
            res["fpr"]=fpr
            res["tpr"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            print(path+"res.pickle")
        if return_data:
            return X_tr
    
if __name__ == "__main__":
    print("test_run")
    k_means_process(dataset="1h5",
                k_clusters=100,
                SIGMA=3,
                crop=100000,
                preproc=None, #reprocessing.reproc_4rt,
                Id=0,
                knc=5,                                 
                characterise=True,
                SAVE_CHAR=True,
                REVERSE=False,
                SCORE_TYPE="logLds",
                data=None,
                plot_dim=True)


