import numpy as np 
import copy
from utilities import d_ball_volume, dm1_sphere_area
import matplotlib
import matplotlib.pyplot as plt
import set_matplotlib_default as smd

def infinity_to_min_max(bg_scores, sg_scores, tr_scores):
    max_score=max(np.max(tr_scores[np.isfinite(tr_scores)],initial=0), np.max(sg_scores[np.isfinite(sg_scores)],initial=0), np.max(bg_scores[np.isfinite(bg_scores)],initial=0))
    min_score=min(np.min(tr_scores[np.isfinite(tr_scores)],initial=0), np.min(sg_scores[np.isfinite(sg_scores)],initial=0), np.min(bg_scores[np.isfinite(bg_scores)],initial=0))
    #replacing inf with max*1.1 (if max>0) has no effect on the tagging performance (inf was 0 befor -log)
    #replacing inf with max*0.9 (if max<0) has no effect on the tagging performance (inf was 0 befor -log)
    #replacing -inf with min*0.9 (if min>0) has no effect on the tagging performance
    #replacing -inf with min*1.1 (if max>0) has no effect on the tagging performance
    if max_score>0:   
        bg_scores[bg_scores==np.inf]=max_score*1.1
        sg_scores[sg_scores==np.inf]=max_score*1.1
        tr_scores[tr_scores==np.inf]=max_score*1.1
    else:
        bg_scores[bg_scores==np.inf]=max_score*0.9
        sg_scores[sg_scores==np.inf]=max_score*0.9
        tr_scores[tr_scores==np.inf]=max_score*0.9
    if min_score>0:
        bg_scores[bg_scores==np.NINF]=min_score*0.9
        sg_scores[sg_scores==np.NINF]=min_score*0.9
        tr_scores[tr_scores==np.NINF]=min_score*0.9
    else:
        bg_scores[bg_scores==np.NINF]=min_score*1.1
        sg_scores[sg_scores==np.NINF]=min_score*1.1
        tr_scores[tr_scores==np.NINF]=min_score*1.1

def estimate_dim_uniform_point(dist_tr, cluster_i, r_i, c=1.1):
    dists=dist_tr[:, cluster_i]
    N_1=np.sum([dists<=r_i])
    N_2=np.sum([dists<=r_i*c])
    dim=np.log(N_2/N_1)/np.log(c)
    return dim 
    
def estiamte_dim_uniform(dist_tr, cluster_i, r_i, c=1.1, plotting=False):
    dists=dist_tr[:, cluster_i]
    min_d=np.min(dists)
    max_d=np.max(dists) 
    c_plot=1.01    
    r_arr=min_d*c_plot**np.arange(0, (int)(np.floor(np.log(max_d/(min_d+10**-10))/np.log(c_plot)))+3)
    dims=[]
    for r in r_arr:
        dims.append(estimate_dim_uniform_point(dist_tr, cluster_i, r, c=1.1))
    dims=np.array(dims)
    if plotting:
        ax=plt.gca()
        vline_color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(r_arr, dims, c=vline_color)
        plt.axvline(r_i, color=vline_color)
    return estimate_dim_uniform_point(dist_tr, cluster_i, r_i, c=1.1)

def likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, density_function, log_likelyhood=True, res=None, plot_dim=True, d=None):
    
    # find all the point-to-cluster distances
    dist_tr = kmeans.transform(X_tr)
    dist_bg_val = kmeans.transform(X_bg_val)
    dist_sg_val = kmeans.transform(X_sg_val)
    
    #initialise a plot for illustrating the dimension estimation for clusters
    if plot_dim:
        matplotlib.rcParams.update({'font.size': 14})
        plt.figure(figsize=(8, 5))
        plt.xlabel("$R$")
        plt.xscale("log")
        plt.ylabel("$d_i(R)$")
        axs=plt.gca()
        axs.grid( which='both', alpha=0.5 )
        
    means=[]    #list of rho_i of the clusters
    sigmas=[]   #list of sigma_i of the clusters
    weights=[]  #list of N_i/N_tot of the clusters
    dims=[]     #list of dim_i of each cluster
    for i in range(k):
        dist=dist_tr[kmeans.labels_==i, i] #find distances to cluster i of points assigned to cluster i
        means.append(np.mean(dist))  #calculate rho_i
        sigmas.append(np.std(dist))  #calculate sigma_i
        weights.append(len(dist)/crop) #calculate N_i/N_tot
        if d==None: #If d is not set by user find the effective dimensions d_i for each cluster
            if plot_dim:
                if i<5:
                    dims.append(estiamte_dim_uniform(dist_tr, i, means[-1], c=1.1, plotting=True))
                else:
                    dims.append(estiamte_dim_uniform(dist_tr, i, means[-1], c=1.1))
            else:
                dims.append(estiamte_dim_uniform(dist_tr, i, means[-1], c=1.1))
    
    #Transform lists into arrays
    means=np.array(means)
    sigmas=np.array(sigmas)
    weights=np.array(weights)
    dims=np.array(dims)

    #pint out some results
    print("means")
    print(means)
    print("sigmas")
    print(sigmas)
    print("dims")
    print(dims)
    
    if d==None:
        d=np.median(dims) #If d is not set by user find the effective dimension d for the dataset        
    print("d=", d)
    
    # Some tests 
    # TODO: delete this part
    N_0=d_ball_volume(d, means) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, means)*sigmas*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    rat=N_1/N_0
    print("ratios", rat.sort())
    print("max ratios =", np.max(rat))
    print("max sig/mu =", np.max(sigmas/means))
    
    if res!=None: #put some results in the res dictionary to use later
        res["means"]=means
        res["sigmas"]=sigmas
        res["dims"]=dims
        res["d_med"]=np.median(dims)
        res["d_mean"]=np.mean(dims)
        res["mu_max"]=np.max(means)
        res["mu_mean"]=np.mean(means)
        res["mu_median"]=np.median(means)
        res["mu_min"]=np.min(means)
        res["sig_max"]=np.max(sigmas)
        res["sig_min"]=np.min(sigmas)
        res["sig/mu_max"]=np.max(sigmas/means)
        res["V_0/V_1_max"]=np.max(rat)

    #define arrays for partial likelyhoods
    part_L_bg=np.zeros(dist_bg_val.shape) 
    part_L_sg=np.zeros(dist_sg_val.shape)
    part_L_tr=np.zeros(dist_tr.shape)
    bg_L, sg_L, tr_L=0, 0, 0
    for i in range(k):
        #calcualte partial likelyhoods for each cluster
        part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i], d)
        part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i], d)
        part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i], d)
        #Subtract a prtial likelyhood from a total negative likelyhood
        bg_L-=part_L_bg[:, i]
        sg_L-=part_L_sg[:, i]
        tr_L-=part_L_tr[:, i]
        
    if log_likelyhood: #if one wants to use negative log likelyhood as a score
        # calculate the negative partial log likelyhoods (useful for analyesis)
        tr_losses = -np.log(part_L_tr)
        bg_losses = -np.log(part_L_bg)
        sg_losses = -np.log(part_L_sg)
        # Calculate the negative log likelyhoods
        bg_scores = -np.log(-bg_L)
        sg_scores = -np.log(-sg_L)
        tr_scores = -np.log(-tr_L)
        #resolve isues with infinities arrising from log(0)
        infinity_to_min_max(bg_scores, sg_scores, tr_scores)
        infinity_to_min_max(bg_losses, sg_losses, tr_losses)
    else: #if one wants to use negative likelyhood as a score
        tr_losses = part_L_tr
        bg_losses = part_L_bg
        sg_losses = part_L_sg
        bg_scores = bg_L
        sg_scores = sg_L
        tr_scores = tr_L
    #Both scores give the same tagging results as they are connected by a monotonic log function
    return tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses