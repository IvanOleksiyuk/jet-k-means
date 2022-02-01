import numpy as np

def calorimeter_image(event, 
                      IMG_SIZE=40, 
                      phi_bonds=(-0.8, 0.8),
                      eta_bonds=(-0.8, 0.8),
                      obj_num=200):
    """

    Parameters
    ----------
    event : list/array of 4*obj_num floats  
        [E_1, px_1, py_1, pz_1, ... E_n, px_n, py_n, pz_n] 4-vectors of obj_num constituents of the jet.
    IMG_SIZE : int
        Length and width of the jet image in pixels. The default is 40.
    phi_bonds : tuple of floats
        Horizontal bounds for pixelisation, usually (-R, R). The default is (-0.8, 0.8).
    eta_bonds : tuple of floats
        Vertical bounds for pixelisation, usually (-R, R). The default is (-0.8, 0.8).
    obj_num : int
        Number of jet constituents. The default is 200.

    Returns
    -------
    TYPE
        Normalised jet image as a IMG_SIZE*IMG_SIZE numpy array.

    """
    
    #calculate pT eta and phi
    end=obj_num*4
    px=event[1:end:4]
    py=event[2:end:4]
    pz=event[3:end:4]
    pT=(px**2+py**2)**0.5
    phi=np.arctan2(py, px)
    P=(px**2+py**2+pz**2)**0.5
    #theta=np.arctan2(pT, pz)
    eta=-0.5*np.log((P+pz)/(P-pz))
    eta[eta==np.inf]=0
    eta[eta==-np.inf]=0
    eta=np.nan_to_num(eta)
    
    # substract phi and eta of the hardest objectst from other:
    phi=phi-phi[0]
    eta=eta-eta[0]
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)

    phi-=phi_centroid
    eta-=eta_centroid
    
    phi[phi<-np.pi]+=2*np.pi
    phi[phi>np.pi]-=2*np.pi
    
    #calculate the pT centroid
    phi_centroid=np.sum(pT*phi)/np.sum(pT)
    eta_centroid=np.sum(pT*eta)/np.sum(pT)
    phi-=phi_centroid
    eta-=eta_centroid
    
    #calculate the moment of inertia tensor
    I_xx=np.sum(pT*(phi)**2)
    I_yy=np.sum(pT*eta**2)
    I_xy=np.sum(pT*eta*phi)
    I=np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #calculate the major principal axis
    w, v=np.linalg.eig(I)
    if(w[0]>w[1]):
        major=0
    else:
        major=1

    #turn the immage 
    alpha=-np.arctan2(v[1, major], v[0, major])
    phi_new=phi*np.cos(alpha)-eta*np.sin(alpha)
    eta_new=phi*np.sin(alpha)+eta*np.cos(alpha)
    phi=phi_new
    eta=eta_new
    
    #flip the image according to the largest constituent
    q1=sum(pT[(phi>0)*(eta>0)])
    q2=sum(pT[(phi<=0)*(eta>0)])
    q3=sum(pT[(phi<=0)*(eta<=0)])
    q4=sum(pT[(phi>0)*(eta<=0)])
    indx=np.argmax([q1, q2, q3, q4])
    if indx==1:
        phi*=-1
    elif indx==2:
        phi*=-1
        eta*=-1
    elif indx==3:
        eta*=-1
       
    #create a calorimeter picture (pixelation)
    image=np.histogram2d(phi, eta, IMG_SIZE, [phi_bonds, eta_bonds], weights=pT)
    image=image[0]
    
    #small check of the image
    if np.sum(image)==0:
        raise NameError('Image is 0')
    if np.sum([image>0])>200:
        raise NameError('Too many non-zero pixels')
        
    return image/np.sum(image)
