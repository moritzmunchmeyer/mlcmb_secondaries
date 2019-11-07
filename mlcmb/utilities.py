import numpy as np
import quicklens as ql
import scipy
import config

r2d = 180./np.pi
d2r = np.pi/180.


#pass array of form [img_id,:,:,channel], return same array normalized channel wise, and also return variances
def normalize_channelwise(images):
    
#     #remove mean per image
#     for img_id in range(images.shape[0]):
#         for channel_id in range(images.shape[-1]):     
#             avg = (images[img_id,:,:,channel_id]).sum() / images[img_id,:,:,channel_id].size 
#             images[img_id,:,:,channel_id] = images[img_id,:,:,channel_id]-avg
       
    #calculate variance over all images per channel
    variances = np.zeros(images.shape[-1])
    for channel_id in range(images.shape[-1]): 
        if len(images.shape) == 4:
            variances[channel_id] = (images[:,:,:,channel_id]*images[:,:,:,channel_id]).sum() / images[:,:,:,channel_id].size 
            images[:,:,:,channel_id] = (images[:,:,:,channel_id])/variances[channel_id]**(1./2.) 
        if len(images.shape) == 3:
            variances[channel_id] = (images[:,:,channel_id]*images[:,:,channel_id]).sum() / images[:,:,channel_id].size 
            images[:,:,channel_id] = (images[:,:,channel_id])/variances[channel_id]**(1./2.)             
    return images,variances




def ell_filter_maps(maps, nx, dx, lmax, lmin=0):
    nsims = maps.shape[0]
    
    ell_filter = np.ones(10000)   #itlib.lib_qlm.ellmax=5133 for some reason
    ell_filter[lmax:] = 0 #3500
    ell_filter[0:lmin] = 0
    
    for map_id in range(nsims): 
        fullmap_cfft = ql.maps.rmap(nx, dx,map=maps[map_id]).get_cfft()
        filteredmap_cfft = fullmap_cfft * ell_filter
        filteredmap_cfft.fft[0,0] = 0.
        filteredmap = filteredmap_cfft.get_rffts()[0].get_rmap().map
        maps[map_id] = filteredmap
    
    return maps
        
        

def estimate_ps(maps, binnr=30, lmin=2, lmax=3000):
    nmaps = maps.shape[0]
    lbins      = np.linspace(lmin, lmax, binnr)       
    ell_binned = lbins[:-1] + np.diff(lbins)
    power_avg = np.zeros(ell_binned.shape[0])        
    for map_id in range(nmaps): 
        rmap = maps[map_id,:,:]     
        cfft = ql.maps.rmap(config.nx, config.dx,map=rmap).get_cfft()
        power = cfft.get_cl(lbins)
        power_avg += power.cl.real
    power_avg = power_avg/nmaps
    return ell_binned, power_avg
    
    

#periodic padding for image array (img_id,x,y,channels)
def periodic_padding(images,npad):
    if len(images.shape)==4:
        images = np.pad(images,pad_width=((0,0),(npad,npad),(npad,npad),(0,0)),mode='wrap')
    if len(images.shape)==3:
        images = np.pad(images,pad_width=((npad,npad),(npad,npad),(0,0)),mode='wrap')        
    return images



    
        
        
        
