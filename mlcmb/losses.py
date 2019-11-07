import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 
import utilities
import scipy
import numpy as np
import quicklens as ql
import config




#from quicklens maps.py
def get_lxly(nx, dx, ny=None, dy=None):
    if ny is None:
        ny = nx
        dy = dx
    """ returns the (lx, ly) pair associated with each Fourier mode. """
    return np.meshgrid( np.fft.fftfreq( nx, dx )[0:nx/2+1]*2.*np.pi,
                            np.fft.fftfreq( ny, dy )*2.*np.pi )  

#from quicklens maps.py
def get_ell(nx, dx, ny=None, dy=None):
    """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
    lx, ly = get_lxly(nx, dx)
    return np.sqrt(lx**2 + ly**2)


def get_pixel_transfer(nx,dx, ny=None, dy=None):  
    if ny is None:
        ny = nx
        dy = dx
    """ return the FFT describing the map-level transfer function for the pixelization of this object. """
    lx, ly = get_lxly(nx,dx)
    fft = np.zeros( lx.shape )
    fft[ 0, 0] = 1.0
    fft[ 0,1:] = np.sin(dx*lx[ 0,1:]/2.) / (dx * lx[0,1:] / 2.)
    fft[1:, 0] = np.sin(dy*ly[1:, 0]/2.) / (dy * ly[1:,0] / 2.)
    fft[1:,1:] = np.sin(dx*lx[1:,1:]/2.) * np.sin(dy*ly[1:,1:]/2.) / (dx * dy * lx[1:,1:] * ly[1:,1:] / 4.)
    return fft


#adapted from quicklens
def get_rfft(rmap, nx, dx):
    """ return an rfft array containing the real fourier transform of this map. """
    #from quicklens.maps.rmap.get_rfft()
    tfac = np.sqrt((dx * dx) / (nx * nx))
    rfft = tf.spectral.rfft2d(rmap) * tfac
    return rfft



def get_lxly_cfft(nx, dx, ny=None, dy=None):
    """ returns the (lx, ly) pair associated with each Fourier mode. """
    if ny is None:
        ny = nx
        dy = dx    
    return np.meshgrid( np.fft.fftfreq( nx, dx )*2.*np.pi,
                        np.fft.fftfreq( ny, dy )*2.*np.pi )


def get_ell_cfft(nx, dx, ny=None, dy=None):
    """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """ 
    lx, ly = get_lxly_cfft(nx, dx)
    return np.sqrt(lx**2 + ly**2)


def get_cfft(rmap, nx, dx):
    """ return the complex FFT. """    
#     rfft = get_rfft(rmap, nx, dx)
#     cfft = np.zeros( (nx, nx), dtype=np.complex )
#     cfft[:,0:(nx/2+1)] = rfft[:,:]
#     cfft[0,(nx/2+1):]  = np.conj(rfft[0,1:nx/2][::-1])
#     cfft[1:,(nx/2+1):]  = np.conj(rfft[1:,1:nx/2][::-1,::-1])    
    tfac = np.sqrt((dx * dx) / (nx * nx))
    rmap = tf.cast(rmap,tf.complex64)
    cfft = tf.spectral.fft2d(rmap) * tfac
    return cfft



#adapted from quicklens
def get_rmap(rfft, nx, dx):
    #from quicklens.maps.rfft.get_rmap()
    """ return the rmap given by this FFT. """
    tfac = np.sqrt((nx * nx) / (dx * dx))
    rmap = tf.spectral.irfft2d(rfft)*tfac    
    return rmap

def get_modepowers(y,nx,dx):

    #real space map, defined by rmap, nx, dx
    rmap = y[:,:,:,0] #need to strip last dimension, because fft2d acts on last two channels

    #fft
    rfft = get_rfft(rmap, nx, dx)
    rfft_shape = rfft.get_shape().as_list()

    #power of difference map
    power = tf.real((rfft * tf.conj(rfft))) #tf.math.conj in higher versions
    power = tf.reshape(power,[-1,rfft_shape[1]*rfft_shape[2]]) #flatten except batch dimension

    return power


#based on https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py tqumap.get_teb(self)
def get_ebfft(qmap,umap, nx, dx):
    """ return e b containing the fourier transform of the Q,U maps. """

    lx, ly = get_lxly(nx,dx)
    tpi  = 2.*np.arctan2(lx, -ly)

    tfac = np.sqrt((dx * dx) / (nx * nx))
    qfft = tf.spectral.rfft2d(qmap) * tfac
    ufft = tf.spectral.rfft2d(umap) * tfac

    efft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
    bfft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
    return efft,bfft


#https://github.com/dhanson/quicklens/blob/master/quicklens/maps.py tebfft.get_tqu
def get_qumaps(efft,bfft,nx,dx):
    """ returns the tqumap given by the inverse Fourier transform of this object. """
    lx, ly = get_lxly(nx,dx)
    tpi  = 2.*np.arctan2(lx, -ly)
    tfac = np.sqrt((nx * nx) / (dx * dx))
    qmap = tf.spectral.irfft2d(np.cos(tpi)*efft - np.sin(tpi)*bfft) * tfac
    umap = tf.spectral.irfft2d(np.sin(tpi)*efft + np.cos(tpi)*bfft) * tfac
    return qmap, umap



def ell_filter(y,nx,dx):
    #real space map, defined by rmap, nx, dx
    rmap = y[:,:,:,0]    

    #fft
    rfft = get_rfft(rmap, self.params.nx, self.params.dx)

    #multiply tensor by a k-space mask
    mask = powerspectra.ellmask_nonzero_nonflat.astype(int)
    rfft = rfft*mask

    #get back real space map
    rmap = get_rmap(rfft, nx, dx)
    rmap = tf.expand_dims(rmap,axis=-1) #reintroduce the channel dimension

    return rmap




class Lossfunctions():
    
    def __init__(self,params):
        self.params = params  


    def loss_pixelMSE_ellfiltered(self,y_true, y_pred):
        rmap_true = y_true
        rmap_pred = ell_filter(y_pred,self.params.nx,self.params.dx)

        loss = tf.reduce_mean((rmap_true-rmap_pred)*(rmap_true-rmap_pred))     
        return loss


    def loss_pixelMSE_unfiltered(self,y_true, y_pred):
        rmap_true = y_true
        rmap_pred = y_pred

        loss = tf.reduce_mean((rmap_true-rmap_pred)*(rmap_true-rmap_pred))     
        return loss


