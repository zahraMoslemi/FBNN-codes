"""
Geometric functions by emulator emulation
"""

import numpy as np
import tensorflow as tf

def vec2img(v):
    gdim=2
    imsz = np.floor(v.size**(1./gdim)).astype('int')
    im_shape=(-1,)+(imsz,)*(gdim-1)
    return v.reshape(im_shape)

def geom(unknown,bip,emulator,geom_ord=[0],whitened=False,**kwargs):
    loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None

    if whitened:
        unknown=bip.prior['cov'].dot(unknown)
    
    u_input = {'DNN':unknown[None,:], 'CNN':vec2img(unknown)[None,:,:,None]}[type(emulator).__name__]
    
    epsilon = 1e-8
    ll_f = lambda x: tf.math.reduce_sum(bip.y[None,:] * np.log(emulator.model(x)+epsilon) + (1.0 - bip.y[None,:]) * np.log(1 - emulator.model(x)+epsilon), axis=1)

    
    if any(s>=0 for s in geom_ord):
        loglik = ll_f(u_input).numpy()
    
    if len(kwargs)==0:
        return loglik,gradlik,metact,rtmetact
    else:
        return loglik,gradlik,metact,eigs