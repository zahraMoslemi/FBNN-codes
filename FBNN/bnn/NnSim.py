import numpy as np
import pickle
np.random.seed(2023)

class NnSim:
    """
    NN Problem
    -----------------------------------------------------
    likelihood: y ~ Bernoulli(p)
                p = Sigmoid(G(u, X))
                G(u, X) is a neural network with structure (3, 5, 5, 1)
    forward mapping: G(u, X), where u (56,)  X (1000, 3)
    output: y (1000,)
    prior: u = flatten(W1, W2, W3, b1, b2, b3) ~ N(0, I_56)
    posterior: unknown
    """
    def __init__(self,input_dim=3,output_dim=1000,nn=None,pr_cov=1.,**kwargs):
        """
        Initialization
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.nn=nn
        self.y=kwargs.pop('y', np.ones(self.output_dim))

        def sigmoid(x):
          return 1/(1 + np.exp(-x))

        if self.nn is None:
            self.X=kwargs.pop('X',np.ones((self.output_dim,self.input_dim))) # (m,d)
            self.nn=lambda u: sigmoid((np.maximum(np.maximum(self.X.dot(u[:15].reshape((3,5))) + u[45:50], 0).dot(u[15:40].reshape((5,5))) + u[50:55], 0).dot(u[40:45].reshape((5,1))) + u[55:]).flatten())


        self.pr_cov=pr_cov
        if np.ndim(self.pr_cov)<2 and np.size(self.pr_cov)<self.input_dim: self.pr_cov=np.resize(self.pr_cov, 56)

    
    def forward(self,input,geom_ord=0):
        """
        Forward mapping
        """
        if geom_ord==0:
            output=self.nn(input)
        return output

    
    def logpdf(self,input,type='likelihood',geom_ord=[0]):
        """
        Log probability density function and its gradient
        """
        fwdout=self.forward(input)
        epsilon = 1e-8
        if 0 in geom_ord: 
          loglik = np.sum(self.y * np.log(fwdout + epsilon) + (1 - self.y) * np.log(1 - fwdout + epsilon))
        if 1 in geom_ord:
          # need to change
            dfwdout=self.forward(input,1)
            dloglik=np.sum(((self.y-fwdout)/self.nz_var)[:,None]*dfwdout,axis=0)
            if np.ndim(input)==1:
                dlogpri=-input/self.pr_cov
                logpri=0.5*input.dot(dlogpri)

        elif type=='likelihood':
            logpri=0; dlogpri=0
        out=[]
        if 0 in geom_ord: out.append(loglik+logpri)
        if 1 in geom_ord: out.append(dloglik+dlogpri)
        return out
    
    def get_geom(self,input,geom_ord=[0],**kwargs):
        """
        Get geometric quantities of loglikelihood
        """
        loglik=None; gradlik=None; metact=None; rtmetact=None; eigs=None
        
        out=self.logpdf(input,geom_ord=geom_ord)
        if 0 in geom_ord: loglik=out[0]
        if 1 in geom_ord: gradlik=out[-1]
        
        if any(s>1 for s in geom_ord):
            print('Requested geometric quantity not provided yet!')
        
        if len(kwargs)==0:
            return loglik,gradlik,metact,rtmetact
        else:
            return loglik,gradlik,metact,eigs
    
    def sample(self,prng=np.random.RandomState(2023),num_samp=1,type='prior'):
        """
        Generate sample
        """
        samp=None
        if type=='prior':
            samp=np.sqrt(self.pr_cov)*prng.randn(num_samp,56)
        return np.squeeze(samp)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os,sys
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    sys.path.append( "../" )
    from util.common_colorbar import common_colorbar
    np.random.seed(2023)
    
    # set up
    folder='./result'
    print(os.path.join(folder, 'nn.pickle'))
    d=3; m=1000
    try:
      with open(os.path.join(folder, 'nn.pickle'), 'rb') as f:
          [pr_cov,X,y] = pickle.load(f)
      print('Data loaded!\n')
      print(X.shape, y.shape) 
      kwargs={'X':X,'y':y}
    except:
      print('No data found. Generate new data...\n')
      pr_cov=1.
      X_full, Y_full = make_classification(
        n_samples=2000,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2, 
        n_classes=2,
        random_state=42 
      )
      X_train, X_test, y_train, y_test = train_test_split(X_full, Y_full, test_size=0.5, random_state=42)


      kwargs={'X':X_train, 'y':y_train}
      nnsim=NnSim(d,m,pr_cov=pr_cov,**kwargs)
      print(nnsim.X.shape, nnsim.y.shape)  
      # save data
      if not os.path.exists('./result'): os.makedirs('./result')
      with open('./result/nn.pickle','wb') as f:
          pickle.dump([nnsim.pr_cov, X_train, y_train],f)
      with open('./result/test/nn.pickle','wb') as f:
          pickle.dump([nnsim.pr_cov, X_test, y_test],f)
