import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PMF:
    """
    The PMF model class
    """

    def __init__(self, X=None, sigma=None, G=None, F=None, E=None, Q=None,
                 Qbar=None, nprofile=None):
        """
        Initialization suff
        """
        self.X = X
        
        nsp, nobs = X.shape

        self.sigma = sigma
        self.G = np.random.randn(nprofile, nobs)
        self.F = np.random.randn(nsp, nprofile)
        self.G[self.G<0]=0
        self.F[self.F<0]=0
        self.E = np.random.randn(*X.shape)
        self.Q = np.random.randn(*X.shape)
        self.Qbar = np.zeros_like(X.shape)
        
        self.tol = 0.00001
        
        self.gradF = self.F @ (self.G @ self.G.T) - self.X @ self.G.T
        self.gradG = (self.F.T @ self.F) @ self.G - self.F.T @ self.X
        
        self.tolF = self.tolG = max(0.00001,self.tol)*self.normGrad();
        self.initgrad = self.normGrad()

    def compute_Qbar(self):
        """

        """
        alpha = beta = gamma = delta = 1
        Qbar = self.Q\
                - alpha * np.log(self.G).sum().sum()\
                - beta * np.log(self.F).sum().sum()\
                + gamma * (self.G**2).sum().sum()\
                + delta * (self.F**2).sum().sum()
        return Qbar
    
    def normGrad(self):
        tmp = np.vstack((self.gradF, self.gradG.T))
        return np.linalg.norm(tmp, ord="fro")

    #def gradF(self):
    #    f = self.F @ (self.G @ self.G.T) - self.X @ self.G.T
    #    return f

    #def gradG(self):
    #    g = (self.F.T @ self.F) @ self.G - self.F.T @ self.X
    #    return g


    def run(self):
        """
        Run the PMF model
        """

        def nlssubprob(V=None, W=None, Hinit=None, tol=None, maxiter=None):
            """
            H, grad: output solution and gradient
            iter: #iterations used
            V, W: constant matrices
            Hinit: initial solution
            tol: stopping tolerance
            maxiter: limit of iterations
            """
            H = Hinit
            WtV = W.T @ V
            WtW = W.T @ W 
            
            alpha = 1
            beta = 0.1
            
            #print(WtW, H, WtV, sep="\n")
            grad = WtW @ H - WtV
            #print(grad)
            for iter in range(maxiter):
                grad = WtW @ H - WtV
                # projgrad = np.linalg.norm(grad(grad < 0 | H >0));
                gradtmp = grad[(grad < 0) + (H < 0)] 
                projgrad = np.linalg.norm(grad);
                if projgrad < tol:
                    break
            
                # search step size 
                for inner_iter in range(0,20):
                    #print("H: ", H)
                    #print("grad: ", grad)
                    Hn = H - alpha * grad
                    Hn[Hn<0]= 0
                    #print("Hn: ", Hn)
                    d = Hn - H
                    gradd = np.sum(np.sum( grad * d ))
                    #print("gradd :", gradd)
                    dQd = np.sum(np.sum((WtW @ d) * d ))
                    suff_decr = (0.99 * gradd + 0.5*dQd) < 0
                    #print("suff_decr: ",suff_decr)
                    if inner_iter == 0:
                        decr_alpha = not suff_decr
                        Hp = H
                    #print("Hp: ", Hp)
                    if decr_alpha:
                        if suff_decr:
                            H = Hn
                            break
                        else:
                            alpha = alpha * beta
                    else:
                        if not suff_decr or (Hp==Hn).all():
                            H = Hp
                            break
                        else:
                            alpha = alpha/beta
                            Hp = Hn


            if iter==maxiter:
                print('Max iter in nlssubprob\n');
            return (H, grad, iter)
            #end of nlssubprob function
        maxiter = 1000
        for iter in range(maxiter):
            # stopping condition
            # gradF = self.gradF()
            # gradG = self.gradG()
            # tmp = np.vstack((gradF,gradG))
            projnorm = self.normGrad() #np.linalg.norm(tmp)
            print("iter :", iter)
            if projnorm < self.tol * self.initgrad:
                break
            
            print("F stuff")
            F, gradF, iterF = nlssubprob(self.X.T, 
                                          self.G.T,
                                          self.F.T,
                                          self.tolF,
                                          1000)
            self.F = F.T
            self.gradF = gradF.T
            if iterF == 0:
                self.tolF = 0.1 * self.tolF
            
            print("G stuff")
            G, gradG, iterG = nlssubprob(self.X,
                                         self.F,
                                         self.G,
                                         self.tolG,
                                         1000)
            self.G = G
            self.gradG = gradG
            if iterG == 0:
                self.tolG = 0.1 * self.tolG

