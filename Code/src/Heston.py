

import numpy as np
from src.BlackScholes import BSModel


class HestonModel:
    
    def __init__(self,
                 rf,     # annual risk free rate
                 kappa,  # variance return rate
                 vbar,  # long term variance
                 xi,     # volatility of the volatility
                 rho    # correlation between two Brownian motions
    ):
    
        self.rf = rf
        self.kappa = kappa
        self.vbar = vbar
        self.xi = xi
        self.rho = rho
        self.FellerConditionSatisfied = 2 * kappa * vbar > xi**2
        
        # Parameters for European Call Pricing
        self.BSmodel = BSModel(rf)
        self.BS = lambda tau, sigma, s, k : self.BSmodel.PriceEuropean(np.exp(s), sigma, tau, np.exp(k))
     
        
        
    def PriceEuropean(self, S, v, tau, K, type=1):
        
        # S     : current underlying price 
        # v     : current variance
        # tau   : time to maturity tau = T - t, unit year
        # K     : strike
        # type  : 'call' for 1 or 'put' for -1
        
        s = np.log(S)
        k = np.log(K)
        
        rf = self.rf
        kappa = self.kappa
        vbar = self.vbar
        xi = self.xi
        rho = self.rho
        kappa_tau = kappa * tau
        vbar_kappa_tau = vbar * kappa * tau
        ekt = np.exp(-kappa_tau)
        
        sigma_hat  = np.sqrt(vbar + (v - vbar)/kappa_tau * (1-ekt))
        std_hat = sigma_hat * np.sqrt(tau)
        d1 = ( np.log(S/K) + tau * ( rf + sigma_hat**2/2 ) ) / std_hat
        
        U = (rho**2 * xi) / (2*kappa**2) * ( vbar_kappa_tau - 2*vbar + v + 
                                         ekt*(vbar_kappa_tau + 2*vbar - v - v*kappa_tau) ) 
        
        R = xi**2 / (16*kappa**3) * ( 2*vbar_kappa_tau + 2*v - 5*vbar + 
                                      4*ekt*(vbar_kappa_tau + vbar - v*kappa_tau) +
                                      ekt**2*(vbar-2*v) ) 
        
        H = np.exp(s-d1**2/2) / (np.sqrt(2*np.pi) * std_hat) * (1-d1/std_hat)
        L = np.exp(s-d1**2/2) / (np.sqrt(2*np.pi) * std_hat) * (d1**2-std_hat*d1-1)/(std_hat**2)
        
        CallPrice = self.BS(tau, sigma_hat, s, k) + U * H + R * L
        
        if type == 1:
            return CallPrice
        elif type == -1:
            return K * np.exp(-rf*tau) - S + CallPrice
        else :
            raise ValueError("Wrong type given.")
        
        
        
    def getImpliedVol(self, MarketOptionPrice, S, tau, K, type=1):
        return self.BSmodel.getImpliedVol(MarketOptionPrice, S, tau, K, type)
    
    
    
    def PathSimulation(self, S0, v0, tau, dt=1/365):
        
        rf = self.rf
        kappa = self.kappa
        vbar = self.vbar
        xi = self.xi
        rho = self.rho
        N_steps = round(tau/dt)
        
        S = [S0]
        v = [v0]
        t = 1
        
        while t <= N_steps:
            
            Z1 = np.random.normal(0, 1)
            Zv = np.random.normal(0, 1)
            ZS = np.sqrt(1-rho**2) * Z1 + rho * Zv
            
            S_previous , v_previous = S[t-1] , v[t-1]
            
            vt = v_previous + kappa * (vbar-v_previous) * dt + xi * np.sqrt(v_previous*dt) * Zv + xi**2*dt*(Zv**2-1)/4
            vt = max(vt,-vt)
            
            St = S_previous + rf*S_previous*dt + np.sqrt(v_previous*dt) * S_previous * ZS \
                            +  S_previous*v_previous*dt*(ZS**2-1)/2
                            
            S.append(St)
            v.append(vt)            
            t+=1
            
        return (S,v)
        
        
    
    