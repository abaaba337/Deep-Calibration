
import numpy as np
from scipy.special import ndtr
from scipy import optimize

class BSModel:
    
    def __init__(self, rf):
    
        self.rf = rf  # annual risk free rate

    def PriceEuropean(self, S, sigma, tau, K, type=1):
        
        # S     : current underlying price 
        # sigma : constant volatility
        # tau   : time to maturity tau = T - t, unit year
        # K     : strike
        # type  : 'call' for 1 or 'put' for -1
        
        rf = self.rf
        std = sigma * np.sqrt(tau)
        discount = np.exp(-rf*tau)
        
        d1 = ( np.log(S/K) + tau * ( rf + sigma**2/2 ) ) / std
        d2 = d1 - std
        
        CallPrice = S * ndtr(d1) - K * discount * ndtr(d2)
        
        if type == 1:
            return CallPrice
        elif type == -1:
            return K * discount - S + CallPrice
        else :
            raise ValueError("Wrong type given.")
        
    
    def prior_func(self, sigma, S, tau, K, type, option_price):
        return self.PriceEuropean(S, sigma, tau, K, type) - option_price
    
    def getImpliedVol(self, option_price, S, tau, K, type=1):
        implied_vol = optimize.root_scalar(self.prior_func, bracket=[0.001, 10], args=(S, tau, K, type, option_price))
        return implied_vol.root
        
                