import numpy as np
import math
#Multi-level thresholding image segmentation Cross Entropy
class MLTHIMS_CE(object):
    """
    Cross entropy function for multi-level thresholding image segmentation
    Created on  JUne 15  2022
    
    @author: Thaer Thaher
    
    % _____________________________________________________
    % Main paper:
    % 
    % 
    % 
    % _____________________________________________________
    
    """
    def __init__(self,hist, dim = 10):
        """
        Constructs a PSO algorithm
        USAGE: pygmo.problem(pygmo.cross_entropy(dim = 10))
        
        * dim (int) – problem dimension (number of thresholds)
        * hist (array) – INput image historgram
        """
        
        #We start defining the problem ’private’ data members
        self.__dim = dim
        self.__hist = hist

        
    # Reimplement the virtual method called fitness that defines the objective function.
    # used to return the fitness of the input decision vector
    def fitness(self,x):
        
        """ x is the input decision vector i.e., (candidate solution) """
        x.sort()
        f = self.cross_entropy(x, self.__hist)
        return [f]
        """
        f = 0;
        for i in range(self.__dim):
            f = f + (x[i])*(x[i])
            #note that we return a tuple with one element only. In PyGMO the objective functions
            #return tuples so that multi-objective optimization is also possible.
        return (f,)
        """
        
    ############################################
      
    # return the box bounds of the problem, , which also implicitly define the dimension of the problem   
    def get_bounds(self):
        
        return ([1] * self.__dim,[256] * self.__dim)
        
    
    def get_name(self):
        return "Cross Entropy"
    
    def get_extra_info(self):
        
        return "\n\t Problem dimension: " + str(self.__dim)
    
    
    def cross_entropy(self, x , h):
        """
        * x (array) – decision vector
        * h (array): Image histogram
        """
        st = len(x)
        nu = np.zeros(st+1)
    
        for i in range(st+1):
            #print(i)
            if i == 0:                 # First Part
                ti = x[i]
                ti_1 = 1
                #print(ti_1)
                #print(mEin(ti_1,ti,h))
                #nu(i) = 5
                #nu[i] = mEin(ti_1,ti,h)*(math.log(mEin(ti_1,ti,h)/mZero(ti_1,ti,h))); #cross entropy
            
            
            elif i > (st-1):           # Last Part
                ti = 256
                ti_1 = x[i-1]
                #print(ti_1)
                #print(mEin(ti_1,ti,h))
                #nu[i] = mEin(ti_1,ti,h)*(math.log(mEin(ti_1,ti,h)/mZero(ti_1,ti,h))); #cross entropy
            
            else:                     # Original
                ti = x[i]
                ti_1 = x[i-1]
                #print(ti_1)
                #print(mEin(ti_1,ti,h))
                #nu[i] = mEin(ti_1,ti,h)*(math.log(mEin(ti_1,ti,h)/mZero(ti_1,ti,h))); #cross entropy
                    
            try:
                nu[i] = self.mEin(round(ti_1),round(ti),h)*(math.log(self.mEin(round(ti_1),round(ti),h) / self.mZero(round(ti_1),round(ti),h))) #cross entropy
            except ZeroDivisionError:
                nu[i] = 0
            
            
        sumNU=np.sum(nu)
        
        if (np.isnan(sumNU)):        
            fit= float('inf') #-1*sumNU
        else:
            fit=-1*sumNU;    
        #print("fitness = ", fit)
        return fit


    def mEin(self,aa,bb,ih):
        bm_1 = bb-1
        #print(aa)
        #print(bm_1)
        dif_bm_1 = abs(aa-bb)
        #print(dif_bm_1)
        inten = np.linspace(aa,bm_1,dif_bm_1)
        #print("len of inten",len(inten))
        #print(len(inten))
        #inten = np.transpose(inten)
        #print(inten)
    
        H = ih[aa-1 : bm_1]
        #print("len of h", len(H))
        uu = np.multiply(inten,H)
        u = sum(uu)
    
        return u

    def mZero(self,aa,bb,ih):
        bm_1=bb-1
        h=ih[aa-1:bm_1]
        u=  sum(h)
    
        return u