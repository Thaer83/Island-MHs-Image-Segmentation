import numpy as np
import math
#Multi-level thresholding image segmentation Cross Entropy
class MLTHIMS_Otsu(object):
    """
    OTSU'S BETWEEN CLASS VARIANCE AS OBJ FUNC: for multi-level thresholding image segmentation (RGB/GRAY IMAGES)
    Created on  JUne 26  2022
    
    @author: Thaer Thaher
    
    % _____________________________________________________
    % Main paper:
    % 
    % 
    % 
    % _____________________________________________________
    
    """
    def __init__(self,hist, probR, dim = 10):
        """
        USAGE: pygmo.problem(pygmo.MLTHIMS_Otsu(dim = 10))
        
        * dim (int) – problem dimension (number of thresholds)
        * hist (array) – INput image historgram
        """
        
        #We start defining the problem ’private’ data members
        self.__dim = dim
        self.__hist = hist
        self.__probR = probR

        
    # Reimplement the virtual method called fitness that defines the objective function.
    # used to return the fitness of the input decision vector
    def fitness(self,x):
        
        """ x is the input decision vector i.e., (candidate solution) """
        x.sort()
        #print("before", x)
        x = [round(num) for num in x]
        #print("after", x)
        f = self.Otsu(x, self.__dim, self.__probR)
        return [f]
        
    ############################################
      
    # return the box bounds of the problem, , which also implicitly define the dimension of the problem   
    def get_bounds(self):
        
        return ([1] * self.__dim,[256] * self.__dim)
        
    
    def get_name(self):
        return "OTSU'S BETWEEN CLASS VARIANCE"
    
    def get_extra_info(self):
        
        return "\n\t Problem dimension: " + str(self.__dim)
    
    
    def Otsu(self,u,level,probR):
        """
        * u (array) – decision vector
        * (level): Number of thresholds
        * ProbR: %grayscale image (probR(i) = n_countR(i) / Totla_PIxels)
        """
        y = sum(range(1,u[0]) * probR[1:u[0]]/sum(probR[1:u[0]]))
        x = y - sum(range(1,255) * probR[1:255])
        fitR=sum(probR[1:u[0]])*(x)**2
        
        for jlevel in range(1, level):
            xx = sum(range(u[jlevel-1], u[jlevel]) * probR[u[jlevel-1]:u[jlevel]]/sum(probR[u[jlevel-1]:u[jlevel]]))- sum(range(1,255) * probR[1:255])
            fitR = fitR + sum(probR[u[jlevel-1]:u[jlevel]]) * (xx)**2
        
        y = sum ((range(u[level-1],255)) * probR[u[level-1]:255]/sum(probR[u[level-1]:255]))
        x = y - sum(range(1,255) * probR[1:255])
        fitR=fitR + sum(probR[u[level-1]:255]) * (x)**2;
        
        
        if (np.isnan(fitR)):        
            fit= float('-inf')
        else:
            fit= fitR 
                  
        fit= -1 * fit
        
        return fit