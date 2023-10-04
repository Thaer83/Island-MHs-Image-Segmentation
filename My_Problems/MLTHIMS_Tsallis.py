import numpy as np
import math
#Multi-level thresholding image segmentation Cross Entropy
class MLTHIMS_Tsallis(object):
    """
    tsallis AS OBJ FUNC: for multi-level thresholding image segmentation (RGB/GRAY IMAGES)
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
        * ProbR: grayscale image (probR(i) = n_countR(i) / Totla_PIxels)
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
        f = self.tsallis(x, self.__dim, self.__probR)
        return [f]
        
    ############################################
      
    # return the box bounds of the problem, , which also implicitly define the dimension of the problem   
    def get_bounds(self):
        
        return ([1] * self.__dim,[256] * self.__dim)
        
    
    def get_name(self):
        return "OTSU'S BETWEEN CLASS VARIANCE"
    
    def get_extra_info(self):
        
        return "\n\t Problem dimension: " + str(self.__dim)
    
    def tsallis(self, u,level,probR):
        sum2 = np.zeros(level+1)
        q=0.5;
        p1=sum(probR[1:u[0]]);
        n1=(probR/p1)**q;
        sum1=sum(n1[1:u[0]]);
        sum2[0]=(1-sum1)/(q-1);
        
        for jlevel in range(1, level):
            p2=sum(probR[u[jlevel-1]:u[jlevel]]);
            n2=(probR/p2)**q;
            sum1=sum(n2[u[jlevel-1]:u[jlevel]]);
            sum2[jlevel]=((1-sum1)/(q-1));
    
    
        pe=sum(probR[u[level-1]:255]);
        ne=(probR/pe)**q;
        sum1=sum(ne[u[level-1]:255]);
        sum2[jlevel+1]=((1-sum1)/(q-1));
        sumfinal=sum(sum2);
        prodfinal=sum2.prod();
        sumprofinal=sumfinal+(1-q)*prodfinal;
    
    
        if (np.isnan(sumprofinal)):        
            fit= float('-inf')
        else:
            fit= sumprofinal 
                  
        fit=-1 * fit
        
        return fit
