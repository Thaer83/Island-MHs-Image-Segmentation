import numpy as np
import math
import My_Problems.math_functions_formulas as formulas
#Multi-level thresholding image segmentation Cross Entropy
class Math_Benchmarks(object):
    """
    F1
    Created on  JUne 26  2022
    
    @author: Thaer Thaher
    
   
    """
    #def __init__(self, lb=-100, ub=100, dim = 10, f_name):
    def __init__(self, func_details ):

        """
        USAGE: pygmo.problem(pygmo.MLTHIMS_Otsu(dim = 10))
        
        * dim (int) – problem dimension (number of decision variable )
        * lb (float) – Lower bound for the decision variable
        * lb (float) – Upper bound for the decision variable
        * f_name (string) - Name of the mathebnechmark Function
        """
        
        #We start defining the problem ’private’ data members
        #self.__dim = dim
        #self.__lb = lb
        #self.__ub = ub
        #self.__f_name = f_name
        self.__f_name = func_details[0]
        self.__lb = func_details[1]
        self.__ub = func_details[2]
        self.__dim = func_details[3]

        
    # Reimplement the virtual method called fitness that defines the objective function.
    # used to return the fitness of the input decision vector
    def fitness(self,x):
        
        """ x is the input decision vector i.e., (candidate solution) """
        objf = getattr(formulas, self.__f_name)
        fit =  objf(x)
        #f = np.sum(x ** 2)
        return [fit]
        
    ############################################
      
    # return the box bounds of the problem, , which also implicitly define the dimension of the problem   
    def get_bounds(self):
        
        return ([self.__lb] * self.__dim,[self.__ub] * self.__dim)
        
    
    def get_name(self):
        return self.__f_name
    
    def get_extra_info(self):
        
        return "\n\t Problem dimension: " + str(self.__dim)