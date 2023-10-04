import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_SSA():
    """
    Salp Swarm Algorithm (SSA)
    Created on Sunday October 26 2022
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , verbosity=1, seed = random.seed()):
        """
        Constructs an SSA algorithm
        USAGE: algorithm.my_SSA(iter = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__verbosity = verbosity
       
        #self.convergence_curve = numpy.zeros(gen)
        self.fevals = 0
        self.executionTime = 0
        self.__seed = seed 
        #
        self.conv_list = []
        self.diversity_list = []
        
    #This is the ’juice’ of the algorithm, the method where the actual optimzation is coded.
    def evolve(self,pop):
        #If the population is empty (i.e. no individuals) nothing happens
        
        if len(pop) == 0:
            return pop
        
        # Use the pop methods to extract evaluate set the various chromosomes

        #Here we rename some variables, in particular the problem
        prob = pop.problem
        
        #Its dimensions (total and continuous)
        dim, cont_dim = prob.get_nx(), prob.get_nx() - prob.get_nix()
        
        #And the lower/upper bounds for the chromosome
        bounds = prob.get_bounds()
        lb, ub = bounds[0], bounds[1]
        
        #Extract the chromosomes of the individuals as a 2D NumPy array
        SalpPositions = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        SalpFitness = pop.get_f()
        
        # initialize position vector and score for the leader
        FoodPosition = numpy.zeros(dim)
        FoodFitness = float("inf")  # change this to -inf for maximization problems
        
        FoodFitness = pop.champion_f
        FoodPosition = pop.champion_x
        
        timerStart = time.time()
        
        sorted_salps_fitness = numpy.sort(SalpFitness)
        I = numpy.argsort(SalpFitness)
        Sorted_salps = numpy.copy(SalpPositions[I, :])
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(SalpPositions, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                 
          
            c1 = 2 * math.exp(-((4 * t / self.__gen) ** 2))  # Eq. (3.2) in the paper 
                                      
            # Evaluate new positions 
            for i in range(0, PopSize):
                
                SalpPositions = numpy.transpose(SalpPositions)
                
                if i < PopSize / 2:
                    for j in range(0, dim):
                        c2 = random.random()
                        c3 = random.random()
                        # Eq. (3.1) in the paper
                        if c3 < 0.5:
                            SalpPositions[j, i] = FoodPosition[j] + c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )                            
                        else:
                            SalpPositions[j, i] = FoodPosition[j] - c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )                            
                   
                elif i >= PopSize / 2 and i < PopSize + 1:
                    point1 = SalpPositions[:, i - 1]
                    point2 = SalpPositions[:, i]
                    
                    SalpPositions[:, i] = (point2 + point1) / 2      # Eq. (3.4) in the paper
                    
                SalpPositions = numpy.transpose(SalpPositions)
                
            # Evaluate new positions
            for i in range(0, PopSize):
                
                # Check if salps go out of the search spaceand bring it back
                SalpPositions[i, :] = numpy.clip(SalpPositions[i, :], lb, ub)
                    
                # Calculate objective function for each new particle
                SalpFitness[i] = pop.problem.fitness(SalpPositions[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,SalpPositions[i, :],SalpFitness[i])
                    
            FoodFitness = pop.champion_f
                    
            FoodPosition = pop.champion_x
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(FoodFitness))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "SSA"
    
    def get_extra_info(self):
        
        #info = [self.fevals, self.executionTime, self.convergence_curve.tolist()]
        info = [self.fevals, self.executionTime, self.conv_list, self.diversity_list]
        # use map() to convert values into a string before using the join method.
        return "$".join(map(str,info))
        #return str(info)
        
        #return "n_iter=" + str(self.__gen)
    def set_seed(self, s):
        random.seed(s)
        numpy.random.seed(s)
        
    def set_verbosity(self, l):
        
        self.__verbosity = l
        