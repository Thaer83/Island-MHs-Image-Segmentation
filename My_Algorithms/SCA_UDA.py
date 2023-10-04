import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_SCA():
    """
    Sine Cosine OPtimization Algorithm (WOA)
    Created on Monday NOvember 21 2022
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , verbosity=1, seed = random.seed()):
        """
        Constructs a SCA algorithm
        USAGE: algorithm.my_SCA(iter = 10)
        
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
        Positions = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        fitness = pop.get_f()
        
        # initialize position vector and score for the leader
        Dest_pos = numpy.zeros(dim)
        Dest_score = float("inf")  # change this to -inf for maximization problems
        
        Dest_score = pop.champion_f
        Dest_pos = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(Positions, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                 
          
            # Eq. (3.4)
            a = 2
            r1 = a - t * ((a) / self.__gen)  # r1 decreases linearly from a to 0
                           
            # Update the Position of search agents
            for i in range(0, PopSize):
                for j in range(0, dim):
                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = (2 * numpy.pi) * random.random()
                    r3 = 2 * random.random()
                    r4 = random.random()
                    
                    # Eq. (3.3)
                    if r4 < (0.5):
                        # Eq. (3.1)
                        Positions[i, j] = Positions[i, j] + (
                            r1 * numpy.sin(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                        )
                    else: 
                        # Eq. (3.2)
                        Positions[i, j] = Positions[i, j] + (
                            r1 * numpy.cos(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                        )
                     
            for i in range(0, PopSize):
                
                # Return back the search agents that go beyond the boundaries of the search space
                Positions[i, :] = numpy.clip(Positions[i, :], lb, ub)
                    
                # Calculate objective function for each search agent
                fitness[i] = pop.problem.fitness(Positions[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,Positions[i, :],fitness[i])
                    
            # Update Dest_Score
            Dest_score = pop.champion_f       
            Dest_pos = pop.champion_x
            #
            self.conv_list.append(float(Dest_score))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "SCA"
    
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
        