import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_WOA():
    """
    Whale Optimization Algorithm (WOA)
    Created on Sunday October 25 2022
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , verbosity=1, seed = random.seed()):
        """
        Constructs a WOA algorithm
        USAGE: algorithm.my_WOA(iter = 10)
        
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
        Leader_pos = numpy.zeros(dim)
        Leader_score = float("inf")  # change this to -inf for maximization problems
        
        Leader_score = pop.champion_f
        Leader_pos = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(Positions, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                 
          
            a = 2 - t * ((2) / self.__gen)       # a decreases linearly fron 2 to 0 in Eq. (2.3) 
            
            a2 = -1 + t * ((-1) / self.__gen)    # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
                           
            # Evaluate new positions 
            for i in range(0, PopSize):
                
                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]
                
                A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                C = 2 * r2          # Eq. (2.4) in the paper

                b = 1               #  parameters in Eq. (2.5)
                l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)
                
                p = random.random()  # p in Eq. (2.6)
                
                for j in range(0, dim):
                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = math.floor(
                                PopSize * random.random()
                            )
                            X_rand = Positions[rand_leader_index, :]
                            D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                            Positions[i, j] = X_rand[j] - A * D_X_rand             
                    
                        elif abs(A) < 1:
                            D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                            Positions[i, j] = Leader_pos[j] - A * D_Leader                        
                    
                    elif p >= 0.5:
                        distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                        # Eq. (2.5)
                        Positions[i, j] = (
                            distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                            + Leader_pos[j]
                        )                    
                                   
            for i in range(0, PopSize):
                
                #if(t!=0):
                # Return back the search agents that go beyond the boundaries of the search space
                Positions[i, :] = numpy.clip(Positions[i, :], lb, ub)
                    
                # Calculate objective function for each new particle
                fitness[i] = pop.problem.fitness(Positions[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,Positions[i, :],fitness[i])
                    
            Leader_score = pop.champion_f
                    
            Leader_pos = pop.champion_x
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(Leader_score))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "WOA"
    
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
        