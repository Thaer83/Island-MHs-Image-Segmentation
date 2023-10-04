import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_JAYA():
    """
    JAYA algorithm
    Created on Sunday October 28 2022
    
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
        #Best_pos = numpy.zeros(dim)
        #Best_score = float("inf")  # change this to -inf for maximization problems
        
        #Worst_pos = numpy.zeros(dim)
        #Worst_score = float("-inf")

        
        Best_score = pop.champion_f
        Best_pos = pop.champion_x
        
        timerStart = time.time()
        
        Worst_score = max(fitness)
        max_index = fitness.argmax()
        Worst_pos = Positions[max_index, :].copy()
        
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(Positions, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                                            
            # Evaluate new positions 
            for i in range(0, PopSize):
                
                New_Position = numpy.zeros(dim)
                for j in range(0, dim):
                    
                    # Update r1, r2
                    r1 = random.random()
                    r2 = random.random()
                    
                    # JAYA Equation
                    New_Position[j] = (
                        Positions[i][j]
                        + r1 * (Best_pos[j] - abs(Positions[i, j]))
                        - r2 * (Worst_pos[j] - abs(Positions[i, j]))
                    )
                
                # checking if New_Position[j] lies in search space
                New_Position = numpy.clip(New_Position, lb, ub) 
                
                # Calculate objective function for each new position
                new_fitness = pop.problem.fitness(New_Position)
                
                current_fit = fitness[i]
                
                # replacing current element with new element if it has better fitness
                if new_fitness < current_fit:
                    Positions[i] = New_Position
                    fitness[i] = new_fitness
            
                    # Sets the -th individual decision vector, and fitness.
                    pop.set_xf(i,Positions[i, :],fitness[i])
                
            
            # finding the best and worst element
            Best_score = pop.champion_f        
            Best_pos = pop.champion_x
            Worst_score = max(fitness)
            max_index = fitness.argmax()
            Worst_pos = Positions[max_index, :].copy()
                        
            self.conv_list.append(float(Best_score))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "JAYA"
    
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
        