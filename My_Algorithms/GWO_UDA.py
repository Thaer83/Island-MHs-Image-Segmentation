import random
import numpy
import math
import time
from . import diversity_measures
import random
class my_GWO():
    """
    Grey Wolf Optimizer (GWO)
    Created on Thirsday May 15  2022
    
    @author: Thaer Thaher
    
    % _____________________________________________________
    % Main paper:
    % Grey Wolf Optimizer
    % Seyedali Mirjalili, Seyed Mohammad Mirjalili, Andrew Lewis
    % Advances in Engineering Software
    % DOI: https://doi.org/10.1016/j.advengsoft.2013.12.007
    % _____________________________________________________
    
    """
    def __init__(self,gen = 10, verbosity=1, seed = random.seed()):
        """
        Constructs a GWO algorithm
        USAGE: algorithm.my_GWO(iter = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__verbosity = verbosity
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
        pos = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        fitness = pop.get_f()
        #print(fitness)
        
        # initialize alpha, beta, and delta_pos
        
        Alpha_pos = numpy.zeros(dim)
        Alpha_score = float("inf")
        
        
        Beta_pos = numpy.zeros(dim)
        Beta_score = float("inf")

        Delta_pos = numpy.zeros(dim)
        Delta_score = float("inf")
        
                
        timerStart = time.time()
        
        for i in range(0, PopSize):
            # Update Alpha, Beta, and Delta
            if fitness[i] < Alpha_score:
                # Update alpha
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness[i]
                Alpha_pos = pos[i, :].copy()
                
            if fitness[i] > Alpha_score and  fitness[i] < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness[i]  # Update beta
                Beta_pos = pos[i, :].copy()
                    
            if fitness[i] > Alpha_score and  fitness[i] > Beta_score and fitness[i] < Delta_score:
                Delta_score = fitness[i]  # Update delta
                Delta_pos = pos[i, :].copy()
        
        #print("initial fitness", fitness.reshape(-1))
        #print("initial alpha score", Alpha_score)
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):
            Alpha_score = pop.champion_f
            # ========= Compute population diversity ======================
            moi = diversity_measures.moment_of_inertia(pos, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                     
            # a decreases linearly fron 2 to 0
            a = 2 - t * ((2) / self.__gen)
                    
            # Update the Position of search agents including omegas
            for i in range(0, PopSize):
                for j in range(0, dim):
                    
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
                    
                    A1 = 2 * a * r1 - a
                    # Equation (3.3)
                    C1 = 2 * r2
                    # Equation (3.4)
                    
                    D_alpha = abs(C1 * Alpha_pos[j] - pos[i, j])
                    # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    # Equation (3.6)-part 1
                    
                    r1 = random.random()
                    r2 = random.random()
                    
                    A2 = 2 * a * r1 - a
                    # Equation (3.3)
                    C2 = 2 * r2
                    # Equation (3.4)
                    
                    D_beta = abs(C2 * Beta_pos[j] - pos[i, j])
                    # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta
                    # Equation (3.6)-part 2
                    
                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    # Equation (3.3)
                    C3 = 2 * r2
                    # Equation (3.4)

                    D_delta = abs(C3 * Delta_pos[j] - pos[i, j])
                    # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta
                    # Equation (3.5)-part 3
                    
                    pos[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
                
                # Return back the search agents that go beyond the boundaries of the search space
                pos[i, :] = numpy.clip(pos[i, :], lb, ub) 
                    
                # Calculate objective function for each new particle
                fitness[i] = pop.problem.fitness(pos[i, :])
                #print(fitness[i])
                
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,pos[i, :],fitness[i])
                
                # Update Alpha, Beta, and Delta
                #print("if ", fitness[i], " < " , Alpha_score)
                if fitness[i] < Alpha_score:
                    #print(fitness[i], " < ", Alpha_score)
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = Alpha_score  # Update beta
                    Beta_pos = Alpha_pos.copy()
                    # Update alpha
                    Alpha_score = fitness[i]
                    Alpha_pos = pos[i, :].copy()
                
                elif fitness[i] < Beta_score:
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = fitness[i]  # Update beta
                    Beta_pos = pos[i, :].copy()
                    
                elif fitness[i] < Delta_score:
                    Delta_score = fitness[i]  # Update delta
                    Delta_pos = pos[i, :].copy()
            
            #gBestScoreee = pop.champion_f
            #print("gggg", gBestScoreee)
            #print("hhhh", Alpha_score)
            self.conv_list.append(float(Alpha_score))
        
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "GWO"
    
    def get_extra_info(self):
        
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