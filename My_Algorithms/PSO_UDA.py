import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_PSO():
    """
    Particle Swarm Optimization (PSO)
    Created on Thirsday March 21  2022.
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10, w_max = 0.9, w_min = 0.2, c1 = 2.05, c2 = 2.05, max_vel = 6, verbosity=1, seed = random.seed()):
        """
        Constructs a PSO algorithm
        USAGE: algorithm.my_algorithm(iter = 10)
        
        * gen (int) – number of generations
        * w_max (float) – maximum allowed inertia weight (or constriction factor)
        * w_min (float) – minimum allowed inertia weight 
        * c1 (float) – social component
        * c2 (float) – cognitive component
        * max_vel (float) – maximum allowed particle velocities
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__w_max = w_max
        self.__w_min = w_min
        self.__c1 = c1
        self.__c2 = c2
        self.__max_vel = max_vel
        #
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
        
        #Initializations

        vel = numpy.zeros((PopSize, dim))

        pBestScore = numpy.zeros(PopSize)
        pBestScore.fill(float("inf"))
        pBest = numpy.zeros((PopSize, dim))
        
        for i in range (0, PopSize):
            if pBestScore[i] > fitness[i]:
                pBestScore[i] = fitness[i]
                pBest[i, :] = pos[i, :].copy()
        #gBest = numpy.zeros(dim)
        #gBestScore = float("inf")
        gBest = pop.champion_x
        gBestScore = pop.champion_f

        
        #convergence_curve = numpy.zeros(self.__gen)
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):
            
            # ========= Compute population diversity ======================
            moi = diversity_measures.moment_of_inertia(pos, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                        
            # Update inertia weight 
            w = self.__w_max - t * ((self.__w_max - self.__w_min) / self.__gen)
            
            for i in range(0, PopSize):
                
                r1 = numpy.random.rand(dim)
                r2 = numpy.random.rand(dim)
                
                vel[i, :] = (
                    w * vel[i, :]
                    + self.__c1 * r1 * (pBest[i, :] - pos[i, :])
                    + self.__c2 * r2 * (gBest - pos[i, :])
                )
                
                for j in range(0, dim):
                    if vel[i, j] > self.__max_vel:
                        vel[i, j] = self.__max_vel

                    if vel[i, j] < -self.__max_vel:
                        vel[i, j] = -self.__max_vel
                
                pos[i, :] = pos[i, :] + vel[i, :]
                
                # Check boundries
                pos[i, :] = numpy.clip(pos[i, :], lb, ub) 
                
                # Calculate objective function for each new particle
                fitness[i] = pop.problem.fitness(pos[i, :])
                
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,pos[i, :],fitness[i])
                
                if pBestScore[i] > fitness[i]:
                    pBestScore[i] = fitness[i]
                    pBest[i, :] = pos[i, :].copy()
                
            gBest = pop.champion_x
            gBestScore = pop.champion_f
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(gBestScore))
        
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "PSO"
    
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