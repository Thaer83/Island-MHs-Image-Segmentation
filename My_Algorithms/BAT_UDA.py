import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_BAT():
    """
    Bat algorithm (BA)
    Created on Friday NOvember 18 2022
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , A = 0.5, Pr = 0.5, Qmin = 0, Qmax = 2, verbosity=1, seed = random.seed()):
        """
        Constructs a WOA algorithm
        USAGE: algorithm.my_WOA(iter = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        * A (float) - Loudness deceasing factor
        * Pr (float) - Pulse rate decreasing factor
        * Qmin (int) - Frequency minimum
        * Qmax (int) - Frequency maximum
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__verbosity = verbosity
        self.__A = A
        self.__Pr = Pr
        self.__Qmin = Qmin
        self.__Qmax = Qmax
        
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
        Sol = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        Fitness = pop.get_f()
        
        # Initializing arrays
        Q = numpy.zeros(PopSize)  # Frequency
        v = numpy.zeros((PopSize, dim))  # Velocities
        S = numpy.zeros((PopSize, dim))
        S = numpy.copy(Sol)
        
        #Find the initial best solution and minimum fitness
        fmin = pop.champion_f
        best = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(Sol, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                                            
            # Loop over all bats(solutions) 
            for i in range(0, PopSize):
                Q[i] = self.__Qmin + (self.__Qmin - self.__Qmax) * random.random()
                v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
                S[i, :] = Sol[i, :] + v[i, :]
                                   
                    
                # Pulse rate
                if random.random() > self.__Pr:
                    S[i, :] = best + 0.001 * numpy.random.randn(dim)
                    
                
                # Check boundaries
                S[i, :] = numpy.clip(S[i, :], lb, ub)
                # Evaluate new solutions
                Fnew = pop.problem.fitness(S[i, :])
                
                # Update if the solution improves
                if (Fnew <= Fitness[i]) and (random.random() < self.__A):
                    Sol[i, :] = numpy.copy(S[i, :])
                    Fitness[i] = Fnew
                    # Sets the -th individual decision vector, and fitness.
                    pop.set_xf(i,Sol[i, :],Fitness[i])
                            # Update the current best solution
                if Fnew <= fmin:
                    best = numpy.copy(S[i, :])
                    fmin = Fnew
                        
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(fmin))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "BAT"
    
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
        