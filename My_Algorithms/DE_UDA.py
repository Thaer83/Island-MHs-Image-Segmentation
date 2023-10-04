import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_DE():
    """
    Differential Evolution (DE) (DE)
    Created on Sunday November 20 2022
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , mutation_factor = 0.5, crossover_ratio = 0.7, stopping_func = None, verbosity=1, seed = random.seed()):
        """
        Constructs a WOA algorithm
        USAGE: algorithm.my_WOA(iter = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__verbosity = verbosity
        self.__mutation_factor = mutation_factor
        self.__crossover_ratio = crossover_ratio
        self.__stopping_func = stopping_func
       
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
        population = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        population_fitness = pop.get_f()
        
        # initialize position vector and score for the leader
        best_pos = numpy.zeros(dim)
        best_score = float("inf")  # change this to -inf for maximization problems
        
        best_score = pop.champion_f
        best_pos = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(population, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
            
            # should i stop
            #if stopping_func is not None and stopping_func(best_score, best_pos, t):
            if self.__stopping_func is not None and self.__stopping_func(best_score, best_pos, t):    
                break
                                             
            # loop through population 
            for i in range(0, PopSize):
                
                # 1. Mutation
                
                # select 3 random solution except current solution
                ids_except_current = [_ for _ in range(PopSize) if _ != i]
                id_1, id_2, id_3 = random.sample(ids_except_current, 3)
                
                mutant_sol = []
                for d in range(dim):
                    d_val = population[id_1, d] + self.__mutation_factor * (
                        population[id_2, d] - population[id_3, d]
                    )
                    
                    # 2. Recombination
                    rn = random.uniform(0, 1)
                    if rn > self.__crossover_ratio:
                        d_val = population[i, d]

                    # add dimension value to the mutant solution
                    mutant_sol.append(d_val)
                
                # 3. Replacement / Evaluation
                                   
                
                # clip new solution (mutant)
                mutant_sol = numpy.clip(mutant_sol, lb, ub)
                    
                # Calculate fitness
                mutant_fitness = pop.problem.fitness(mutant_sol)
                
                # replace if mutant_fitness is better
                if mutant_fitness < population_fitness[i]:
                    population[i, :] = mutant_sol
                    population_fitness[i] = mutant_fitness
                    
                    # Sets the -th individual decision vector, and fitness.
                    pop.set_xf(i,population[i, :],population_fitness[i])
                    
            best_score = pop.champion_f
                    
            best_pos = pop.champion_x
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(best_score))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "DE"
    
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
        