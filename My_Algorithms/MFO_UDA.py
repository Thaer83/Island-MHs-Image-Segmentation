import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_MFO():
    """
    Moth-flame optimization algorithm (MFO)
    Created on Sunday November 20 2022
    
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
        Moth_pos = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        Moth_fitness = pop.get_f()
        print(Moth_fitness)
        print(type(Moth_fitness[0]))
        Moth_fitness = Moth_fitness.reshape(-1)
        
        print(Moth_fitness)
        print(type(Moth_fitness[0]))
        
        
        sorted_population = numpy.copy(Moth_pos)
        fitness_sorted = numpy.zeros(PopSize)
        best_flames = numpy.copy(Moth_pos)
        best_flame_fitness = numpy.zeros(PopSize)
        double_population = numpy.zeros((2 * PopSize, dim))
        double_fitness = numpy.zeros(2 * PopSize)
        double_sorted_population = numpy.zeros((2 * PopSize, dim))
        double_fitness_sorted = numpy.zeros(2 * PopSize)
        previous_population = numpy.zeros((PopSize, dim))
        previous_fitness = numpy.zeros(PopSize)
        
        # initialize position vector and score for the leader
        
        Best_flame_score = pop.champion_f
        Best_flame_pos = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for Iteration in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(Moth_pos, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
            
            # Number of flames Eq. (3.14) in the paper
            #Flame_no = round(PopSize - Iteration * ((PopSize - 1) / self.__gen))
            Flame_no = round(PopSize - Iteration * ((PopSize - 1) / self.__gen))
            Flame_no = Flame_no - 1
            #print(Flame_no)
            
            if Iteration == 0:
                # Sort the first population of moths
                fitness_sorted = numpy.sort(Moth_fitness)
                I = numpy.argsort(Moth_fitness.flatten())
                I = numpy.reshape(I[0:len(Moth_fitness)],len(Moth_fitness))
                sorted_population = Moth_pos[I, :]
                
                # Update the flames
                best_flames = sorted_population
                best_flame_fitness = fitness_sorted
                
            else:
                # Sort the moths
                double_population = numpy.concatenate(
                    (previous_population, best_flames), axis=0
                )
               
                double_fitness = numpy.concatenate(
                    (previous_fitness, best_flame_fitness), axis=0
                )
               
                double_fitness_sorted = numpy.sort(double_fitness)
                I2 = numpy.argsort(double_fitness,axis=0)
                I2 = numpy.reshape(I2[0:len(double_fitness)],len(double_fitness))
                #
                #
                for newindex in range(0, 2 * PopSize):
                    double_sorted_population[newindex, :] = numpy.array(
                        double_population[I2[newindex], :]
                    )
                fitness_sorted = double_fitness_sorted[0:PopSize] 
                sorted_population = double_sorted_population[0:PopSize, :]
                #
                #        # Update the flames
                best_flames = sorted_population
                best_flame_fitness = fitness_sorted
                    
            #   # Update the position best flame obtained so far
            Best_flame_score = fitness_sorted[0]
            Best_flame_pos = sorted_population[0, :]
            #
            previous_population = Moth_pos
            previous_fitness = Moth_fitness
            #
            # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + Iteration * ((-1) / self.__gen)
            
            # Loop counter
            for i in range(0, PopSize):
                for j in range(0, dim):
                    if (i <= Flame_no):  # Update the position of the moth with respect to its corresponsing flame
                        # D in Eq. (3.13)
                        distance_to_flame = abs(sorted_population[i, j] - Moth_pos[i, j])
                        b = 1
                        t = (a - 1) * random.random() + 1
                        #
                        #                % Eq. (3.12)
                        Moth_pos[i, j] = (
                            distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi)
                            + sorted_population[Flame_no, j]
                        )
                
                # Check if moths go out of the search spaceand bring it back
                Moth_pos[i, :] = numpy.clip(Moth_pos[i, :], lb, ub)
                
                # Calculate objective function for each new particle
                Moth_fitness[i] = pop.problem.fitness(Moth_pos[i, :])
                
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,Moth_pos[i, :],Moth_fitness[i])
                                                       
            Best_flame_score = pop.champion_f    
            Best_flame_pos = pop.champion_x
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(Best_flame_score))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "MFO"
    
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
        