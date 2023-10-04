import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_HHO():
    """
    Harris hawks optimization (HHO)
    Created on Thirsday February 18  2022
    
    @author: Thaer Thaher
    
    % _____________________________________________________
    % Main paper:
    % Harris hawks optimization: Algorithm and applications
    % Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
    % Future Generation Computer Systems, 
    % DOI: https://doi.org/10.1016/j.future.2019.02.028
    % _____________________________________________________
    """
    def __init__(self,gen = 10, verbosity=1, seed = random.seed()):
        """
        Constructs HHO algorithm
        USAGE: algorithm.my_HHO(gen = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
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
        PopSize = len(pop)
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        fitness = pop.get_f()
        
        # Extract the location and Energy of the rabbit in the initial population
        Rabbit_Location = pop.champion_x
        Rabbit_Energy = pop.champion_f  
        
        #convergence_curve = numpy.zeros(iters)
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):
            
            # ========= Compute population diversity ======================
            moi = diversity_measures.moment_of_inertia(pos, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                         
            if(t!=0):
                for i in range(0, PopSize):
                    # Check boundries
                    pos[i, :] = numpy.clip(pos[i, :], lb, ub)   
                    
                    # Calculate objective function for each new individual
                    fitness[i] = pop.problem.fitness(pos[i, :])
                    
                    # Sets the ith individual decision vector, and fitness.
                    pop.set_xf(i,pos[i, :],fitness[i])
                
                # Update the location of Rabbit
                Rabbit_Location = pop.champion_x
                Rabbit_Energy = pop.champion_f             
            
            # Update E1 [factor to show the decreaing energy of rabbit] 
            E1 = 2 * (1 - (t / self.__gen))  
            
            # Update the location of Harris' hawks
            for i in range(0, PopSize):
                
                E0 = 2 * random.random() - 1  # -1<E0<1
                
                Escaping_Energy = E1 * (E0)  # escaping energy of rabbit Eq. (3) in the paper
                
                # -------- Exploration phase Eq. (1) in paper -------------------
                
                if abs(Escaping_Energy) >= 1:
                    # Harris' hawks perch randomly based on 2 strategy:
                    q = random.random()
                    rand_Hawk_index = math.floor(PopSize * random.random())
                    X_rand = pos[rand_Hawk_index, :]
                    
                    if q < 0.5:
                        # perch based on other family members
                        pos[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * pos[i, :]
                        )

                    elif q >= 0.5:
                        # perch on a random tall tree (random site inside group's home range)
                        pos[i, :] = (Rabbit_Location - pos.mean(0)) - random.random() * (
                            (ub - lb) * random.random() + lb
                        )
                        
                # -------- Exploitation phase -------------------
                elif abs(Escaping_Energy) < 1:
                    # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                    # phase 1: ----- surprise pounce (seven kills) ----------
                    # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                    
                    r = random.random()  # probablity of each event
                    
                    if (r >= 0.5 and abs(Escaping_Energy) < 0.5):  # Hard besiege Eq. (6) in paper
                        
                        pos[i, :] = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - pos[i, :])
                    
                    if (r >= 0.5 and abs(Escaping_Energy) >= 0.5):  # Soft besiege Eq. (4) in paper
                        
                        Jump_strength = 2 * (1 - random.random())  # random jump strength of the rabbit
                        pos[i, :] = (Rabbit_Location - pos[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - pos[i, :]
                        )
                    
                    # phase 2: --------performing team rapid dives (leapfrog movements)----------
                    
                    if (r < 0.5 and abs(Escaping_Energy) >= 0.5):  # Soft besiege Eq. (10) in paper
                        # rabbit try to escape by many zigzag deceptive motions
                        Jump_strength = 2 * (1 - random.random())
                        X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - pos[i, :]
                        )
                        
                        X1 = numpy.clip(X1, lb, ub) 
                        
                        if pop.problem.fitness(X1) < fitness[i]:  # improved move?
                            pos[i, :] = X1.copy()
                        else:  # hawks perform levy-based short rapid dives around the rabbit
                            X2 = (
                                Rabbit_Location
                                - Escaping_Energy
                                * abs(Jump_strength * Rabbit_Location - pos[i, :])
                                + numpy.multiply(numpy.random.randn(dim), self.Levy(dim))
                                )
                            
                            X2 = numpy.clip(X2, lb, ub) 
                            if pop.problem.fitness(X2) < fitness[i]:
                                pos[i, :] = X2.copy()
                    
                    if (r < 0.5 and abs(Escaping_Energy) < 0.5):  # Hard besiege Eq. (11) in paper
                        Jump_strength = 2 * (1 - random.random())
                        X1 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - pos.mean(0)
                        )
                        
                        X1 = numpy.clip(X1, lb, ub)  
                        
                        if pop.problem.fitness(X1) < fitness[i]:  # improved move?
                            pos[i, :] = X1.copy()
                            
                        else:  # Perform levy-based short rapid dives around the rabbit
                            X2 = (
                                Rabbit_Location
                                - Escaping_Energy
                                * abs(Jump_strength * Rabbit_Location - pos.mean(0))
                                + numpy.multiply(numpy.random.randn(dim), self.Levy(dim))
                            )
                            
                            X2 = numpy.clip(X2, lb, ub)  
                            
                            if pop.problem.fitness(X2) < fitness[i]:
                                pos[i, :] = X2.copy()

            #self.convergence_curve[t] = Rabbit_Energy 
            self.conv_list.append(float(Rabbit_Energy))
        
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart  
        return pop
    ############################################
      
        
    def get_name(self):
        return "HHO"
    
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
    
    def Levy(self,dim):
        beta = 1.5
        sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * numpy.random.randn(dim) * sigma
        v = numpy.random.randn(dim)
        zz = numpy.power(numpy.absolute(v), (1 / beta))
        step = numpy.divide(u, zz)
        
        return step