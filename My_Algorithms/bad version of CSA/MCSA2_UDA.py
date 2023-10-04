import random
import numpy
import math
import time
class my_MCSA2():
    """
    Dynamic Awareness Probability (DAP)Crow Search Algorithm (CSA)
    Created on Thirsday September 29 2022
    
    @author: Thaer Thaher
    """
    import random
    def __init__(self,gen = 10, fl =2 , verbosity=1, seed = random.seed()):
        """
        Constructs a CSA algorithm
        USAGE: algorithm.my_CSA(iter = 10)
        
        * gen (int) – number of generations
        * AP (float) – Awareness probability
        * fl (float) – Flight length (fl) 
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        #self.__AP = AP
        self.__fl = fl
        self.__verbosity = verbosity
       
        #self.convergence_curve = numpy.zeros(gen)
        self.fevals = 0
        self.executionTime = 0
        self.__seed = seed 
        #
        self.conv_list = []
        
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
        X = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        fitness = pop.get_f()
        
        #Initializations

        xnew = numpy.zeros((PopSize, dim))
        
        xn = X.copy()   # the position of the crow
        mem = X.copy()  # Memory initialization
        fit_mem=fitness.copy()    # % Fitness of memory positions

        # best solution found in the initial population
        gBest = pop.champion_x
        gBestScore = pop.champion_f
        
        gWorstScore = max(fit_mem)   # to be used in the dynamic awarness probability formula
               
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):
                        
            for i in range(0, PopSize):
                
                r=random.random()
                num = random.randint(0,PopSize-1)  # Generation of random candidate crows for following (chasing)
                
                # Daynamic Awarmess Probability (DAP)----------
                
                DAP = 0.9 * (fitness[i]/gWorstScore) + 0.1
                # --------------------------------------------
                
                if r >= DAP: 
                    xnew[i,:]= xn[i,:]+self.__fl*r*(mem[num,:]-xn[i,:])# Generation of a new position for crow i (state 1)
                else:                       # Generation of a new position for crow i (state 2)
                    xnew[i,:]= numpy.random.uniform(0, 1, dim) * (ub - lb) + lb
            
            # Evaluate new positions 
            for i in range(0, PopSize):
                # Calculate objective function for each new crow
                fitness[i] = pop.problem.fitness(xnew[i, :])
            
            # Check the feasibility of new positions
            for i in range(0, PopSize):
                if self.check_feasibility (xnew[i, :], lb[0], ub[0]):
                    xn[i, :] = xnew[i, :].copy()    # update position
                    # Sets the -th individual decision vector, and fitness.
                    pop.set_xf(i,xn[i, :],fitness[i])
                    if fitness[i] < fit_mem[i]:
                        mem[i, :] = xnew[i, :].copy()
                        fit_mem[i]=fitness[i]
                        
            gBestScore = min(fit_mem)
            min_index = fit_mem.argmin()
            gBest = mem[min_index, :].copy()
            gWorstScore = max(fitness)
                       
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(gBestScore))
            
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "CSA"
    
    def get_extra_info(self):
        
        #info = [self.fevals, self.executionTime, self.convergence_curve.tolist()]
        info = [self.fevals, self.executionTime, self.conv_list]
        # use map() to convert values into a string before using the join method.
        return "$".join(map(str,info))
        #return str(info)
        
        #return "n_iter=" + str(self.__gen)
    def set_seed(self, s):
        random.seed(s)
        numpy.random.seed(s)
        
    def set_verbosity(self, l):
        
        self.__verbosity = l
        
    def check_feasibility(self, pos, lb, ub):
        for val in pos:
            if val < lb or val > ub:
                return False
        return True