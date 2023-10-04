import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_GSA():
    """
    Gravitational Search Algorithm (GSA) 
    Created on Saturday NOvember 19 2022
    The code template used is similar given at link: https://github.com/himanshuRepo/Gravitational-Search-Algorithm/blob/master/GSA.py
    
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
        pos = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        fitness = pop.get_f()
        
        # GSA parameters
        ElitistCheck =1
        Rpower = 1 
        
        # Initializations
        vel=numpy.zeros((PopSize,dim))
        #fit = numpy.zeros(PopSize)
        M = numpy.zeros(PopSize)
        
        # initialize position vector and score for the leader
        gBest=numpy.zeros(dim)
        gBestScore=float("inf")
        
        gBestScore = pop.champion_f
        gBest = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):

            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(pos, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
                                             
            """ Calculating Mass """
            M = self.massCalculation(fitness,PopSize,M)
            
            """ Calculating Gravitational Constant """
            G = self.gConstant(t,self.__gen)
            
            """ Calculating Gfield """
            acc = self.gField(PopSize,dim,pos,M,t,self.__gen,G,ElitistCheck,Rpower)
            
            """ Calculating Position """ 
            pos, vel = self.move(PopSize,dim,pos,vel,acc)
            
            
            # Evaluate new positions                       
            for i in range(0, PopSize):
                
                # Return back the search agents that go beyond the boundaries of the search space
                '''
                l1 = [None] * dim
                l1=numpy.clip(pos[i,:], lb, ub)
                pos[i,:]=l1
                '''
                pos[i, :] = numpy.clip(pos[i, :], lb, ub)
                    
                # Calculate objective function for each new particle
                '''
                fitness=[]
                fitness=objf(l1)
                fit[i]=fitness
                '''
                fitness[i] = pop.problem.fitness(pos[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,pos[i, :],fitness[i])
                    
                '''
                if(gBestScore>fitness):
                    gBestScore=fitness
                    gBest=l1
                '''
            
            gBestScore = pop.champion_f
                    
            gBest = pop.champion_x
            #self.convergence_curve[t] = gBestScore 
            self.conv_list.append(float(gBestScore))
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "GSA"
    
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
    
    def gConstant(self,l,iters):
        alfa = 20
        G0 = 100
        Gimd = numpy.exp(-alfa*float(l)/iters)
        G = G0*Gimd
        return G
    
    def gField(self,PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower):
        final_per = 2
        if ElitistCheck == 1:
            kbest = final_per + (1-l/iters)*(100-final_per)
            kbest = round(PopSize*kbest/100)
        else:
            kbest = PopSize
        
        kbest = int(kbest)
        ds = sorted(range(len(M)), key=lambda k: M[k],reverse=True)
        
        Force = numpy.zeros((PopSize,dim))
        # Force = Force.astype(int)
        
        for r in range(0,PopSize):
            for ii in range(0,kbest):
                z = ds[ii]
                R = 0
                if z != r:
                    x=pos[r,:]
                    y=pos[z,:]
                    esum=0
                    imval = 0
                    for t in range(0,dim):
                        imval = ((x[t] - y[t])** 2)
                        esum = esum + imval
                    
                    R = math.sqrt(esum)
                    for k in range(0,dim):
                        randnum=random.random()
                        Force[r,k] = Force[r,k]+randnum*(M[z])*((pos[z,k]-pos[r,k])/(R**Rpower+numpy.finfo(float).eps))
        acc = numpy.zeros((PopSize,dim))
        for x in range(0,PopSize):
            for y in range (0,dim):
                acc[x,y]=Force[x,y]*G
        return acc
    
    def massCalculation(self,fit,PopSize,M):
        Fmax = max(fit)
        Fmin = min(fit)
        Fsum = sum(fit)        
        Fmean = Fsum/len(fit)
        
        if Fmax == Fmin:
             M = numpy.ones(PopSize)
        else:
            best = Fmin
            worst = Fmax
            
            for p in range(0,PopSize):
                M[p] = (fit[p]-worst)/(best-worst)
        
        Msum=sum(M)
        for q in range(0,PopSize):
            M[q] = M[q]/Msum
        return M
    
    def move(self,PopSize,dim,pos,vel,acc):
        for i in range(0,PopSize):
            for j in range (0,dim):
                r1=random.random()
                #r2=random.random()
                vel[i,j]=r1*vel[i,j]+acc[i,j]
                pos[i,j]=pos[i,j]+vel[i,j]
        return pos, vel
        