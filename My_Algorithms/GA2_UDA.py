import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_GA2():
    """
    genetic algorithm (GA)
    Created on Saturday November 19 2022
    The code template used is similar given at link: https://github.com/7ossam81/EvoloPy
    
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , cp = 0.8, mp = 0.001, verbosity=1, seed = random.seed()):
        """
        Constructs a WOA algorithm
        USAGE: algorithm.my_WOA(iter = 10)
        
        * gen (int) – number of generations
        * verbosity (int) - the verbosity of logs and screen output
        * cp (float) -  crossover Probability
        * mp (float) -  Mutation Probability
        * keep (int) - # elitism parameter: how many of the best individuals to keep from one generation to the next
        """
        
        #We start defining the algorithm ’private’ data members
        self.__gen = gen
        self.__verbosity = verbosity
        self.__cp = cp
        self.__mp = mp
       
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
        ga = pop.get_x()
        
        ## get number of rows (population size)
        PopSize = len(pop)#pop.shape[0]
        
        #Extract fitness values (return the fitness vectors of the individuals as a 2D NumPy array)
        scores = pop.get_f()
        
        # initialize position vector and score for the leader
        bestIndividual = numpy.zeros(dim)
        bestScore = float("inf")  # change this to -inf for maximization problems
        
        bestScore = pop.champion_f
        bestIndividual = pop.champion_x
        
        timerStart = time.time()
        
        #The algorithm now starts manipulating the population
        for t in range(0, self.__gen):
            
            # ========= Compute population diversity ======================
            moi = diversity_measures.moment_of_inertia(ga, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
            
            #Apply evolutionary operators to chromosomes
            ga = self.runOperators(ga, scores, bestIndividual, bestScore, self.__cp, self.__mp, PopSize, lb, ub)
            
            
            #Loop through chromosomes in population
            for i in range (0, PopSize):
                # Return back the search agents that go beyond the boundaries of the search space
                ga[i,:]=numpy.clip(ga[i,:], lb, ub)
            
                # Calculate objective function for each search agent
                scores[i] = pop.problem.fitness(ga[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,ga[i, :],scores[i])
            
            #scores, pop = self.calculateCost(pop, ga, PopSize, lb, ub)
            
            bestScore = pop.champion_f
            bestIndividual = pop.champion_x
           
            self.conv_list.append(float(bestScore))
            
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "GA2"
    
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
        
    
    def runOperators(self, population, scores, bestIndividual, bestScore,  
                  crossoverProbability, mutationProbability, 
                  PopSize, lb, ub):
        """    
        This method calls the evolutionary operators
    
        Parameters
        ----------    
        population : list
            The list of chromosomes
        scores : list
            The list of fitness values for each chromosome
        bestIndividual : list
            A chromosome of the previous generation having the best fitness value          
        bestScore : float
            The best fitness value of the previous generation
        crossoverProbability : float
            The probability of crossover
        mutationProbability : float
            The probability of mutation
        PopSize: int
            Number of chrmosomes in a population
        lb: int
            lower bound limit
        ub: int
            Upper bound limit
    
        Returns
        -------
        list
            newPopulation: the new generated population after applying the genetic operations
        """
        #Elitism operation
        self.elitism(population, scores, bestIndividual, bestScore)
        #initialize a new population
        newPopulation = numpy.empty_like(population)
    
        #Create pairs of parents. The number of pairs equals the number of chromosomes divided by 2
        
        for i  in range(0, PopSize, 2):
            #pair of parents selection
        
            parent1, parent2 = self.pairSelection(population, scores, PopSize)
        
            #crossover
            crossoverLength = min(len(parent1), len(parent2))
            parentsCrossoverProbability = random.uniform(0.0, 1.0)
            if parentsCrossoverProbability < crossoverProbability:
                offspring1, offspring2 = self.crossover(crossoverLength, parent1, parent2)
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()
            
            #Mutation   
            offspringMutationProbability = random.uniform(0.0, 1.0)
            if offspringMutationProbability < mutationProbability:
                self.mutation(offspring1, len(offspring1), lb, ub)
            offspringMutationProbability = random.uniform(0.0, 1.0)
            if offspringMutationProbability < mutationProbability:
                self.mutation(offspring2, len(offspring2), lb, ub)
            
            #Add offsprings to population
            newPopulation[i] = numpy.copy(offspring1)
            newPopulation[i + 1] = numpy.copy(offspring2)
        
        return newPopulation
    
    def elitism(self,population, scores, bestIndividual, bestScore):
        """    
        This method performs the elitism operator
    
        Parameters
        ----------    
        population : list
            The list of chromosomes
        scores : list
            The list of fitness values for each chromosome
        bestIndividual : list
            A chromosome of the previous generation having the best fitness value          
        bestScore : float
            The best fitness value of the previous generation        
    
        Returns
        -------
        list
            population : The updated population after applying the elitism
        list
            scores : The updated list of fitness values for each chromosome after applying the elitism
        """
    
        # get the worst chromosome
        worstFitnessId = self.selectWorstChromosome(scores)
    
        #replace worst cromosome with best one from previous generation if its fitness is less than the other
        if scores[worstFitnessId] > bestScore:
            population[worstFitnessId] = numpy.copy(bestIndividual)
            scores[worstFitnessId] = numpy.copy(bestScore)


    def selectWorstChromosome(self, scores):
        """    
        It is used to get the worst chromosome in a population based n the fitness value
    
        Parameters
        ---------- 
        scores : list
            The list of fitness values for each chromosome
        
        Returns
        -------
        int
            maxFitnessId: The chromosome id of the worst fitness value
        """
    
        maxFitnessId = numpy.where(scores == numpy.max(scores))
        maxFitnessId = maxFitnessId[0][0]
        return maxFitnessId
    
    def pairSelection(self, population, scores, PopSize):    
        """    
        This is used to select one pair of parents using roulette Wheel Selection mechanism
    
        Parameters
        ---------- 
        population : list
            The list of chromosomes
        scores : list
            The list of fitness values for each chromosome
        PopSize: int
            Number of chrmosome in a population
          
        Returns
        -------
        list
            parent1: The first parent chromosome of the pair
        list
            parent2: The second parent chromosome of the pair
        """
        parent1Id = self.rouletteWheelSelectionId(scores, PopSize)
        parent2Id = numpy.copy(parent1Id)
    
        parent1 = population[parent1Id].copy()
        while parent1Id == parent2Id:  
            parent2Id = self.rouletteWheelSelectionId(scores, PopSize)
    
        parent2 = population[parent2Id].copy()
   
        return parent1, parent2

    def rouletteWheelSelectionId(self, scores, PopSize): 
        """    
        A roulette Wheel Selection mechanism for selecting a chromosome
    
        Parameters
        ---------- 
        scores : list
            The list of fitness values for each chromosome
        sumScores : float
            The summation of all the fitness values for all chromosomes in a generation
        PopSize: int
            Number of chrmosome in a population
          
        Returns
        -------
        id
            chromosomeId: The id of the chromosome selected
        """
    
        ##reverse score because minimum value should have more chance of selection
        reverse = max(scores) + min(scores)
        reverseScores = reverse - scores.copy()
        sumScores = sum(reverseScores)
        pick = random.uniform(0, sumScores)
        current = 0
        for chromosomeId in range(PopSize):
            current += reverseScores[chromosomeId]
            if current > pick:
                return chromosomeId

    def crossover(self, chromosomeLength, parent1, parent2):
        """    
        The crossover operator
    
        Parameters
        ---------- 
        chromosomeLength: int
            The maximum index of the crossover
        parent1 : list
            The first parent chromosome of the pair
        parent2 : list
            The second parent chromosome of the pair
          
        Returns
        -------
        list
            offspring1: The first updated parent chromosome of the pair
        list
            offspring2: The second updated parent chromosome of the pair
        """
    
        # The point at which crossover takes place between two parents. 
        crossover_point = random.randint(0, chromosomeLength - 1)

        # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
        offspring1 = numpy.concatenate([parent1[0:crossover_point],parent2[crossover_point:]])
        # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
        offspring2 = numpy.concatenate([parent2[0:crossover_point],parent1[crossover_point:]])
      
        return offspring1, offspring2 
    
    def mutation(self, offspring, chromosomeLength, lb, ub):
        """    
        The mutation operator
    
        Parameters
        ---------- 
        offspring : list
            A generated chromosome after the crossover
        chromosomeLength: int
            The maximum index of the crossover
        lb: int
            lower bound limit
        ub: int
            Upper bound limit
         
        Returns
        -------
        list
            offspring: The updated offspring chromosome
        """
        mutationIndex = random.randint(0, chromosomeLength - 1)
        mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
        offspring[mutationIndex] = mutationValue

    




        