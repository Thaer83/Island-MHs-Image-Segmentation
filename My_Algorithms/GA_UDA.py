import random
import numpy
import math
import time
from . import diversity_measures 
import random
class my_GA():
    """
    genetic algorithm (GA)
    Created on Saturday November 19 2022
    The code template used is similar given at link: https://github.com/7ossam81/EvoloPy
    
    
    @author: Thaer Thaher
    """
    def __init__(self,gen = 10 , cp = 1, mp = 0.01, keep = 2, verbosity=1, seed = random.seed()):
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
        self.__keep = keep
       
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
            
            # ========= Copute population diversity ======================
            moi = diversity_measures.moment_of_inertia(ga, dim, PopSize)
            self.diversity_list.append(float(moi))
            #=============================================================
            
            # crossover
            ga = self.crossoverPopulaton(ga, scores, PopSize, self.__cp, self.__keep)
            
            # mutation
            self.mutatePopulaton(ga, PopSize, self.__mp, self.__keep, lb, ub)
            
            ga = self.clearDups(ga, lb, ub)
            
            for i in range (0, PopSize):
                # Return back the search agents that go beyond the boundaries of the search space
                ga[i,:]=numpy.clip(ga[i,:], lb, ub)
            
                # Calculate objective function for each agent
                scores[i] = pop.problem.fitness(ga[i, :])
                    
                # Sets the -th individual decision vector, and fitness.
                pop.set_xf(i,ga[i, :],scores[i])
            
            #scores, pop = self.calculateCost(pop, ga, PopSize, lb, ub)
            
            bestScore = pop.champion_f
            bestIndividual = pop.champion_x
            
            # Sort from best to worst
            ga, scores = self.sortPopulation(ga, scores)
            
            # Evaluate new positions 
                
       
            self.conv_list.append(float(bestScore))
            
               
        timerEnd = time.time()
        self.fevals = pop.problem.get_fevals()
        self.executionTime = timerEnd - timerStart
        return pop
    ############################################
      
        
    def get_name(self):
        return "GA"
    
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
        
    def crossoverPopulaton(self, population, scores, popSize, crossoverProbability, keep):
        """
        The crossover of all individuals

        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population
        crossoverProbability: float
            The probability of crossing a pair of individuals
        keep: int
            Number of best individuals to keep without mutating for the next generation


        Returns
        -------
        N/A
        """
        # initialize a new population
        newPopulation = numpy.empty_like(population)
        newPopulation[0:keep] = population[0:keep]
        # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
        for i in range(keep, popSize, 2):
            # pair of parents selection
            parent1, parent2 = self.pairSelection(population, scores, popSize)
            crossoverLength = min(len(parent1), len(parent2))
            parentsCrossoverProbability = random.uniform(0.0, 1.0)
            if parentsCrossoverProbability < crossoverProbability:
                offspring1, offspring2 = self.crossover(crossoverLength, parent1, parent2)
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            # Add offsprings to population
            newPopulation[i] = numpy.copy(offspring1)
            newPopulation[i + 1] = numpy.copy(offspring2)

        return newPopulation
    
    def mutatePopulaton(self, population, popSize, mutationProbability, keep, lb, ub):
        """
        The mutation of all individuals

        Parameters
        ----------
        population : list
            The list of individuals
        popSize: int
            Number of chrmosome in a population
        mutationProbability: float
            The probability of mutating an individual
        keep: int
            Number of best individuals to keep without mutating for the next generation
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list

        Returns
        -------
        N/A
        """
        for i in range(keep, popSize):
            # Mutation
            offspringMutationProbability = random.uniform(0.0, 1.0)
            if offspringMutationProbability < mutationProbability:
                self.mutation(population[i], len(population[i]), lb, ub)
     
    def elitism(self, population, scores, bestIndividual, bestScore):
        """
        This melitism operator of the population

        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        bestIndividual : list
            An individual of the previous generation having the best fitness value
        bestScore : float
            The best fitness value of the previous generation

        Returns
        -------
        N/A
        """

        # get the worst individual
        worstFitnessId = self.selectWorstIndividual(scores)

        # replace worst cromosome with best one from previous generation if its fitness is less than the other
        if scores[worstFitnessId] > bestScore:
            population[worstFitnessId] = numpy.copy(bestIndividual)
            scores[worstFitnessId] = numpy.copy(bestScore)
            
    
    
    def selectWorstIndividual(self, scores):
        """
        It is used to get the worst individual in a population based n the fitness value

        Parameters
        ----------
        scores : list
            The list of fitness values for each individual

        Returns
        -------
        int
            maxFitnessId: The individual id of the worst fitness value
        """

        maxFitnessId = numpy.where(scores == numpy.max(scores))
        maxFitnessId = maxFitnessId[0][0]
        return maxFitnessId
    
    def pairSelection(self, population, scores, popSize):
        """
        This is used to select one pair of parents using roulette Wheel Selection mechanism

        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population

        Returns
        -------
        list
            parent1: The first parent individual of the pair
        list
            parent2: The second parent individual of the pair
        """
        parent1Id = self.rouletteWheelSelectionId(scores, popSize)
        parent1 = population[parent1Id].copy()

        parent2Id = self.rouletteWheelSelectionId(scores, popSize)
        parent2 = population[parent2Id].copy()

        return parent1, parent2
    
    def rouletteWheelSelectionId(self, scores, popSize):
        """
        A roulette Wheel Selection mechanism for selecting an individual

        Parameters
        ----------
        scores : list
            The list of fitness values for each individual
        popSize: int
            Number of chrmosome in a population

        Returns
        -------
        id
            individualId: The id of the individual selected
        """

        ##reverse score because minimum value should have more chance of selection
        reverse = max(scores) + min(scores)
        reverseScores = reverse - scores.copy()
        sumScores = sum(reverseScores)
        pick = random.uniform(0, sumScores)
        current = 0
        for individualId in range(popSize):
            current += reverseScores[individualId]
            if current > pick:
                return individualId
    
    def crossover(self, individualLength, parent1, parent2):
        """
        The crossover operator of a two individuals

        Parameters
        ----------
        individualLength: int
            The maximum index of the crossover
        parent1 : list
            The first parent individual of the pair
        parent2 : list
            The second parent individual of the pair

        Returns
        -------
        list
            offspring1: The first updated parent individual of the pair
        list
            offspring2: The second updated parent individual of the pair
        """

        # The point at which crossover takes place between two parents.
        crossover_point = random.randint(0, individualLength - 1)
        # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
        offspring1 = numpy.concatenate(
            [parent1[0:crossover_point], parent2[crossover_point:]]
        )
        # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
        offspring2 = numpy.concatenate(
            [parent2[0:crossover_point], parent1[crossover_point:]]
        )

        return offspring1, offspring2
    
    def mutation(self, offspring, individualLength, lb, ub):
        """
        The mutation operator of a single individual

        Parameters
        ----------
        offspring : list
            A generated individual after the crossover
        individualLength: int
            The maximum index of the crossover
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list

        Returns
        -------
        N/A
        """
        mutationIndex = random.randint(0, individualLength - 1)
        mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
        offspring[mutationIndex] = mutationValue
        
    
    def clearDups(self,Population, lb, ub):

        """
        It removes individuals duplicates and replace them with random ones

        Parameters
        ----------
        objf : function
            The objective function selected
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list

        Returns
        -------
        list
            newPopulation: the updated list of individuals
        """
        newPopulation = numpy.unique(Population, axis=0)
        oldLen = len(Population)
        newLen = len(newPopulation)
        if newLen < oldLen:
            nDuplicates = oldLen - newLen
            newPopulation = numpy.append(
                newPopulation,
                numpy.random.uniform(0, 1, (nDuplicates, len(Population[0])))
                * (numpy.array(ub) - numpy.array(lb))
                + numpy.array(lb),
                axis=0,
            )

        return newPopulation
    
    
    def calculateCost(self, pop, population, popSize, lb, ub):

        """
        It calculates the fitness value of each individual in the population

        Parameters
        ----------
        objf : function
            The objective function selected
        population : list
            The list of individuals
        popSize: int
            Number of chrmosomes in a population
        lb: list
            lower bound limit list
        ub: list
            Upper bound limit list

        Returns
        -------
        list
            scores: fitness values of all individuals in the population
        """
        scores = numpy.full(popSize, numpy.inf)

        # Loop through individuals in population
        for i in range(0, popSize):
            # Return back the search agents that go beyond the boundaries of the search space
            population[i] = numpy.clip(population[i], lb, ub)
            
            # Calculate objective function for each agent
            scores[i] = pop.problem.fitness(population[i, :])
                    
            # Sets the -th individual decision vector, and fitness.
            pop.set_xf(i,population[i, :],scores[i])


        return scores, pop
    
    
    def sortPopulation(self,population, scores):
        """
        This is used to sort the population according to the fitness values of the individuals

        Parameters
        ----------
        population : list
            The list of individuals
        scores : list
            The list of fitness values for each individual

        Returns
        -------
        list
            population: The new sorted list of individuals
        list
            scores: The new sorted list of fitness values of the individuals
        """
        sortedIndices = numpy.argsort(scores,axis=0)
        sortedIndices = numpy.reshape(sortedIndices[0:len(scores)],len(scores))
        population = population[sortedIndices]
        scores = scores[sortedIndices]
        
        return population, scores







        