import numpy


def moment_of_inertia(X, D, N):
    # D: Problem dimension
    # N: Population size
    # X: set of individuals
    centroid = numpy.zeros(D)
    I = 0
    #=================== Compute Population Diversity =============
    for i in range (0,D):
        for j in range(0,N):
            centroid[i] += X[j,i] / N
        #centroid[i] = centroid[i] / PopSize
            
    for i in range (0, D):
        for j in range (0, N):
            I += (X[j,i] - centroid[i])**2   
    #=============================================================
    
    return I  



    
