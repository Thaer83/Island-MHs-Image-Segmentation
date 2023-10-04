# -*- coding: utf-8 -*-
"""
Created on Fri JUly 1 15:50:25 2016

@author: Thaer
"""

from optimizer import run

# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizer = ["PSO", "GWO", "HHO"]
#optimizer = ["PSO"]

# Select tested images"
# "test1.jpg","test2.jpg","test3.jpg","test4.jpg","test5.jpg","test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"
# "Ca1","Ca2","Gt1","Mes","Mef","Sag","Tan","Ros"
#objectivefunc = ["test1.jpg", "test2.jpg", "test3.jpg"]

objectivefunc = ["test1.jpg","test2.jpg","test3.jpg","test4.jpg","test5.jpg","test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"]
#objectivefunc = ["Xtest1.jpg","Xtest2.jpg","Xtest3.jpg","Xtest4.jpg","Xtest5.jpg","Xtest6.jpg","Xtest7.jpg","Xtest8.jpg","Xtest9.jpg","Xtest10.jpg"]


# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 10

# NUmber of thresolds
Level = 2

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 30, "Iterations": 500}

# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags, Level)