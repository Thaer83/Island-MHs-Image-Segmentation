# -*- coding: utf-8 -*-
"""
Created on Mon JUly 3 15:50:25 2016

@author: Thaer
"""

from cooperative_optimizer import run

# Select cooperative optimizers of the archipelago
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizers = ["PSO", "GWO", "HHO"]

# Select tested images
# "test1.jpg","test2.jpg","test3.jpg","test4.jpg","test5.jpg","test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"
#objectivefunc = ["test1.jpg", "test2.jpg", "test3.jpg"]

objectivefunc = ["test1.jpg","test2.jpg","test3.jpg","test4.jpg","test5.jpg","test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"]


# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 10

# NUmber of thresolds
Level = 20

# Select general parameters for all optimizers (population size, number of iterations) ....
common_params = {"PopulationSize": 30, "Iterations": 100}

# Select migration parameters
mig_params = {"mig_freq": 5, "mig_rate": 0.3}

# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": False,
    "Export_boxplot": False,
}

run(optimizers, objectivefunc, NumOfRuns, common_params, mig_params, export_flags, Level)