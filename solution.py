# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:06:19 2016

@author: 
"""


class solution:
    def __init__(self):
        self.fevals = 0
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.diversity = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        #self.lb = 0
        #self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0
        self.champion = 0     # for heterogenous cooperative  model