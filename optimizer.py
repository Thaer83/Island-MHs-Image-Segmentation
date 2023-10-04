# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
from pathlib import Path
"""
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import benchmarks
"""

from Process_image import load_image
from solution import solution

# UDA Algorithms
import My_Algorithms.PSO_UDA as pso
import My_Algorithms.HHO_UDA as hho
import My_Algorithms.GWO_UDA as gwo

# Thresholding Evaluation Functions
import My_Problems.MLTHIMS_CE as CS
import My_Problems.MLTHIMS_Otsu as Otsu
import My_Problems.MLTHIMS_Tsallis as Tsallis



# Pygmo library
import pygmo as pg

# Quality assessment measures
import performance_metric as measures   

# Others
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import math
import csv
import numpy as np
import time
import warnings
import os
import plot_convergence as conv_plot
import plot_boxplot as box_plot
import ast

warnings.simplefilter(action="ignore")


def selector(algorithm, pop, Iter, image_name):

    """
    if algorithm == "PSO":
        algo = pg.algorithm(pso.my_PSO(gen = Iter))
        #algo = pg.algorithm(pg.pso(gen = 500))
        
        #x = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algorithm == "GWO":
        algo = pg.algorithm(gwo.my_GWO(gen = Iter))
        
    elif algorithm == "HHO":
        algo = pg.algorithm(hho.my_HHO(gen = Iter))
    """
    if algorithm == "DE":
        algo = pg.algorithm(pg.de(gen = Iter)) 
    elif algorithm == "iDE":
        algo = pg.algorithm(pg.sade(gen = Iter)) 
    elif algorithm == "GWO":
        algo = pg.algorithm(pg.gwo(gen = Iter)) 
    elif algorithm == "iHS":
        algo = pg.algorithm(pg.ihs(gen = Iter)) 
    elif algorithm == "PSO":
        algo = pg.algorithm(pg.pso(gen = Iter)) 
    elif algorithm == "GA":
        algo = pg.algorithm(pg.sga(gen = Iter)) 
    elif algorithm == "ABC":
        algo = pg.algorithm(pg.bee_colony(gen = Iter)) 
    elif algorithm == "ACO":
        algo = pg.algorithm(pg.gaco(gen = Iter)) 
    else:
        return null
    
    evolved_pop = algo.evolve(pop)
    
    s = solution()
    #---------------------------------
    s.fevals = evolved_pop.problem.get_fevals()
    extra_info = algo.get_extra_info()
    #li = list(extra_info.split("$"))    # li[0]: fevals       li[1]: executionTime       li[2]: convergence
    #res = li[2][1:-1].split(', ')
    # String to list   https://www.tutorialspoint.com/convert-a-string-representation-of-list-into-list-in-python
    #res = ast.literal_eval(li[2])
    s.convergence = []
    
    champion_individual = evolved_pop.champion_x
    champion_individual_fitness = best_fitness2 = evolved_pop.champion_f   #better
    s.bestIndividual = champion_individual
    s.best = champion_individual_fitness
    s.optimizer = algo.get_name()
    s.objfname = image_name
    s.executionTime = 0
    
    #--------------------------------
    return s


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags, Level):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for for the cinvergence
    CnvgHeader = []
    
    # CSV Header for for best thresholds
    ThrHeader = []

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))
        
    for l in range(0, Level):
        ThrHeader.append("THr" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            bestfit = [0] * NumOfRuns
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            fevals = [0] * NumOfRuns
            psnr = [0] * NumOfRuns
            ssim = [0] * NumOfRuns
            uqi = [0] * NumOfRuns
            rmse = [0] * NumOfRuns
            scc = [0] * NumOfRuns
            vifp = [0] * NumOfRuns
            #------------------------load image-------------------------------------
            file_name = 'COVID19-images/CT/' + objectivefunc[j]
            img_attributes = load_image(file_name)
            h = img_attributes[0]
            probR = img_attributes[2]
            problems=[CS.MLTHIMS_CE(hist = h, dim = Level), Otsu.MLTHIMS_Otsu(hist = h, probR = probR, dim = Level), Tsallis.MLTHIMS_Tsallis(hist = h, probR = probR, dim = Level)]
            # problems[0] : Cross_Entropy      problems[1]: Otsu's function      problems[2] : Tsallis function
            udp = problems[0]    
            prob = pg.problem(udp)
            
            #image_name = prob.get_name() + " " + objectivefunc[j]
            image_name = objectivefunc[j]
            #----------------------------------------------------------------------
            for k in range(0, NumOfRuns):
                
                #----------- # The initial population -----------------------------
                pop = pg.population(prob, size = PopulationSize)
                #------------------------------------------------------------------
                #func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                x = selector(optimizer[i], pop, Iterations, image_name)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname
                bestIndividual = x.bestIndividual
                
                #======= Evaluate best solution #==================================
                gray_img_array=np.asarray(img_attributes[1])
                bestIndividual= np.sort(bestIndividual)
                bestIndividual= np.round(bestIndividual)
                regions = np.digitize(gray_img_array, bins = bestIndividual)
                regions = (regions*int(255/(len(bestIndividual)))).astype(np.uint8)   #  len(champion_individual)-1   bad results
                metrics = measures.Quality_Assessment(gray_img_array, regions)
                
                # save segmented image
                seg_file_name = image_name + '_optimizer_' + optimizerName + '_segmented_run_' + str(k) + '.jpg'
                imageio.imwrite(seg_file_name, regions)
                #==================================================================

                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  # just one time to write the header of the CSV file
                            header = np.concatenate(
                                [["Optimizer", "objfname", "bestFit", "ExecutionTime", "fevals", "psnr", "ssim", "uqi", "rmse", "scc", "vifp"], CnvgHeader, ThrHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  # at least one experiment
                        executionTime[k] = x.executionTime
                        bestfit[k] = x.best
                        fevals[k] = x.fevals
                        psnr[k] = metrics[0]
                        ssim[k] = metrics[1][0]
                        uqi[k] = metrics[2]
                        rmse[k] = metrics[3]
                        scc[k] = metrics[4]
                        vifp[k] = metrics[5]
                        #print(x.optimizer , " " , x.objfname, " ", x.convergence )
                        a = np.concatenate(
                            [[x.optimizer, x.objfname, x.best[0], x.executionTime, x.fevals, metrics[0], metrics[1][0], metrics[2], metrics[3], metrics[4], metrics[5]], x.convergence, np.sort(x.bestIndividual)]
                        )
                        writer.writerow(a)
                    out.close()
                    
                    print(optimizerName, "," , objfname, " Run ", k ,  "CE completed")
                #print("time" , executionTime)
                

            #executionTime = list(map(float, executionTime))    # Thaer        
                
            if Export == True:
                ExportToFile = results_directory + "experiment.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = np.concatenate(
                            [["Optimizer", "objfname", "bestFit", "ExecutionTime", "fevals", "psnr", "ssim", "uqi", "rmse", "scc", "vifp"],CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True
                    
                    avgExecutionTime = float("%0.6f" % (sum(executionTime) / NumOfRuns))
                    avgbestfit = float("%0.6f" % (sum(bestfit) / NumOfRuns))
                    avgfevals = float("%0.6f" % (sum(fevals) / NumOfRuns))
                    avgpsnr = float("%0.6f" % (sum(psnr) / NumOfRuns))
                    avgssim = float("%0.6f" % (sum(ssim) / NumOfRuns))
                    avguqi= float("%0.6f" % (sum(uqi) / NumOfRuns))
                    avgrmse= float("%0.6f" % (sum(rmse) / NumOfRuns))
                    avgscc= float("%0.6f" % (sum(scc) / NumOfRuns))
                    avgvifp= float("%0.6f" % (sum(vifp) / NumOfRuns))
                    avgConvergence = np.around(
                        np.mean(convergence, axis=0, dtype=np.float64), decimals=2
                    ).tolist()
                    a = np.concatenate(
                        [[optimizerName, objfname, avgbestfit, avgExecutionTime, avgfevals, avgpsnr, avgssim, avguqi, avgrmse, avgscc, avgvifp],avgConvergence]
                    )
                    writer.writerow(a)
                out.close()

    if Export_convergence == True:
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Export_boxplot == True:
        box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
