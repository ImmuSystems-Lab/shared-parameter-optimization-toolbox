###Imports###
import scipy
import numpy as np
from numpy import random
import array
from numba import jit
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import multiprocessing
from multiprocessing import Pool, cpu_count
import itertools
import pandas as pd
import math
from deap import base
from deap import creator
from deap import tools
import random as rand
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import contextlib


###Function definitions###
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            
#Turn Lower and Upper bounds vectors into a packed bounds vector, using a sharing genome
def genomeBounds(bndsmin, bndsmax, genome, depth):
    size = len(genome)
    for i in range(0,size*(depth-1)):
        idex = i%size
        if genome[idex]==True:
            bndsmin = np.append(bndsmin, bndsmin[idex])
            bndsmax = np.append(bndsmax, bndsmax[idex])

    return np.stack((bndsmin,bndsmax))


#Unpack a short vector and a shared genome into full length parameter vectors
#Credit to Plonetheus on StackOverflow for finding a 10x faster way to do this operation than my original loops!
def convertVectors(par, G, C):
    """
    example:
    par = [1,2,3,4,5,6,10,30,40,60,100,300,400,600]
    G = [1,0,1,1,0,1]
    C = 2
    res = [[1, 2, 3, 4, 5, 6],[10, 2, 30, 40, 5, 60],[100, 2, 300, 400, 5, 600]]
    """
    
    res = np.empty((C, len(G)))  # result
    r = np.arange(0,len(G)*C)    #How many parameters need to be assigned

    if len(par) > (C-1)*np.sum(G)+len(G): #Predict par length and check for par>needed
        raise ValueError("More parameters have been given than needed")
    else:
        
        for i in r:
            c, p = np.divmod(i,len(G)) #equal to x//y, x%y
            cur=len(G[0:p])
            tot=np.sum(G[0:p])
            res[c,p] = par[p+G[p]*c*int(len(G)-cur+tot)]

    return res


#Normalize data for convergence and regularization
def normalizeData(data, normMethod='none', control=None):
    
    if normMethod=='scale':
        #Scale the data from 0-1
        
        #Get some stats to aid (de)normalization
        denorm = (np.min(data),np.max(data))

        #Normalize the data
        normData = (data-denorm[0])/(denorm[1]-denorm[0])  

        #return the normalized data and stats required for denormalization
        return normData
    
    elif normMethod=='none':
        #Return the original data, no normalization performed
        return data
    
    elif normMethod=='lfc' and control is not None:
        #log fold change the data
        y=np.shape(data)[0]
        normData=np.zeros(y)
        
        for i in range (0,y-1):
            normData[i]=np.log2(data[i]/control)
        
        return normData
    
    elif normMethod=='lfc' and control is None:
        #lfc intended, but no control value provided
        raise ValueError("Control value for log-fold change not provided")
        
    else:
        #Bad normalization method
        raise ValueError("Unrecognized normalization method:",normMethod,
                         ". Acceptable methods are 'none', 'lfc', and 'scale'")

        
#convert standard deviations to coefficients of variation
def STDtoCOV(mean, std):
    #change standard deviation to coefficient of variation
    cov = std/mean
    
    #Prevent divide-by-zero
    Epsilon = 1e-4
    
    #add a small value to 0's in coefficient of variation
    cov = np.where(cov > 1e-5, cov, (cov+Epsilon))
    
    return cov

#return data to a denormalized form
def denormalize(data, denorm):
    mn = denorm[0] #min of original data
    mx = denorm[1] #max of original data
    denormData = data*(mx - mn) + mn
    return denormData

#Picks a uniformly random parameter vector, p, in half interval range of a vector of [low,high) bounds.
def drawP(bounds):
    rng = np.random.default_rng()
    drawnP=rng.uniform(bounds[0,:],bounds[1,:])
    return drawnP

#Takes in a list of strings, pointed to csv's with data, and returns multidimensional arrays of data
def stacker(datcol, stdcol, timecol, target):
    #Needed for stacking data arrays in the loop below
    firstIteration = True

    if len(datcol) != len(stdcol):
        #Raise error
        raise ValueError("Standard Deviation and Data shape mismatch")
    else:
        for i in range(0,len(target)):
            #Read in data
            data=np.genfromtxt(target[i],
                     skip_header=1,delimiter=',',usecols=datcol)
            
            #Read in standard deviations
            std = np.genfromtxt(target[i],
                    skip_header=1,delimiter=',',usecols=stdcol)

            #Read in time points
            tData=np.genfromtxt(target[i],
                     skip_header=1,delimiter=',',usecols=timecol)

            #Preallocate arrays for data, coefficients of variation, and denorming stats
            normData = np.zeros(data.shape)
            cov = np.zeros(std.shape)

            #Normalize the data
            for j in range(0,len(datcol)):
                normData[:,j] = normalizeData(data[:,j])
                cov[:,j] = STDtoCOV(np.mean(data[:,j]),std[:,j])

            if firstIteration:
                #Prepare the data, time, and cov arrays to be stacked depth-wise
                dataStack = np.expand_dims(normData,axis=-1)
                covStack = np.expand_dims(cov,axis=-1)
                tStack = np.expand_dims(tData,axis=-1)
                firstIteration = False

            else:
                #Stack the data, time, and cov arrays
                dataStack = np.concatenate((dataStack,np.expand_dims(normData,axis=-1)),-1)
                covStack = np.concatenate((covStack,np.expand_dims(cov,axis=-1)),-1)
                tStack = np.concatenate((tStack,np.expand_dims(tData,axis=-1)),-1)

        return dataStack, covStack, tStack


#Define an objective function wrapper for SSE
def objective_function(p, fxn, u, tPred, data, dataIndex, tData, genome, cohorts):

    #Split the combined parameter vector
    pUnpacked = convertVectors(p, genome, cohorts)

    #Initialize the error
    error = 0
    
    for l in range(0,len(pUnpacked)):
        pSplit = pUnpacked[l]
        
        #Run the ODE system to find predicted ODE solution
        argTup = np.append(pSplit,u[l][2]) #Add MCP1@t=0. Shouldn't need this for most datasets
        with stdout_redirected():
            sol = odeint(fxn,u[l],tPred[:,l],args=(argTup,)) #Integrate the ODEs
        
        #Check for infs and NaNs in the ODE soution
        if (np.all(np.isfinite(sol))==False):
            #print("Objective Function caught an Inf or NaN")
            error = float("inf")
        else:
            
            #Calculate SSE based on states which are a) given in data and b) present in ODE solution
            #Condition b is always satisfied by any state which satisfies a, 
            #so this check should only catch errors
            for j in np.intersect1d(np.arange(0,sol.shape[-1]),dataIndex):
                
                #Normalize the ODE predictions
                solNorm = sol[:,j]
                
                #Select only predicted points whose time matches data's time 
                pred = solNorm[np.searchsorted(tPred[:,l],tData[:,l], side='right')-1]
                
                #User provided cost function goes here
                #For EEA's MCP model currently, any SSE-like method should work
                #Calculate the SSE for the indicated data set/prediction
                A = (pred - data[:,j,l])**2
                B = 2*data[:,j,l]
                error += np.nansum(A/B)
    
    return error

#Custom step-function for bounded basin hopping
class RandomDisplacementBounds(object):
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
            
        range_step = self.xmax - self.xmin
        min_step = np.maximum(self.xmin - x, -self.stepsize * range_step)
        max_step = np.minimum(self.xmax - x, self.stepsize * range_step)

        RandomStep = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        
        xnew = x + RandomStep

        return xnew
    
#Function to calculate BIC from a genome
def evaluate(individual):
        
    #Tile the bounds according to the genome
    bnds = genomeBounds(bndsmin, bndsmax, individual, tPred.shape[-1])
    
    #Create an initial parameter guess, randomly (cold start)
    #Avoid warm-starts per Jim
    p0 = drawP(bnds)

    #Determine number of cohorts present
    cohorts = tPred.shape[-1]
    
    #Arguments for objective function
    argTuple = tuple((fxn, u, tPred, dataStack, dataIndex, tStack, individual, cohorts))
    
    #Arguments for non-stochastic gradient descent portion of basin hopping
    kwargs = {'method':lclMin,'args':(argTuple),'bounds':scipy.optimize.Bounds(bnds[0,:], bnds[1,:]),
              'tol':lclTol, 'options':{'maxiter':lclIter, 'disp':False, 'ftol':lclTol}}
    
    #Define the custom bounded step function for Basin Hopping. Must be defined after "bnds" are set
    bounded_step = RandomDisplacementBounds(np.array([b for b in bnds[0,:]]),
                                                 np.array([b for b in bnds[1,:]]),
                                                 bhStep)
    
    #Run the basin hopping
    result = basinhopping(objective_function, p0, niter = bhIter, 
                          minimizer_kwargs = kwargs, disp = bhDisp,
                          take_step = bounded_step, T = Temp, 
                          stepsize = bhStep, interval = bhInt, niter_success = bhIterExit)

    #Calculate the Bayesian Information Criterion
    BIC = (len(individual) + np.sum(individual)) * math.log(math.prod(dataStack.shape)) + 2 * result.fun #BIC
    
    return BIC

#Function to generate synthetic data sets, given a noise level (0-1, typically stop at ~0.3), and change parameter values with multi, par
def traj(noise=None, multi=None, par=None, fxn=None):
    #timepoints for solver
    #TODO consider grabbing these from an argument tuple
    #Same goes for initial conditions, parameter values, noise, etc.
    #Maybe randomize the "dataset's" parameters for non-value dependent SPOT analyses
    #Not sure if SPOT is parameter value dependent, structurally, or both/neither
    tSol = (0, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 5)

    #Initial conditions
    u01 = [6.819798, 0, 121.7883333] #h1n1
    u05 = [7.10321, 0, 121.525] #h5n1

    if fxn.__name__ == ("f1"):
        #h1n1
        par1 = np.array((3.27, 10.9, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 1.0E-4, 2.5E-2))
        #h5n1
        par5 = np.array((3.27, 21.8, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 1.0E-4, 2.5E-2))
    elif fxn.__name__ == ("f2"):
        #h1n1
        par1 = np.array((3.27, 10.9, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 2.5E-2))
        #h5n1
        par5 = np.array((3.27, 21.8, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 2.5E-2))
    elif fxn.__name__ == ("f3"):
        #h1n1
        par1 = np.array((3.27, 10.9, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 1.0E-4))
        #h5n1
        par5 = np.array((3.27, 21.8, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45, 1.0E-4))
    elif fxn.__name__ == ("f4"):
        #h1n1
        par1 = np.array((3.27, 10.9, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45))
        #h5n1
        par5 = np.array((3.27, 21.8, 0.00016, 0.000201, 0.1, 1.08, 14700, 18500, 1.04, 5.45))
    else:
        print("No parameter bounds for function handle")
    #Parameter values
    #h1n1
    mcp101 = 121.7883333

    #h5n1
    if multi is not None:
        par5[par] *= multi #Apply a modifier, multi, to H5N1's parameter, par5[par].
    mcp105 = 121.525

    #Add noise to the parameters, then remove zeros
    if noise is not None:
        parN1 = np.random.normal(par1,noise*par1)
        parN1[parN1<0] = 1e-3

        parN5 = np.random.normal(par5,noise*par5)
        parN5[parN5<0] = 1e-3

    #Prepare arguments for solver
    argTup1 = np.append(parN1,mcp101)
    argTup5 = np.append(parN5,mcp105)

    #Solve the model
    with stdout_redirected():
        sol1 = odeint(fxn,u01,tSol,args=(argTup1,),mxstep=5000)
        sol5 = odeint(fxn,u05,tSol,args=(argTup5,),mxstep=5000)

    #Put MCP1 in log2 space
    sol1[:,2] = np.log2(sol1[:,2])
    sol5[:,2] = np.log2(sol5[:,2])
    
    #Reshape the model outputs to match what we'd make from csv's (see stacker() function)
    out1 = np.expand_dims(sol1,axis=-1)
    out5 = np.expand_dims(sol5,axis=-1)
    
    out = np.concatenate((out1,out5),-1)
    
    return out
    
###User Input starts here###

#Create the ODE systems
@jit(nopython=True)
def f1(u, t, p):
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn, k1, k2, d_mcp1, n1, d_v_mcp1, p_ifn_mcp1, mcp10 = p
    v, ifn, mcp1 = u
    dy = (k*v*(1-v/big_k) - r_ifn_v*(ifn)*v - d_v*v - d_v_mcp1*v*(mcp1-mcp10),
    p_v_ifn*v - d_ifn*(ifn) + p_ifn_mcp1*(mcp1-mcp10),
    (k1*(ifn)**n1)/(k2+(ifn)**n1)-(mcp1-mcp10)*d_mcp1)
    return dy

@jit(nopython=True)
def f2(u, t, p):
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn, k1, k2, d_mcp1, n1, d_v_mcp1, mcp10 = p
    v, ifn, mcp1 = u
    dy = (k*v*(1-v/big_k) - r_ifn_v*(ifn)*v - d_v*v - d_v_mcp1*v*(mcp1-mcp10),
    p_v_ifn*v - d_ifn*(ifn),
    (k1*(ifn)**n1)/(k2+(ifn)**n1)-(mcp1-mcp10)*d_mcp1)
    return dy

@jit(nopython=True)
def f3(u, t, p):
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn, k1, k2, d_mcp1, n1, p_ifn_mcp1, mcp10 = p
    v, ifn, mcp1 = u
    dy = (k*v*(1-v/big_k) - r_ifn_v*(ifn)*v - d_v*v,
    p_v_ifn*v - d_ifn*(ifn) + p_ifn_mcp1*(mcp1-mcp10),
    (k1*(ifn)**n1)/(k2+(ifn)**n1)-(mcp1-mcp10)*d_mcp1)
    return dy

@jit(nopython=True)
def f4(u, t, p):
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn, k1, k2, d_mcp1, n1, mcp10 = p
    v, ifn, mcp1 = u
    dy = (k*v*(1-v/big_k) - r_ifn_v*(ifn)*v - d_v*v,
    p_v_ifn*v - d_ifn*(ifn),
    (k1*(ifn)**n1)/(k2+(ifn)**n1)-(mcp1-mcp10)*d_mcp1)
    return dy
        
#Where is the data located? A list of string file locations
target = ('./Data/Real Data/data_h1n1_logged.csv',
          './Data/Real Data/data_h5n1_logged.csv'
         )

#Which column contains time? Same for all data sets 
timecol = 1

#Which column(s) contain data? Same for all data sets
datcol = (2,3,4)

#Which column(s) contain standard deviation? Same for all data sets
stdcol = (6,7,8)

#Get the multidimensional stacked data
#Shouldn't require any user input/modification
dataStack, covStack, tStack = stacker(datcol, stdcol, timecol, target)

#What is the function handle that contains the ODE system? jit encouraged. Same system for all strains
fxn = f4

#Lower and Upper bounds vectors for each parameter in p
#TODO: make this pull from a csv shared for all scripts using a given model
if fxn.__name__ == ("f1"):
    bndsmin = np.array((0, 1e-3, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0))
    bndsmax = np.array((10.0, 50.0, 1.0, 5.0, 5.0, 5.0, 1e5, 1e5, 10.0, 10.0, 10.0, 10.0))
elif fxn.__name__ == ("f2"):
    bndsmin = np.array((0, 1e-3, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0))
    bndsmax = np.array((10.0, 50.0, 1.0, 5.0, 5.0, 5.0, 1e5, 1e5, 10.0, 10.0, 10.0))
elif fxn.__name__ == ("f3"):
    bndsmin = np.array((0, 1e-3, 0, 0, 0, 0, 0, 1e-3, 0, 0, 0))
    bndsmax = np.array((10.0, 50.0, 1.0, 5.0, 5.0, 5.0, 1e5, 1e5, 10.0, 10.0, 10.0))
elif fxn.__name__ == ("f4"):
    bndsmin = np.array((0, 1e-3, 0, 0, 0, 0, 0, 1e-3, 0, 0))
    bndsmax = np.array((10.0, 50.0, 1.0, 5.0, 5.0, 5.0, 1e5, 1e5, 10.0, 10.0))
else:
    print("No parameter bounds for function handle")

#What temperature should the basin hopping algorithm operate with? ~= objective fxn difference between minima
#Default 1.0
Temp = 50

#Known initial conditions. Use [[x1,y1,z1],...,[xn,yn,zn]] for different ICs
u=[[6.819798, 0, dataStack[0][2][0]],[7.10321, 0, dataStack[0][2][1]]]

#Known time span to solve in. Fed directly to the solver! Increase number of points for stiff/unstable systems
t=np.linspace(0,6,100) 

#Ensure that every data point's time is exactly within the predicted tspan
#Shouldn't require any user input/modification
tPred = np.sort(np.unique(np.concatenate((t,tStack[:,0])),0))

#If only 1 data set/model, create a singleton axis of the ODE time points
#Otherwise, repeat the time points along a new axis. One per strain.
#Shouldn't require any user input/modification
if (dataStack.shape[-1] == 1):
    tPred = np.expand_dims(tPred,-1)
else:
    tPred = np.repeat(np.expand_dims(tPred,-1),dataStack.shape[-1],axis=-1)

#Which states should be used for SSE calculations? Applies to all data sets! 
dataIndex = np.array((0,1,2))

#What method should be used for the local minimizer portion of basinhopping?
#Valid, bounded methods are: TNC, Powell, trust-constr, SLSQP, Nelder-Mead, and L-BFGS-B
#Defaults to SLSQP, which is typically the fastest method
lclMin = 'SLSQP'
    
#Tolerance parameter for gradient descent portion of Basin Hopping. Default 1e-3
lclTol=1e-3
    
#Max iterations for local solver. 100-250 typically plenty for moderate ODE systems
lclIter=500

#Maximum number of basinhopping iterations. Default 100
bhIter = 500

#Should basinhopping print progress messages to the console? Defaults to True
bhDisp = False

#Initial step size of basinhopping algorithm. Defaults to 0.5 (50%)
#Taken as a proportion of each parameter's range (upper bound - lower bound)
bhStep = 0.5

#basinhopping will attempt to update Step Size per n iterations to maintain 50% acceptance. Default 10
bhInt = 10

#If no new global minimum is found in n basinhopping iterations, exit. Default 50
bhIterExit = 250

#register DEAP functions
toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate) #Calls basinhopping, returns AIC/BIC

#Brute force approach, i.e., evaluate all permuations

#Protective "main" function required for multiprocessing when on Windows
def main(orgs):
    threads = multiprocessing.cpu_count() #How many threads does the system have?
    reserve = 8 #Number of threads to reserve
    fraction = 1 #1/fraction number of threads will be used
    useable = int(np.max((threads/fraction-reserve,1))) #Actual # of threads to assign
    pool = multiprocessing.Pool(processes=useable) #Create the pool
    toolbox.register("map", pool.imap) #Map the evaluation fxn to the pool
    fitbrute = toolbox.map(toolbox.evaluate, orgs) #Run the mapped evaluations
    pool.close() #Terminate the pool and release threads
    return fitbrute

#Controls actual execution
if __name__ == "__main__":
    method = 'sweep' #Sets which method will be used in sampling.
    #sweep: Sweep over all parameters in a model, increasing the value in one strain in incremental levels.
    #resample: Take multiple genome fitness samples repeatedly, at a given artificial noise level
    
    l = [0,1] #Possible states for each bit in the genome: boolean, 1/0
    genomeLength = bndsmin.size #How many bases the genome will have
    print('Genome length: ' + str(genomeLength))
    allOrganisms = list(itertools.product(l, repeat=genomeLength)) #Create a list of all possible genomes
    orgSampleFraction = 1.0 #Fraction of all possible genomes to sample
    sampledOrganisms = rand.sample(allOrganisms,int(orgSampleFraction*len(allOrganisms))) #Downsample from all possible genomes
    samples = 1 #How many samples to pull at each noise level?
    noise = 0 #Noise percent to apply to underyling parameters

    if method == 'resample':
        parameter = None
        multiplier = None

        #Evaluate all genomes "samples" times
        for i in range(0,samples):

    
            dataStack = traj(noise, multiplier, parameter, fxn)
            fititer = main(sampledOrganisms) #Run the pool-based evaluation of all genomes

            if i==0:
                org_df = pd.DataFrame(sampledOrganisms) 
                org_df.insert(genomeLength,'BIC',fititer)
            else:
                temp_df = pd.DataFrame(sampledOrganisms) #Transform the genome list into a dataframe
                temp_df.insert(genomeLength,'BIC',fititer) #Append fitness to dataframe
                org_df = pd.concat([org_df, temp_df])
                del temp_df

        #TODO have name data info from methods applied/data structures
        z = ('Resamples at ' + str(noise) + ' noise.csv')
        org_df.to_csv(z) #Save result to disk

    elif method == 'sweep':

        #Sweep over parameters and multiplier levels
        levels = (1.25, 2.0, 3.0, 4.0, 5.0) #Multiplier levels for parameters. Note that 1x is the base case, and needn't be repeated
        for i in range(0,genomeLength): 
            for j in levels:
                dataStack = traj(noise, j, i, fxn)
                fititer = main(sampledOrganisms) #Run the pool-based evaluation of all genomes
                org_df = pd.DataFrame(sampledOrganisms) 
                org_df.insert(genomeLength,'BIC',list(fititer))
                z = ('Data/Synthetic Data/Parameter Ratio Sweeps/Model 4/' + 
                str(i) + ' x ' + str(j) + '.csv')
                print(z + ' done')
                org_df.to_csv(z) #Save result to disk
    else:
        print('Method not recognized. Terminating.')
                