
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import odeint
import scipy.stats as stats


# from one_state_model import f1
# from one_error import errorfxn
# from new_position import new_position 

# Import data
tvec = np.genfromtxt('data_h1n1_logged.csv', usecols = 0, delimiter = ',',
                     skip_header = 1) # timepoints
data1 = np.genfromtxt('data_h1n1_logged.csv', usecols = [1,2,3,4], delimiter = ',',
                     skip_header = 1) # data

std1 =  np.genfromtxt('std_h1n1_logged.csv', usecols = [1,2,3,4], delimiter = ',',
                     skip_header = 1) #std dev of data
ic =  np.genfromtxt('logged_4state_init.csv', usecols = [2,3,4], delimiter = ',',
                     skip_header = 1)#unlogged IC's of data

def f4(u, t, p, ic):
    """2 state ODE model of Virus and IFN"""
    k, big_k, r_ifn_v, d_v, p_v_ifn, d_ifn, k1, k2, d_mcp1, n1 = p
    v, ifn, mcp1 = u
    v0, ifn0, mcp10 = ic
    dv = k*v*(1-v/big_k) - r_ifn_v*(ifn-ifn0)*v - d_v*v
    difn = p_v_ifn*v - d_ifn*(ifn - ifn0)
    dmcp1 = (k1*(ifn-ifn0)**n1)/(k2+(ifn-ifn0)**n1)-(mcp1-mcp10)*d_mcp1
    return [dv, difn, dmcp1]

def errorfxn(tvec, tspan, result, data, std):
    """Return residual error"""
    vdata = data[:,0]; vstd = std[:,0]
    ifndata = data[:,1]; ifnstd = std[:,1]
    mcp1data = data[:,2]; mcp1std = std[:,2]

    vcount = np.count_nonzero(~np.isnan(vdata))
    v_interp = np.interp(tvec, tspan, result[:,0])
    v_err = np.array(np.square(v_interp - vdata)/(2*vdata), dtype=float)

    ifn_interp = np.interp(tvec, tspan, result[:,1])
    ifn_err = np.array(np.square(ifn_interp - ifndata)/(2*ifndata), dtype=float)

    mcp1_interp = np.log2(np.interp(tvec, tspan, result[:,2]))
    mcp1_err = np.array((np.square(mcp1_interp - mcp1data)/(2*(mcp1data))), dtype=float)

    if np.any(np.isfinite([v_err[1:], ifn_err[1:,], mcp1_err[1:,]]) == False):
        v_err = 1e10
        ifn_err = 1e10
        mcp1_err = 1e10
    else:
        v_err = np.sum(v_err)
        ifn_err = np.nansum(ifn_err)
        mcp1_err = np.sum(mcp1_err)
    return  np.nansum(v_err) + np.nansum(ifn_err) + np.nansum(mcp1_err), v_err, ifn_err, mcp1_err

def new_position(par, upper, lower, ind, beta):
    npar = len(par)
    new_par = np.zeros(np.shape(par))
    for i in ind:
        scale = (np.log(upper[i]) - np.log(lower[i]))/2
        new_par[i] = par[i]*np.exp(0.003/beta*np.random.normal(0,1)*scale)
        if new_par[i] > upper[i]:
            new_par[i] = (lower[i]/upper[i])*new_par[i]
        if new_par[i] < lower[i]:
            new_par[i] = (upper[i]/lower[i])*new_par[i]
    return new_par

# set initial conditions
u01 = ic[0,:]
par1 = np.array([8.71955537e-01,8.13383635e+01,1.16408036e-01,3.91285299e-1, 1.07421400e+00,1.34917879e+00,2.52756818e+01,4.73549243e+01, 3.95699115e3])
par1 = np.array([9.09103515e-01,1.13502578e+02,1.15089610e-01,7.33591015e-02, 1.07563662e+00,1.37687733e+00,1.04703131e+03,2.68467731e+02, 2.15375074e-00])
par1 = np.array([1.07196983e+00,5.47883655e+01,1.09859969e-01,1.51641261e-01, 1.04699116e+00,1.31112263e+00,3.85300588e+04,1.23022285e+04, 2.44368290e+00])
par1 = np.array([1.03333088e+00,5.49487657e+01,1.09968709e-01,1.19091210e-01, 1.05234498e+00,1.32093139e+00,1.46655305e+04,4.81088769e+03, 2.26443442e+00])
par1 = np.array([9.69879700e-01,4.93445493e+01,1.09979661e-01,4.55187955e-02, 1.07353753e+00,1.35879588e+00,1.58230131e+05,2.48306446e+03, 7.27395681e-1])
par1 = np.array([8.71955537e-01,8.13383635e+01,1.16408036e-01,3.91285299e-1, 1.07421400e+00,1.34917879e+00,2.52756818e+01,4.73549243e+01, 3.95699115e0])
par1 = np.array([1.01919593e+00,5.56057923e+01,1.11353372e-01,1.00976637e-01, 1.07455155e+00,1.35788530e+00,1.48596269e+04,7.44769257e+05, 9.68813927e+00])
par1 = np.array([1.20661300e+00,4.24411714e+01,1.22629159e-01,1.42465755e-01, 1.06113975e+00,1.34057640e+00,5.19416140e+05,1.72404728e+07,1.74651677e+01, 6])
par1 = np.array([3.20661300e+00,4.24411714e+00,8.22629159e-02,1.42465755e-01, 1.06113975e-02,1.34057640e+00,5.19416140e+05,1.72404728e+07,1.74651677e+02, 6])
upper = par1*1000
lower = par1/1000
lower[9] = 1
upper[9] = 10

def g(model, t, u0, p):
    """Return integration of model"""
    sol = odeint(model, u0, t, args=(p,u0))
    return sol

n_arr = np.array([6.4])
for n in n_arr:
    print('*****STARTING N = ', n)
    # initialize parameters
    beta = np.array([.99, .9, .8, .4, .2, .05]) #, 0.1, 0.05]) 
    eps = 0.005/np.sqrt(beta)
    nstates = 3
    nruns =2000001
    nchains = len(beta) 
    npar = len(par1)
    output = 50000
    tmax = 5
    tspan = np.arange(0,tmax+.05,0.05)
    
    # initialize matricies
    yh1n1 = np.zeros([nchains, npar]) #current par
    ychainh1n1 = np.zeros([nchains, nruns, npar]) #all saved pars 
    pyh1n1 = np.zeros([nchains, npar]) #proposed pars
    min_ychainh1n1 = np.zeros([nchains, npar])
    
    Energy = np.zeros([nchains, nruns]) #Energy
    energyh1n1 = np.zeros([nchains, nruns]) #EnergyH1N1
    e = np.zeros([nchains, 1]) #E
    eh1 = np.zeros([nchains, 1]) #EH1
    v_err1 = np.zeros([nchains,nruns])
    ifn_err1 = np.zeros([nchains,nruns])
    mcp1_err1 = np.zeros([nchains,nruns])
    accept = np.zeros([nchains, 1]) 
    reject = np.zeros([nchains, 1])
    accept_swap = np.zeros([nchains, 1])
    reject_swap = np.zeros([nchains, 1])
    
    pars2change = np.array([0,1,2,3,4,5,6,7,8,9], dtype=int)    
    parsnotchanged = np.array([], dtype=int)
    
    for c in np.arange(0, nchains): 
        ifn1_interp = np.interp(tspan, tvec, data1[:,1])
        ychainh1n1[c, 0, :] = copy.deepcopy(par1[:])
        yh1n1[c, :] = copy.deepcopy(par1[:])
        sol = g(f4, tspan, u01, par1)
        energyh1n1[c, 0], v_err1[c,0], ifn_err1[c,0], mcp1_err1[c,0] = errorfxn(tvec, tspan, sol, data1, std1)
        Energy[c, 0] = energyh1n1[c, 0] 
    for run in np.arange(1, nruns): 
        for c in np.arange(0, nchains): 
            pyh1n1[c,:] = new_position(yh1n1[c,:], upper, lower, np.arange(0,len(par1)), beta[c]) 
            sol = g(f4, tspan, u01, pyh1n1[c,:])
            eh1[c], v_err1[c,run], ifn_err1[c,run], mcp1_err1[c,run] = errorfxn(tvec, tspan, sol, data1, std1)
            e[c] = (eh1[c])
            delta = e[c] - Energy[c, run-1]
            prob = np.min((1, np.exp(-beta[c]*delta)))
            rand_prob = np.random.uniform(0,1) # np.random.uniform(0,1)
            if rand_prob < prob:
                yh1n1[c,:] = copy.deepcopy(pyh1n1[c,:])
                accept[c] += 1
                Energy[c, run] = copy.deepcopy(e[c])
                energyh1n1[c, run] = copy.deepcopy(eh1[c])
            else:
                Energy[c, run] = copy.deepcopy(Energy[c,run-1])
                energyh1n1[c, run] = copy.deepcopy(energyh1n1[c, run-1])
                reject[c] += 1
    
        #SWAPPING
        cind = len(beta)-1 
        while cind >= 1: 
            deltaEchain = Energy[cind, run] - Energy[cind-1, run]
            deltaBeta = beta[cind] - beta[cind-1] 
            r = np.random.uniform(0,1)
            if np.minimum(1,np.exp(deltaEchain*deltaBeta)) > r: 
                Etmp = copy.deepcopy(Energy[cind-1, run]); 
                Energy[cind-1, run] = copy.deepcopy(Energy[cind, run]);
                Energy[cind, run] = copy.deepcopy(Etmp);
    
                Etmp = copy.deepcopy(energyh1n1[cind-1, run]); 
                energyh1n1[cind-1, run] = copy.deepcopy(energyh1n1[cind, run]);
                energyh1n1[cind, run] = copy.deepcopy(Etmp);
    
                Extmp1 = copy.deepcopy(yh1n1[cind-1,:]);
                yh1n1[cind-1,:] = copy.deepcopy(yh1n1[cind, :]);
                yh1n1[cind, :] = copy.deepcopy(Extmp1);
                accept_swap[nchains-cind, 0] += 1;
            else:
                reject_swap[nchains-cind, 0] += 1;
            cind -= 1
    
        for c in np.arange(0,nchains): 
            ychainh1n1[c, run, :] = copy.deepcopy(yh1n1[c,:])
    
            if (run)%output == 0: 
                print("At iteration", run, "for chain temperature", beta[c])
                acceptance_rate = accept[c]/(accept[c]+reject[c])
                print("Acceptance rate = ", acceptance_rate)
                print("accept: ", accept[c], " reject: ", reject[c])
                swapping_rate = accept_swap/(accept_swap + reject_swap)
                print("Swapping rate = ", swapping_rate)
                print("Min energy = ", np.min(Energy[c,0:run-1]))
                print("Max energy = ", np.max(Energy[c,0:run-1]))
                print(np.shape(ychainh1n1[c,np.argmin(Energy[c,:run]),:]), np.shape(min_ychainh1n1))
                min_ychainh1n1[c,:] = ychainh1n1[c,np.argmin(Energy[c,:run]),:]
                print(np.argmin(Energy[c,:run]))
                print("Minimum energy H1N1", np.array2string(min_ychainh1n1[c], separator = ","))
    
    # Save data output
    np.savetxt("Energy1.csv", Energy, delimiter = ",")
    np.savetxt("energyh1n1.csv", energyh1n1, delimiter = ",")
    np.savetxt("min_ychainh1n1.csv", min_ychainh1n1, delimiter = ",")
    np.savetxt("ychainh1n1.csv", ychainh1n1[0,:,:], delimiter = ",")
    
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params) 
    # Plot h1n1 states 
    prev = g(f4, tspan, u01, par1)
    print(prev)
    sol = g(f4, tspan, u01, min_ychainh1n1[0])
    print(sol)
    plt.figure(figsize = (10,5))
    fig, axs = plt.subplots(ncols = 3, figsize = (10,5))
    axs[1].errorbar(tvec, data1[:,1], std1[:,1], marker = 'o', linestyle = 'None')
    axs[1].plot(tspan, sol[:,1], linestyle = '--')
    axs[1].plot(tspan, prev[:,1], linestyle = '--')
    axs[1].set_title('IFN')
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('Gene expression')
    
    axs[2].errorbar(tvec, data1[:,2], std1[:,2], marker = 'o', linestyle = 'None')
    axs[2].plot(tspan, np.log2(sol[:,2]), linestyle = '--')
    axs[2].plot(tspan, np.log2(prev[:,2]), linestyle = '--')
    axs[2].set_title('MCP1')
    axs[2].set_ylabel('log2(pg/mL)')
    axs[2].set_xlabel('days')
    
    axs[0].errorbar(tvec, data1[:,0], std1[:,0], marker = 'o', linestyle = 'None')
    axs[0].plot(tspan, sol[:,0], linestyle = '--')
    axs[0].plot(tspan, prev[:,0], linestyle = '--')
    axs[0].set_title('Virus')
    axs[0].set_xlabel('days')
    axs[0].set_ylabel('PFU/g')
    plt.legend(['New', 'Old'])
    plt.tight_layout()
    # plt.show()
    fig.savefig('h1n1data_scaledv.pdf')
    
    #Plot error per strain
    fig = plt.figure()
    plt.plot(range(0,nruns), energyh1n1[0,:])
    plt.plot(range(0,nruns), energyh1n1[1,:])
    # plt.plot(range(0,nruns), energyh5n1[1,:])
    plt.plot(range(0,nruns), energyh1n1[2,:])
    # plt.plot(range(0,nruns), energyh5n1[2,:])
    plt.plot(range(0, nruns), energyh1n1[3,:]) 
    # plt.legend(['h1n1 low T', 'h5n1 low T', 'h1n1 high T', 'h5n1 high T', 'h1n1 highest T', 'h5n1 highest T'])
    plt.legend(['T = 0.99', 'T = 0.8', 'T = 0.3', 'T = 0.05'])
    # plt.show()
    fig.suptitle("Total Error")
    plt.xlabel("Iteration")
    plt.ylabel('Error')    
    fig.savefig('error1.pdf')
   
    # Plot log error per strain
    fig = plt.figure()
    plt.plot(range(0,nruns), np.log10(energyh1n1[0,:]))
    plt.plot(range(0,nruns), np.log10(energyh1n1[1,:]))
    # plt.plot(range(0,nruns), energyh5n1[1,:])
    plt.plot(range(0,nruns), np.log10(energyh1n1[2,:]))
    # plt.plot(range(0,nruns), energyh5n1[2,:])
    plt.plot(range(0, nruns), np.log10(energyh1n1[3,:]))
    # plt.legend(['h1n1 low T', 'h5n1 low T', 'h1n1 high T', 'h5n1 high T', 'h1n1 highest T', 'h5n1 highest T'])
    plt.legend(['T = 0.99', 'T = 0.8', 'T = 0.3', 'T = 0.05'])
    # plt.show()i
    fig.suptitle("Total Log Error") 
    plt.xlabel("Iteration") 
    plt.ylabel('Error') 
    fig.savefig('log_error1.pdf')
 
    # Plot parameter distributions
    plt.figure(figsize = (15,5))
    fig, axes = plt.subplots(nrows=3, ncols = 4)
    axes = axes.flatten()
    lab = np.array([r'$k$', r'$K$', r'$r_{IFN, V}$', r'$d_V$', r'$p_{V, IFN}$', r'$d_{IFN}$', r'$k_1$', r'$k_2$', r'$d_{MCP1}$', r'$n$'])
    # print(ychainh1n1[0,:,:])
    # print(ychainh1n1[1,:,:])
    for i in np.arange(len(par1)):
        axes[i,].hist(np.log10(ychainh1n1[0,:,i]), alpha = 0.5)
        axes[i,].set_title(lab[i])
        plt.tight_layout()
        # axes[i,].legend(['H1N1'])
    fig.savefig('param_dists1.pdf')
    
    # Plot state error 
    fig, axes = plt.subplots(nrows = 1, ncols = 3)
    axes[0].plot(range(nruns), np.log10(v_err1[0,:]))
    axes[0].set_title("Log Virus Error")
    axes[1].plot(range(nruns), np.log10(ifn_err1[0,:]))
    axes[1].set_title("Log IFN Error")
    axes[2].plot(range(nruns), np.log10(mcp1_err1[0,:]))
    axes[2].set_title("Log MCP1 Error")
    plt.tight_layout()
    axes[0].set_xlabel("Iteration")
    axes[1].set_ylabel('Error')    
    #axes[0].legend(['H1N1'])
    #axes[1].legend(['H1N1'])
    #axes[2].legend(['H1N1'])
    fig.savefig('state_error1.png')
    
