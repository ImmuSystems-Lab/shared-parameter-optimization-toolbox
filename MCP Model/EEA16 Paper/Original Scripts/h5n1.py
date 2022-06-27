
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import odeint
import scipy.stats as stats


# from one_state_model import f1
# from one_error import errorfxn
# f rom new_position import new_position 

# Import data
tvec = np.genfromtxt('data_h5n1_logged.csv', usecols = 0, delimiter = ',',
                     skip_header = 1) # timepoints
data5 = np.genfromtxt('data_h5n1_logged.csv', usecols = [1,2,3,4], delimiter = ',',
                     skip_header = 1) # data
data5_log = copy.deepcopy(data5)
data5_log[:,2] = 2**(data5_log[:,2])

std5 =  np.genfromtxt('std_h5n1_logged.csv', usecols = [1,2,3,4], delimiter = ',',
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
    mcp1_err = np.array((np.square(mcp1_interp - mcp1data)/(2*mcp1data)), dtype=float)

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
u05 = ic[1,:]
par5 = np.array([8.44108617e-01,1.03581996e+02,1.16049647e-01,1.84269283e-12, 1.06445178e+00,1.33654271e+00,1.82989952e+07,7.97155418e+07, 7.85980937e+01])
par5 = np.array([7.55398088e-02,1.61181120e+04,1.15326834e-01,3.89299277e-13,6.87365202e-01,1.22144962e+00,1.85636347e+05,3.76551916e+04, 1.11648618e5])
par5 = np.array([7.55398088e-01,1.61181120e+05,1.15326834e-01,3.89299277e-14,6.87365202e-01,1.22144962e+00,1.85636347e+05,3.76551916e+06, 1.11648618e0])
par5 = np.array([8.71955537e-01,8.13383635e+01,1.16408036e-01,3.91285299e-1, 1.07421400e+00,1.34917879e+00,2.52756818e+07,4.73549243e+07, 3.95699115e+02])
par5 = np.array([1.55398088e00,1.61181120e+05,1.15326834e-01,3.89299277e-3,6.87365202e-01,1.22144962e+00,1.85636347e+05,3.76551916e+06, 1.11648618e0, 4])
upper = par5*1000
lower = par5/1000
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
    nruns = 2000001
    nchains = len(beta) 
    npar = len(par5)
    output = 50000
    tmax = 5
    tspan = np.arange(0,tmax+.05,0.05)
    
    # initialize matricies
    yh5n1 = np.zeros([nchains, npar]) #current par
    ychainh5n1 = np.zeros([nchains, nruns, npar]) #all saved pars 
    pyh5n1 = np.zeros([nchains, npar]) #proposed pars
    min_ychainh5n1 = np.zeros([nchains, npar])
    
    Energy = np.zeros([nchains, nruns]) #Energy
    energyh5n1 = np.zeros([nchains, nruns]) #EnergyH1N1
    preswap_Energy = np.zeros([nchains, nruns]) #Energy
    preswap_energyh5n1 = np.zeros([nchains, nruns]) #EnergyH1N1   
    preswap_ychainh5n1 = np.zeros([nchains, nruns, npar]) #all saved pars 

    e = np.zeros([nchains, 1]) #E
    eh5 = np.zeros([nchains, 1]) #EH1
    v_err5 = np.zeros([nchains,nruns])
    ifn_err5 = np.zeros([nchains,nruns])
    mcp1_err5 = np.zeros([nchains,nruns])
    accept = np.zeros([nchains, 1]) 
    reject = np.zeros([nchains, 1])
    accept_swap = np.zeros([nchains, 1])
    reject_swap = np.zeros([nchains, 1])
    
    pars2change = np.array([0,1,2,3,4,5,6,7,8,9], dtype=int)    
    parsnotchanged = np.array([], dtype=int)
    
    for c in np.arange(0, nchains): 
        ifn5_interp = np.interp(tspan, tvec, data5[:,1])
        ychainh5n1[c, 0, :] = copy.deepcopy(par5[:])
        yh5n1[c, :] = copy.deepcopy(par5[:])
        sol = g(f4, tspan, u05, par5)
        energyh5n1[c, 0], v_err5[c,0], ifn_err5[c,0], mcp1_err5[c,0] = errorfxn(tvec, tspan, sol, data5, std5)
        Energy[c, 0] = energyh5n1[c, 0] 
        
    for run in np.arange(1, nruns): 
        for c in np.arange(0, nchains): 
            pyh5n1[c,:] = new_position(yh5n1[c,:], upper, lower, np.arange(0,len(par5)), beta[c]) 
            sol = g(f4, tspan, u05, pyh5n1[c,:])
            eh5[c], v_err5[c,run], ifn_err5[c,run], mcp1_err5[c,run] = errorfxn(tvec, tspan, sol, data5, std5)    
            e[c] = (eh5[c])
            delta = e[c] - Energy[c, run-1]
            prob = np.min((1, np.exp(-beta[c]*delta)))
            rand_prob = np.random.uniform(0,1) # np.random.uniform(0,1)
            if rand_prob < prob:
                yh5n1[c,:] = copy.deepcopy(pyh5n1[c,:])
                accept[c] += 1
                Energy[c, run] = copy.deepcopy(e[c])
                energyh5n1[c, run] = copy.deepcopy(eh5[c])
            else:
                Energy[c, run] = copy.deepcopy(Energy[c,run-1])
                energyh5n1[c, run] = copy.deepcopy(energyh5n1[c, run-1])
                reject[c] += 1
    
        #SWAPPING
        cind = len(beta)-1
        while cind >= 1:
            deltaEchain = Energy[cind, run] - Energy[cind-1, run]
            deltaBeta = beta[cind] - beta[cind-1]
            r = np.random.uniform(0,1)
            if np.minimum(1,np.exp(deltaEchain*deltaBeta)) > r:
                preswap_Energy[cind, run] = copy.deepcopy(Energy[cind,run])
                preswap_energyh5n1[cind, run] = copy.deepcopy(energyh5n1[cind,run])
                preswap_ychainh5n1[cind, run, :] = copy.deepcopy(yh5n1[cind,:])
                #print('Energy', Energy, 'yh1n1', yh1n1)
                Etmp = np.array([copy.deepcopy(Energy[cind-1, run]), copy.deepcopy(energyh5n1[cind-1, run]), copy.deepcopy(yh5n1[cind-1,:])]);
                Energy[cind-1, run], energyh5n1[cind-1, run], yh5n1[cind-1,:] = np.array([copy.deepcopy(Energy[cind, run]), copy.deepcopy(energyh5n1[cind, run]), copy.deepcopy(yh5n1[cind, :])])
                Energy[cind, run], energyh5n1[cind, run], yh5n1[cind,:]= Etmp
                #print('POST Energy', Energy, 'yh1n1', yh1n1) 
                accept_swap[nchains-cind, 0] += 1;
            else:
                reject_swap[nchains-cind, 0] += 1;
            cind -= 1    
        for c in np.arange(0,nchains): 
            ychainh5n1[c, run, :] = copy.deepcopy(yh5n1[c,:])
    
            if (run)%output == 0: 
                print("At iteration", run, "for chain temperature", beta[c])
                acceptance_rate = accept[c]/(accept[c]+reject[c])
                print("Acceptance rate = ", acceptance_rate)
                print("accept: ", accept[c], " reject: ", reject[c])
                swapping_rate = accept_swap/(accept_swap + reject_swap)
                print("Swapping rate = ", swapping_rate)
                print("Min energy = ", np.min(Energy[c,0:run-1]))
                print("Max energy = ", np.max(Energy[c,0:run-1]))
                print(np.shape(ychainh5n1[c,np.argmin(Energy[c,:run]),:]), np.shape(min_ychainh5n1))
                min_ychainh5n1[c,:] = ychainh5n1[c,np.argmin(Energy[c,:run]),:]
                print(np.argmin(Energy[c,:run]))
                print("Minimum energy H5N1", np.array2string(min_ychainh5n1[c], separator = ","))

    # Save data output
    np.savetxt("Energy5.csv", Energy, delimiter = ",")
    np.savetxt("energyh5n1.csv", energyh5n1, delimiter = ",")
    np.savetxt("min_ychainh5n1.csv", min_ychainh5n1, delimiter = ",")

    np.savetxt("PSEnergy1.csv", preswap_Energy, delimiter = ",")
    np.savetxt("PSenergyh5n1.csv", preswap_energyh5n1, delimiter = ",")
    np.savetxt("PSychainh5n1.csv", preswap_ychainh5n1[0,:,:], delimiter = ",")

    np.savetxt("ychainh1n1_0.csv", ychainh5n1[0,:,:], delimiter = ",")
    np.savetxt("ychainh1n1_1.csv", ychainh5n1[1,:,:], delimiter = ",")
    np.savetxt("ychainh1n1_2.csv", ychainh5n1[2,:,:], delimiter = ",")
    np.savetxt("ychainh1n1_3.csv", ychainh5n1[3,:,:], delimiter = ",")
    np.savetxt("ychainh1n1_4.csv", ychainh5n1[4,:,:], delimiter = ",")
    np.savetxt("ychainh1n1_5.csv", ychainh5n1[5,:,:], delimiter = ",")
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)
    # Plot h5n1 states 
    prev = g(f4, tspan, u05, par5)
    sol = g(f4, tspan, u05, min_ychainh5n1[0])
    print(sol)
    plt.figure(figsize = (10,5))
    fig, axs = plt.subplots(ncols = 3, figsize = (10,5))
    axs[1].errorbar(tvec, data5[:,1], std5[:,1], marker = 'o', linestyle = 'None')
    axs[1].plot(tspan, sol[:,1], linestyle = '--')
    axs[1].plot(tspan, prev[:,1], linestyle = '--')
    axs[1].set_title('IFN')
    axs[1].set_xlabel('days')
    axs[1].set_ylabel('Gene expression')

    axs[2].errorbar(tvec, data5[:,2], std5[:,2], marker = 'o', linestyle = 'None')
    axs[2].plot(tspan, np.log2(sol[:,2]), linestyle = '--')
    axs[2].plot(tspan, np.log2(prev[:,2]), linestyle = '--')
    axs[2].set_title('MCP1')
    axs[2].set_ylabel('log2(pg/mL)')   
    axs[2].set_xlabel('days')

    axs[0].errorbar(tvec, data5[:,0], std5[:,0], marker = 'o', linestyle = 'None')
    axs[0].plot(tspan, sol[:,0], linestyle = '--')
    axs[0].plot(tspan, prev[:,0], linestyle = '--')
    axs[0].set_title('Virus')
    axs[0].set_xlabel('days')
    axs[0].set_ylabel('PFU/g')
    plt.legend(['New', 'Old'])
    plt.tight_layout()
    # plt.show()
    fig.savefig('h5n1data_scaledv.pdf')

    #Plot error per strain
    fig = plt.figure()
    plt.plot(range(0,nruns), energyh5n1[0,:])
    plt.plot(range(0,nruns), energyh5n1[1,:])
    # plt.plot(range(0,nruns), energyh5n1[1,:])
    plt.plot(range(0,nruns), energyh5n1[2,:])
    # plt.plot(range(0,nruns), energyh5n1[2,:])
    plt.plot(range(0, nruns), energyh5n1[3,:])
    # plt.legend(['h1n1 low T', 'h5n1 low T', 'h1n1 high T', 'h5n1 high T', 'h1n1 highest T', 'h5n1 highest T'])
    plt.legend(['T = 0.99', 'T = 0.8', 'T = 0.3', 'T = 0.05'])
    # plt.show()
    fig.suptitle("Total Error")
    plt.xlabel("Iteration")
    plt.ylabel('Error')
    fig.savefig('error5.pdf')

    # Plot log error per strain
    fig = plt.figure()
    plt.plot(range(0,nruns), np.log10(energyh5n1[0,:]))
    plt.plot(range(0,nruns), np.log10(energyh5n1[1,:]))
    # plt.plot(range(0,nruns), energyh5n1[1,:])
    plt.plot(range(0,nruns), np.log10(energyh5n1[2,:]))
    # plt.plot(range(0,nruns), energyh5n1[2,:])
    plt.plot(range(0, nruns), np.log10(energyh5n1[3,:]))
    # plt.legend(['h1n1 low T', 'h5n1 low T', 'h1n1 high T', 'h5n1 high T', 'h1n1 highest T', 'h5n1 highest T'])
    plt.legend(['T = 0.99', 'T = 0.8', 'T = 0.3', 'T = 0.05'])
    # plt.show()i
    fig.suptitle("Total Log Error")
    plt.xlabel("Iteration")
    plt.ylabel('Error')
    fig.savefig('log_error5.pdf')
   
    # Plot parameter distributions
    plt.figure(figsize = (15,5))
    fig, axes = plt.subplots(nrows=3, ncols = 4)
    axes = axes.flatten()
    lab = np.array([r'$k$', r'$K$', r'$r_{IFN, V}$', r'$d_V$', r'$p_{V, IFN}$', r'$d_{IFN}$', r'$k_1$', r'$k_2$', r'$d_{MCP1}$', r'$n$'])
    # print(ychainh1n1[0,:,:])
    # print(ychainh1n1[1,:,:])
    for i in np.arange(len(par5)):
        axes[i,].hist(np.log10(ychainh5n1[0,:,i]), alpha = 0.5)
        axes[i,].set_title(lab[i])
        plt.tight_layout()
        # axes[i,].legend(['H1N1'])
        plt.xlim([np.log10(lower[i]), np.log10(upper[i])])
    fig.savefig('param_dists5.pdf')

    # Plot state error 
    fig, axes = plt.subplots(nrows = 1, ncols = 3)
    axes[0].plot(range(nruns), np.log10(v_err5[0,:]))
    axes[0].set_title("Log Virus Error")
    axes[1].plot(range(nruns), np.log10(ifn_err5[0,:]))
    axes[1].set_title("Log IFN Error")
    axes[2].plot(range(nruns), np.log10(mcp1_err5[0,:]))
    axes[2].set_title("Log MCP1 Error")
    plt.tight_layout()
    axes[0].set_xlabel("Iteration")
    axes[1].set_ylabel('Error')
    #axes[0].legend(['H1N1'])
    #axes[1].legend(['H1N1'])
    #axes[2].legend(['H1N1'])
    fig.savefig('state_error5.png')

 
