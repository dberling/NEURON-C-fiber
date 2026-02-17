from neuron import h
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
import time
import math
import os
from os import path
from pathlib import Path

from defineCell import *
from stimulationProtocols import *
#from saveData import *
#from plot import *
from extracellularRecording import get_ERM

# Use RESULTS_DIR environment variable if set, otherwise default to "Results"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "Results"))
RESULTS_DIR.mkdir(exist_ok=True)

#prot: stimulation protocol number or filename
#gPump: conductance of pump
#gNav17: conductance of Nav 1.7
#gNav18: conductance of Nav 1.8
#dt: step size in time, if set to zero, CVode is activated
#previousStim: sets a pre stimulation before the regular stimulation protocol, if the protocol is loaded from file
def run(prot=1, scalingFactor=1,  dt=0, previousStim=False, tempBranch=32, tempParent=37, 
        gPump=-0.0047891, gNav17=0.10664, gNav18=0.24271, gNav19=9.4779e-05, 
        gKs=0.0069733, gKf=0.012756, gH=0.0025377, gKdr=0.018002, gKna=0.00042,vRest=-55,
        sine=False, ampSine=0.1, extracell_rec=None, Nav17_PEPD=False):
    """

    extracell_rec: None or dict()
        Record extracellularly by setting to dict(electr_xyz_um, cond_SPERm)
        with electrode position (x,y,z) in um and conductivity in S/m.
    """
    
    #start timer
    tic = time.perf_counter()
    
    #define morphology as in Tigerholm
    axon = [0,0,0,0,0,0]
    axon_names = ['extra1', 'branch', 'branchingPoint', 'parent', 'extra2', 'extra3']
    axon_params = dict(
      extra1=        
        dict(L=   100*scalingFactor, diam=(0.25, 0.25), nseg=round(    10*scalingFactor)),
      branch=        
        dict(L= 20000*scalingFactor, diam=(0.25, 0.25), nseg=round(   400*scalingFactor)),
      branchingPoint=
        dict(L=  5000*scalingFactor, diam=(0.25, 1.00), nseg=round(   100*scalingFactor)),
      parent=        
        dict(L=100000*scalingFactor, diam=(1.00, 1.00), nseg=round(10*200*scalingFactor)),
      extra2=        
        dict(L=   100*scalingFactor, diam=(1.00, 1.00), nseg=round(    10*scalingFactor)),
      extra3=        
        dict(L=   100*scalingFactor, diam=(1.00, 1.00), nseg=round(    10*scalingFactor)),
    )
    # set x0 such that branchingPoint begins at x=0:
    x0 = -1 * (axon_params['extra1']['L'] + axon_params['branch']['L'])
    for idx, axon_name in enumerate(axon_names):
        axon[idx] = h.Section(name=axon_name)
        # construct along x
        x1 = x0 + axon_params[axon_name]['L']
        diam = axon_params[axon_name]['diam']
        nseg = axon_params[axon_name]['nseg']
        h.pt3dadd(
            h.Vector([x0, x1]),
            h.Vector([0,0]), 
            h.Vector([0,0]), 
            h.Vector([diam[0], diam[1]]), 
            sec=axon[idx]
        )
        axon[idx].nseg = nseg
        # set new starting point of next seg:
        x0 = x1
    # compute shape (needed for LFPy affects x3d(), y3d(), z3d()
    h.define_shape()

    for i in range(6):
        axon[i].Ra = 35.5
        axon[i].cm = 1
    
    #connect parts
    axon[1].connect(axon[0])
    axon[2].connect(axon[1])
    axon[3].connect(axon[2])
    axon[4].connect(axon[3])
    axon[5].connect(axon[4])
    
    '''
    #print all axon information to check if everything is right
    for i in range(6):
        print(axon[i].psection())
    '''
    
    '''
    h.topology()
    ps = h.PlotShape(False)  # False tells h.PlotShape not to use NEURON's gui
    ps.plot(plt)
    plt.show(0)
    '''
    
    for i in range(6):
        condFactor=1
        if i==0 or i==5:
            condFactor=1e-5
        insertChannels(axon[i], condFactor, gPump, gNav17, gNav18, gNav19, gKs, gKf, gH, gKdr, gKna)

    if Nav17_PEPD:
        # shift voltage in alpha_h and beta_h for the inactivation gate h of Nav17 to simulate Nav17 PEPD mutation
        for sec in axon:
            for seg in sec:
                seg.nattxs.pepd_vshift = -20.0 # mV

    
    cvode = h.CVode()
    if dt==0:
        #variable time step integration method
        cvode.active(1)
        print("CVode active")
    else:
        h.dt=dt

    if sine:
        stim, delay, vec = setStimulationSine(axon[0], prot, ampSine)
        print("Stimulation: Sine Wave")
    else:
        stim, delay, vec = setStimulationProtocol(axon[0], prot, previousStim)
        print("Stimulation: Square Pulse")

    spTimes = h.Vector()
    apc = h.APCount(axon[1](0))
    apc.thresh = -10
    apc.record(spTimes)

    spTimes2 = h.Vector() 
    apc2 = h.APCount(axon[1](0.25))
    apc2.thresh = -10
    apc2.record(spTimes2)
    
    spTimes3 = h.Vector() 
    apc3 = h.APCount(axon[1](0.5))
    apc3.thresh = -10
    apc3.record(spTimes3)
    
    spTimes4 = h.Vector() 
    apc4 = h.APCount(axon[1](0.75))
    apc4.thresh = -10
    apc4.record(spTimes4)
    
    spTimes5 = h.Vector() 
    apc5 = h.APCount(axon[1](1))
    apc5.thresh = -10
    apc5.record(spTimes5)
    
    spTimes6 = h.Vector() 
    apc6 = h.APCount(axon[3](0))
    apc6.thresh = -10
    apc6.record(spTimes6)
    
    spTimes7 = h.Vector() 
    apc7 = h.APCount(axon[3](0.25))
    apc7.thresh = -10
    apc7.record(spTimes7)
    
    spTimes8 = h.Vector() 
    apc8 = h.APCount(axon[3](0.5))
    apc8.thresh = -10
    apc8.record(spTimes8)
    
    spTimes9 = h.Vector() 
    apc9 = h.APCount(axon[3](0.75))
    apc9.thresh = -10
    apc9.record(spTimes9)
    
    spTimes10 = h.Vector() 
    apc10 = h.APCount(axon[3](1))
    apc10.thresh = -10
    apc10.record(spTimes10)

    if extracell_rec != None:
        # turn on fast membrane current recordings:
        cvode.use_fast_imem(1)
        # time
        tvec = h.Vector()
        tvec.record(h._ref_t)
        # record membrane currents for extracellular recording
        imem = []
        for sec in axon:
            for seg in sec:
                imem_vec = h.Vector()
                imem_vec.record(seg._ref_i_membrane_)
                imem.append(imem_vec)

    #simulation
    Vrest=vRest
    h.finitialize(Vrest)

    tempCelsius1 = tempBranch
    tempCelsius2 = tempParent
    setTemp(axon[0], tempCelsius1)    
    setTemp(axon[1], tempCelsius1)
    setTemp(axon[2], (tempCelsius1+tempCelsius2)/2)
    setTemp(axon[3], tempCelsius2)
    setTemp(axon[4], tempCelsius2)
    setTemp(axon[5], tempCelsius2)
    

    h.fcurrent()
    
    for i in range(6):
        balance(axon[i], Vrest)

    
    #create folder
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    
    #create filename
    if isinstance(prot, str) and "/" in prot:
        prot = prot.split("/", 2)
        print(prot)
        prot = prot[2]


    
    #filename can't be too long, full path can't be more than 255 characters
    #therefore values are rounded!
    '''
    fileSuffix=('_Prot'+str(prot)+'_scalingFactor'+str(scalingFactor)
                +'_tempBranch'+str(tempBranch)+'_tempParent'+str(tempParent)
                +'_gPump'+str(gPump)+'_gNav17'+str(gNav17)+'_gNav18'+str(gNav18)+'_gNav19'+str(gNav19)
                +'_gKs'+str(gKs)+'_gKf'+str(gKf)+'_gH'+str(gH)+'_gKdr'+str(gKdr)+'_gKna'+str(gKna)+'_vRest'+str(vRest)+'.csv')
    '''
    fileSuffix=('_Prot'+str(prot)
                +'_gPump'+str(round(gPump,7))
                +'_gNav17'+str(round(gNav17,7))
                +'_gNav18'+str(round(gNav18,7))
                +'_gNav19'+str(round(gNav19,7))
                +'_gKs'+str(round(gKs,7))
                +'_gKf'+str(round(gKf,7))
                +'_gH'+str(round(gH,7))
                +'_gKdr'+str(round(gKdr,7))
                +'_gKna'+str(round(gKna,7))
                +'_vRest'+str(vRest)
                +'_sine'+str(sine)
                +'_ampSine'+str(ampSine)
                +'_Nav17_PEPD'+str(Nav17_PEPD)
                +'.csv')
    filename = RESULTS_DIR / ('potential' + fileSuffix)
    fileSpikes = RESULTS_DIR / ('spikes' + fileSuffix)
    
    #Stimulation times
    fileStim = RESULTS_DIR / ('stim' + fileSuffix)
    with open(fileStim,'w', newline='') as f:
        csv.writer(f).writerow(["StimTime"])
        
    for stimTime in vec:
        with open(fileStim,'a', newline='') as f:
            csv.writer(f).writerow([stimTime])
    
    potentials = []
    spike_times = []
    #start simulation
    tstop = delay
    #h.continuerun(tstop)
    i=0
    while(h.t<tstop):
        potentials.append({
            "Time"       : h.t, 
            "Axon 1 0"   : axon[1](0).v, 
            "Axon 1 0.25": axon[1](0.25).v, 
            "Axon 1 0.5" : axon[1](0.5).v, 
            "Axon 1 0.75": axon[1](0.75).v, 
            "Axon 1 1"   : axon[1](1).v, 
            "Axon 3 0"   : axon[3](0).v, 
            "Axon 3 0.25": axon[3](0.25).v, 
            "Axon 3 0.5" : axon[3](0.5).v,
            "Axon 3 0.75": axon[3](0.75).v,
            "Axon 3 1"   : axon[3](1).v,
        })

        if i<len(spTimes) and i<len(spTimes2) and i<len(spTimes3) and i<len(spTimes4) and i<len(spTimes5) and i<len(spTimes6) and i<len(spTimes7) and i<len(spTimes8) and i<len(spTimes9) and i<len(spTimes10):
            print("Time: "+str(h.t))
            print("AP number:"+str(i+1))
            print("Axon 1 0.5: " + str(spTimes3[i]))
            print("Axon 3 0.5: " + str(spTimes8[i]))
            spike_times.append({
                "Axon 1 0"   : spTimes[i], 
                "Axon 1 0.25": spTimes2[i], 
                "Axon 1 0.5" : spTimes3[i], 
                "Axon 1 0.75": spTimes4[i], 
                "Axon 1 1"   : spTimes5[i], 
                "Axon 3 0"   : spTimes6[i], 
                "Axon 3 0.25": spTimes7[i], 
                "Axon 3 0.5" : spTimes8[i],
                "Axon 3 0.75": spTimes9[i],
                "Axon 3 1"   : spTimes10[i],
            })
            i=i+1
            
        #step 
        h.fadvance()

    # save potentials and spike times
    pd.DataFrame(potentials).to_csv(filename, index=False)
    pd.DataFrame(spike_times).to_csv(fileSpikes, index=False)

    #plot 
    #l = plotLatency(spTimes10, vec)
    
    #print(v3)
    #for x in spTimes: print(x)

    if extracell_rec != None:
        # set up extracellular recording matrix (acting on membrane currents of segs):
        ERM = get_ERM(
                fiber_secs_ordered=axon, 
                electr_xyz_um=extracell_rec['electr_xyz_um'], 
                cond_SPERm=extracell_rec['cond_SPERm'], 
                save_geometry=False # for debugging assumed fiber geometry
        )
        # convert imem to numpy
        imem_np = np.array(
                [list(imem_vec) for imem_vec in imem]
        )
        # calculate extracellularly recorded potential:
        V_ex = ERM @ imem_np
        # write to npy
        fileSuffix_ = (fileSuffix[:-4]
                +'_electr_xyz_um'+str(extracell_rec['electr_xyz_um'])
                +'_cond_SPERm'+str(extracell_rec['cond_SPERm'])
                +'.npy')
        fileExtracellular = RESULTS_DIR / ('extracellular' + fileSuffix_)
        np.save(fileExtracellular, np.array([np.array(tvec), V_ex.flatten()]))
    
    toc = time.perf_counter()
    print(f"Simulation time: {(toc - tic)/60:0.4f} min")
    #return t, v_mid

'''
def range_assignment(sec, var, start, stop):
    """linearly assign values between start and stop to each segment.var in section"""
    import numpy as np
    for seg, val in zip(sec, np.linspace(start, stop, sec.nseg)):
        setattr(seg, var, val)
'''
