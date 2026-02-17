import pandas as pd
import numpy as np
import os
from pathlib import Path

#filetype can be "potential", "spikes" or "stim"
def getFilename(path="Results", filetype="potential", prot=1, scalingFactor=1, tempBranch=32, tempParent=37, 
                gPump=-0.0047891, gNav17=0.10664, gNav18=0.24271, gNav19=9.4779e-05, 
               gKs=0.0069733, gKf=0.012756, gH=0.0025377, gKdr=0.018002, gKna=0.00042, vRest=-55,
               sine=False, ampSine=0.1, extracell_rec=None, Nav17_PEPD=False, Ks_reg=1.0):
    #old
    '''
    fileSuffix=('_Prot'+str(prot)+'_scalingFactor'+str(scalingFactor)
                +'_tempBranch'+str(tempBranch)+'_tempParent'+str(tempParent)
                +'_gPump'+str(gPump)+'_gNav17'+str(gNav17)+'_gNav18'+str(gNav18)+'_gNav19'+str(gNav19)
                +'_gKs'+str(gKs)+'_gKf'+str(gKf)+'_gH'+str(gH)+'_gKdr'+str(gKdr)+'_gKna'+str(gKna)+'_vRest'+str(vRest)+'.csv')
    
    fileSuffix=('_Prot'+str(prot)
                +'_gPump'+str(gPump)+'_gNav17'+str(gNav17)+'_gNav18'+str(gNav18)+'_gNav19'+str(gNav19)
                +'_gKs'+str(gKs)+'_gKf'+str(gKf)+'_gH'+str(gH)+'_gKdr'+str(gKdr)+'_gKna'+str(gKna)+'_vRest'+str(vRest)
                +'_sine'+str(sine)+'_ampSine'+str(ampSine)+'.csv')
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
                +'_Ks_reg'+str(Ks_reg)
                +'.csv')
    if filetype=='extracellular':
        fileSuffix = (fileSuffix[:-4]
                +'_electr_xyz_um'+str(extracell_rec['electr_xyz_um'])
                +'_cond_SPERm'+str(extracell_rec['cond_SPERm'])
                +'.npy')
    filename = Path(path) / (filetype + fileSuffix)
    return filename

#filetype can be "potential", "spikes" or "stim"
def getData(path="Results", filetype="potential", prot=1, scalingFactor=1, tempBranch=32, tempParent=37, 
        gPump=-0.0047891, gNav17=0.10664, gNav18=0.24271, gNav19=9.4779e-05, 
        gKs=0.0069733, gKf=0.012756, gH=0.0025377, gKdr=0.018002, gKna=0.00042, vRest=-55,
        sine=False, ampSine=0.1):

    filename=getFilename(path, filetype, prot, scalingFactor, tempBranch, tempParent, 
        gPump, gNav17, gNav18, gNav19, 
        gKs, gKf, gH, gKdr, gKna, vRest,
        sine, ampSine)
    if os.path.exists(filename):
        data = pd.read_csv(filename)#, index_col=0)#, parse_dates=True, header=None)
        return data
    return None


def calculateLatency(data_aps, data_stim, norm=True):
    l = np.zeros(len(data_stim))
    j=0
    i=0
    while i < len(data_stim):
        if j < len(data_aps):
            point = data_aps["Axon 3 1"][j]-data_stim["StimTime"][i]
            if point>0:
                l[i]=point
                j=j+1
                i=i+1
            else:
                j=j+1
        else:
            break
    #normalize
    if norm:
        first=l[0]
        j=0
        for i in l:
            l[j]=(i/first-1)*100
            j=j+1
    return l

