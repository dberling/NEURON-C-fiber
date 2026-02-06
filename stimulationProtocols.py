from neuron import h
import numpy as np

def setStimulationProtocol(axon, prot, previousStim):
    i=0
    iclamps = []
    vec = []
    delay = 0
    if prot == -1:
        for i in range(1,6):
            vec.append(i*8000)
        lastPulse=5*8000
        delay=lastPulse+500
        '''
        #20 pulses at 0.125 Hz -> one pulse every 8 seconds
        for i in range(1,21):
            vec.append(i*8000)
        lastPulse=20*8000
        delay=lastPulse+500
        '''
    elif prot == 0:#Test protocol (10 pulses, 500 ms)
        vec = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
        delay = 5000
    elif prot == 1:#single pulse after 2 ms
        vec = [2]
        delay = 500
    elif prot == 2:#Protocol from Barbara
        vec, delay = getStimProt()
    elif prot == 3:#High frequency protocol from Tigerholm
        vec, delay = getTigerholmHighfreq()
    elif prot == 4:#Low frequency protocol from Tigerholm
        vec, delay = getTigerholmLowfreq()
    elif prot == 5:#Test protocol (10 pulses, 1000 ms)
        vec = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
        delay = 10000
    elif prot == 6:#Test protocol (begining of Barbaras protocol)
        nrStim=30
        #regular pulses
        for i in range(nrStim):
            vec.append(i*4000)
            
        #4 extra pulses at 10Hz
        vec.append(nrStim*4000-1300)
        vec.append(nrStim*4000-1200)
        vec.append(nrStim*4000-1100)
        vec.append(nrStim*4000-1000)

        #regular pulses
        nrStim2=nrStim+20
        for i in range(nrStim,nrStim2):
            vec.append(i*4000)

        delay = (nrStim2-1)*4000+1000

    elif prot == 7:#Test protocol - 1s
        nrStim=30
        #regular pulses
        for i in range(nrStim):
            vec.append(i*4000)
        
        for j in range(3):
            #4 extra pulses at 10Hz
            vec.append(nrStim*4000-1300)
            vec.append(nrStim*4000-1200)
            vec.append(nrStim*4000-1100)
            vec.append(nrStim*4000-1000)

            #regular pulses
            nrStim2=nrStim+10
            for i in range(nrStim,nrStim2):
                vec.append(i*4000)
            nrStim=nrStim2
        
        delay = (nrStim2-1)*4000+1000

    elif prot == 8:#Test protocol - 2s
        nrStim=30
        #regular pulses
        for i in range(nrStim):
            vec.append(i*4000)
        
        for j in range(3):
            #4 extra pulses at 10Hz
            vec.append(nrStim*4000-2300)
            vec.append(nrStim*4000-2200)
            vec.append(nrStim*4000-2100)
            vec.append(nrStim*4000-2000)

            #regular pulses
            nrStim2=nrStim+10
            for i in range(nrStim,nrStim2):
                vec.append(i*4000)
            nrStim=nrStim2
        
        delay = (nrStim2-1)*4000+1000

    elif prot == 9:#Test protocol - 0.5s
        nrStim=30
        #regular pulses
        for i in range(nrStim):
            vec.append(i*4000)
        
        for j in range(3):
            #4 extra pulses at 10Hz
            vec.append(nrStim*4000-800)
            vec.append(nrStim*4000-700)
            vec.append(nrStim*4000-600)
            vec.append(nrStim*4000-500)

            #regular pulses
            nrStim2=nrStim+10
            for i in range(nrStim,nrStim2):
                vec.append(i*4000)
            nrStim=nrStim2
        
        delay = (nrStim2-1)*4000+1000
 
    elif prot==10:#elID
        #20 pulses at 0.125 Hz -> one pulse every 8 seconds
        for i in range(1,21):
            vec.append(i*8000)
        lastPulse=20*8000
        #20 pulses at 0.25 Hz -> one pulse every 4 seconds
        for i in range(1,21):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+20*4000
        #30 pulses at 0.5 Hz -> one pulse every 2 seconds
        for i in range(1,31):
            vec.append(lastPulse+i*2000)
        delay=lastPulse+30*2000+1000

    elif prot==11:#elID with recovery
        #20 pulses at 0.125 Hz -> one pulse every 8 seconds
        for i in range(1,21):
            vec.append(i*8000)
        lastPulse=20*8000
        #20 pulses at 0.25 Hz -> one pulse every 4 seconds
        for i in range(1,21):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+20*4000
        #30 pulses at 0.5 Hz -> one pulse every 2 seconds
        for i in range(1,31):
            vec.append(lastPulse+i*2000)
        lastPulse=lastPulse+30*2000
        #20 pulses at 0.25 Hz -> one pulse every 4 seconds
        for i in range(1,21):
            vec.append(lastPulse+i*4000)
        delay=lastPulse+20*4000+1000

    elif prot == 12:
        #Protocol from Delayed responses to electrical stimuli reflect C-fiber responsiveness in human microneurography, Schmelz, 1995
        #one, two and four extra pulses, distance to next regular: 500 ms
        ##################################################################
        #30 pulses at 0.25 Hz -> one pulse every 4 seconds
        #to stabilize latency
        offset=0
        for i in range(1,31):
            vec.append(offset+i*4000)
        lastPulse=offset+30*4000
        #one extra pulse, distance to next regular: 500 ms
        vec.append(lastPulse+3500)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #two extra pulse, 200 ms apart, distance to next regular: 500 ms
        vec.append(lastPulse+3300)
        vec.append(lastPulse+3500)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #four extra pulse, 200 ms apart, distance to next regular: 500 ms
        vec.append(lastPulse+2900)
        vec.append(lastPulse+3100)
        vec.append(lastPulse+3300)
        vec.append(lastPulse+3500)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        delay=lastPulse+1000

    elif prot == 13:
        #Protocol from Delayed responses to electrical stimuli reflect C-fiber responsiveness in human microneurography, Schmelz, 1995
        #one, two and four extra pulses, distance to next regular: 1000 ms
        ##################################################################
        #30 pulses at 0.25 Hz -> one pulse every 4 seconds
        #to stabilize latency
        offset=0
        for i in range(1,31):
            vec.append(offset+i*4000)
        lastPulse=offset+30*4000
        #one extra pulse, distance to next regular: 1000 ms
        vec.append(lastPulse+3000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #two extra pulse, 200 ms apart, distance to next regular: 1000 ms
        vec.append(lastPulse+2800)
        vec.append(lastPulse+3000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #four extra pulse, 200 ms apart, distance to next regular: 1000 ms
        vec.append(lastPulse+2400)
        vec.append(lastPulse+2600)
        vec.append(lastPulse+2800)
        vec.append(lastPulse+3000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        delay=lastPulse+1000

    elif prot == 14:
        #Protocol from Delayed responses to electrical stimuli reflect C-fiber responsiveness in human microneurography, Schmelz, 1995
        #one, two and four extra pulses, distance to next regular: 3000 ms
        ##################################################################
        #30 pulses at 0.25 Hz -> one pulse every 4 seconds
        #to stabilize latency
        offset=0
        for i in range(1,31):
            vec.append(offset+i*4000)
        lastPulse=offset+30*4000
        #one extra pulse, distance to next regular: 3000 ms
        vec.append(lastPulse+1000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #two extra pulse, 200 ms apart, distance to next regular: 3000 ms
        vec.append(lastPulse+800)
        vec.append(lastPulse+1000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        #four extra pulse, 200 ms apart, distance to next regular: 3000 ms
        vec.append(lastPulse+400)
        vec.append(lastPulse+600)
        vec.append(lastPulse+800)
        vec.append(lastPulse+1000)
        #10 regular pulses
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+10*4000
        delay=lastPulse+1000
    elif prot==15:#2 Hz, 3min
        #30 pulses at 2 Hz -> two pulses every second
        offset=0
        for i in range(1,361):
            vec.append(offset+i*500)
        lastPulse=360*500
        #recovery: 10 pulses at 1/4 Hz -> one pulse every 4 seconds
        for i in range(1,11):
            vec.append(lastPulse+i*4000)
        delay=lastPulse+10*4000+100
    elif prot==16:#old Doppelpuls
        #20 pulses at 0.25 Hz -> one pulse every 4 seconds
        #to stabilize latency
        offset=0
        for i in range(1,21):
            vec.append(offset+i*4000)
        lastPulse=offset+20*4000
        extras=[2000, 1750, 1500, 1250, 1000, 750, 500, 250, 150, 100, 75, 50, 40, 30, 20, 10]#distance to next regular puls
        numReg=5#number of regular pulses
        for e in extras:
            #one extra pulse
            vec.append(lastPulse+4000-e)
            #regular pulses
            for i in range(1,numReg+1):
                vec.append(lastPulse+i*4000)
            lastPulse=lastPulse+numReg*4000
        delay=lastPulse+1000
    elif prot==17:#following frequency - 5Hz
        offset=0
        numberOfPulses=25
        for i in range(1,numberOfPulses+1):
            vec.append(offset+i*200)
        lastPulse=offset+numberOfPulses*200
        delay=lastPulse+1000
    elif prot==18:#following frequency - 10Hz
        offset=0
        numberOfPulses=25
        for i in range(1,numberOfPulses+1):
            vec.append(offset+i*100)
        lastPulse=offset+numberOfPulses*100
        delay=lastPulse+1000
    elif prot==19:#following frequency - 20Hz
        offset=0
        numberOfPulses=25
        for i in range(1,numberOfPulses+1):
            vec.append(offset+i*50)
        lastPulse=offset+numberOfPulses*50
        delay=lastPulse+1000
    elif prot==20:#following frequency - 50Hz
        numberOfPulses=4
        numberOfBursts=6
        lastPulse=0
        for j in range(1, numberOfBursts+1):
            for i in range(1,numberOfPulses+1):
                vec.append(lastPulse+i*20)
            lastPulse=lastPulse+numberOfPulses*20+180
            
        delay=lastPulse+1000
    elif prot==21:#following frequency - 100Hz
        numberOfPulses=4
        numberOfBursts=6
        lastPulse=0
        for j in range(1, numberOfBursts+1):
            for i in range(1,numberOfPulses+1):
                vec.append(lastPulse+i*10)
            lastPulse=lastPulse+numberOfPulses*10+190
            
        delay=lastPulse+1000
    elif prot=="FF_5Hz_ADS":#following frequency - indirect
        offset=0
        #20 pulses, 1/4Hz
        numberOfPulses=20
        for i in range(1,numberOfPulses+1):
            vec.append(offset+i*4000)
        lastPulse=offset+numberOfPulses*4000
               
        numberOfPulses=2
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+4000-(numberOfPulses+1-i)*200)
        vec.append(lastPulse+4000)
        lastPulse=lastPulse+4000
        
        numberOfPulses=4
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+numberOfPulses*4000
        
        numberOfPulses=4
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+4000-(numberOfPulses+1-i)*200)
        vec.append(lastPulse+4000)
        lastPulse=lastPulse+4000
        
        numberOfPulses=4
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+numberOfPulses*4000
        
        numberOfPulses=6
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+4000-(numberOfPulses+1-i)*200)
        vec.append(lastPulse+4000)
        lastPulse=lastPulse+4000
        
        numberOfPulses=4
        for i in range(1,numberOfPulses+1):
            vec.append(lastPulse+i*4000)
        lastPulse=lastPulse+numberOfPulses*4000
        
        delay=lastPulse+1000
        
    else:#Get protocol from file
        vec, delay = getProtFromFile(prot)
        
        vecPrevious =[]
        if previousStim==True:
            #set stimulation of 2Hz (one pulse every 0.5 seconds)for 3min/180s 
            for i in range(0, 1800, 5):
                vecPrevious.append(i*100)
            #print(vecPrevious)
            #set stimulation of 0.25Hz (one pulse every 4 seconds) for 3min/180s 
            for i in range(180, 360, 4):
                vecPrevious.append(i*1000)
            #print(vecPrevious)
            
            #offset for vec
            for i in range(len(vec)):
                vec[i]=vec[i]+360*1000
            
            #put previous stimulation in front of vec
            vec = vecPrevious + vec
            delay = delay+360*1000
        
    print("Stimulation: "+str(vec))
    print("Duration: "+str(delay))
    
    for t in vec:
        #current clamp
        i = h.IClamp(axon(1))
        i.delay = t # ms
        #working
        #i.dur = 0.2 # ms
        #i.amp = 2.5 # nA
        #like MNG
        i.dur = 0.5 # ms
        i.amp = 0.07 * 10# nA
        
        #working
        #i.dur = 5 # ms 
        #i.amp = 0.02 # nA
        #not working
        #i.dur = 5 # Tigerholm
        #i.amp = 0.01 #Tigerholm

        iclamps.append(i)
            
    return iclamps, delay, vec

def setStimulationSine(axon, prot=0, amplitude=0.1):
    i=0
    iclamps = []
    vec = []
    delay = 0
        
    if prot==0:#single sine wave after 2ms
        vec = [2]
        delay = 600
        #current clamp for sine wave
        i = h.SinClamp(axon(1))
        i.delay = 2 # ms #starttime
        i.dur = 250 # ms #duration
        i.pkamp = amplitude # nA #amplitude
        i.freq = 4 #Hz, frequency
        iclamps.append(i)
    if prot==1:#multiple sine waves
        vec = [2]
        delay = 500*5
        #current clamp for sine wave
        i = h.SinClamp(axon(1))
        i.delay = 2 # ms #starttime
        i.dur = 250*5 # ms #duration -> 5 sine waves of 250ms duration
        i.pkamp = amplitude # nA #amplitude
        i.freq = 4 #Hz, frequency
        iclamps.append(i)
        
    return iclamps, delay, vec

def getStimProt():
    vec=[]
    delay = 0
    RegPulse = 4000 #waittime between the regular pulses (underlying frequency)
    extraPulse=100 #waittime between the extra pulses
    for j in range(3):
        if j == 0:
            distNextReg = 2000 #distance between last of the extra pulses and next regular pulse
        elif j == 1:
            distNextReg = 1000
        elif j == 2:
            distNextReg = 200
        #repeat 10 times
        for i in range(10):
            #10 regular pulses
            vec.append(delay)
            for jj in range(9):
                delay = delay+RegPulse
                vec.append(delay)

            #4 extra pulses
            dist = RegPulse-distNextReg-3*extraPulse
            delay = delay+dist
            vec.append(delay)
            for jj in range(3):
                delay = delay+extraPulse
                vec.append(delay)

            #distance until next regular pulse
            delay = delay+distNextReg
    #print(vec)
    return vec, delay

def getTigerholmHighfreq():
    vec=[]
    delay = 10
    vec.append(delay)
    for x in range(360):#360 pulses with 2Hz frequency
        delay = delay+500
        vec.append(delay)
    for x in range(60):#60 pulses with 0.25Hz frequency
        delay = delay+4000
        vec.append(delay)
    #print(vec)
    delay = delay+100
    return vec, delay

def getTigerholmLowfreq():
    vec=[]
    delay = 10
    vec.append(delay)
    for x in range(20):#20 pulses with 1/8Hz frequency
        delay = delay+8000
        vec.append(delay)
    for x in range(20):#20 pulses with 1/4Hz frequency
        delay = delay+4000
        vec.append(delay)
    for x in range(30):#30 pulses with 1/2Hz frequency
        delay = delay+2000
        vec.append(delay)
    for x in range(20):#20 pulses with 1/4Hz frequency
        delay = delay+4000
        vec.append(delay)
    #print(vec)
    delay = delay+100
    return vec, delay
        
def getProtFromFile(filename):    
    lines = open(filename).readlines()
    #lines = open("TestProt.txt").readlines()
    vec =[]
    for l in lines:
        l = float(l.replace("\n",""))*1000
        vec.append(l)
    vec.sort()
    delay = float(vec[len(vec)-1])+100
    return vec, delay

def getProtFromFile2(filename):    
    lines = open(filename).readlines()
    #lines = open("TestProt.txt").readlines()
    if isinstance(lines[0], str):
        lines.pop(0)#remove first element if text
    vec =[]
    for l in lines:
        split_string = l.split(",", 1)
        substring = split_string[0]
        f = float(substring.replace("\n",""))*1000
        vec.append(f)
    delay = float(vec[len(vec)-1])+100
    return vec, delay

def test():
    vec1, d1 = getProtFromFile2('Protocols/AE1/20_05_13_U1b_reg.txt')
    vec2, d2 = getProtFromFile2('Protocols/AE1/20_05_13_U1b_extra.txt')
    l = list(np.array(vec1) - np.array(vec2))
    return l
        
        
        
        
