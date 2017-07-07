import numpy as np
import math 
class decision_maker:
    def __init__(self, a2a, walkerwt, nwalk, mergedist):	
        self.a2a = a2a
        self.walkerwt = walkerwt
        self.nwalk = nwalk
        self.mergedist = mergedist 
        self.verbose = False
        self.amp =[ 1 for i in range(self.nwalk)]
        self.copy_struct = [ i for i in range(self.nwalk) ]
    def mergeclone (self, ):
        wtfac = np.zeros(self.nwalk)
        pmin = 1e-12
        pmax = 0.1
        lpmin = math.log(pmin/100)
        dpower = 4
        nobalance = False
        
        if not nobalance:        
         # check for previous walker exits:  clone walkers to compensate
            if  np.sum(self.amp) < self.nwalk :  
                nclone = self.nwalk - np.sum(self.amp)
                (spread,wsum) = self.__calcspread(lpmin,dpower)
                cloneind = wsum.argmax()
                self.amp[cloneind] += nclone   

            # calculate initial spread function value
            (spread,wsum) = self.__calcspread(lpmin,dpower)
            print("Starting variance:",spread)

            productive = 1
            while productive :
                productive = 0
                # find min and max wsums, alter @self.amp
                minwsum = 1.0
                maxwsum = 0.0
                minwind = None
                maxwind = None
                for i in range (0, self.nwalk-1):
                    # determine eligibility and find highest wsum (to clone) and lowest wsum (to merge)
                    if self.amp[i] >= 1 or self.walkerwt[i] > pmin:  # find the most clonable walker
                        if wsum[i] > maxwsum:
                            maxwsum = wsum[i]
                            maxwind = i
                    if self.amp[i] == 1 or  self.walkerwt[i] < pmax:  # find the most mergeable walker
                        if wsum[i] < minwsum: # convert or to and  (important)
                            minwsum = wsum[i]
                            minwind = i 

                # does minwind have an eligible merging partner?
                closedist = self.mergedist
                closewalk = None
                if minwind is not None and maxwind is not None and minwind != maxwind:
                    for j in range(0,self.nwalk):
                        if j != minwind and j != maxwind:
                            if (self.a2a[minwind][j] < closedist and self.amp[j] == 1 and self.walkerwt[j] < pmax):
                                closedist = self.a2a[minwind][j]
                                closewalk = j
            # did we find a closewalk?
                if minwind is not None and maxwind is not None and closewalk is not None:
                    if self.amp[minwind] != 1:
                        die("Error! minwind", minwind, "has self.amp =",self.amp[minwind])
                    if self.amp[closewalk] != 1:
                        die("Error! closewalk", closewalk, "has self.amp=",self.amp[closewalk])


                    # change self.amp
                    tempsum = self.walkerwt[minwind] + self.walkerwt[closewalk]
                    self.amp[minwind] = self.walkerwt[minwind]/tempsum
                    self.amp[closewalk] = self.walkerwt[closewalk]/tempsum
                    self.amp[maxwind] += 1 

                    # re-determine spread function, and wsum values
                    (newspread, wsum) = self.__calcspread(lpmin ,dpower)

                    if newspread > spread:
                        print("Variance move to",newspread,"accepted")
                        productive = 1
                        spread = newspread

                        # make a decision on which walker to keep (minwind, or closewalk)
                        r = np.random.random(self.walkerwt[closewalk] + self.walkerwt[minwind])
                        if r < self.walkerwt[closewalk]:
                            # keep closewalk, get rid of minwind
                            self.__merge(minwind,closewalk)
                        else:
                            # keep minwind, get rid of closewalk
                            self.__merge(closewalk, minwind)

                        (newspread, wsum) = self.__calcspread(lpmin ,dpower)
                        print("variance after selection:",newspread);

                    else: # Not productive
                        self.amp[minwind] = 1
                        self.amp[closewalk] = 1
                        self.amp[maxwind] -= 1

            # end of while productive loop:  done with cloning and merging steps
            # set walkerdist arrays
            walkerdist = wsum

            # Perform the cloning and merging steps
            # start with an empty free walker array, add all walkers with amp = 0
            freewalkers =[]
            for r1walk in range(0,self.nwalk):		
                if self.amp[r1walk] == 0 :		
                    freewalkers.append(r1walk)
 
            # for each self.amp[i] > 1, clone!
            for r1walk in range(0,self.nwalk):
                if self.amp[r1walk] > 1:
                    nclone = self.amp[r1walk]-1
                    inds = []
                    for i in range(0,nclone):
                        try:
                            tind = freewalkers.pop()
                        except:
                            raise("Error! Not enough free walkers!")
                        if r1walk == tind:
                            raise("Error!  free walker is equal to the clone!")
                        else: 
                            inds.append(tind )

                    newwt=self.walkerwt[r1walk]/(nclone+1)
                    self.walkerwt[r1walk] = newwt
                    for tind in inds:
                        self.walkerwt[tind]= newwt
                        walkerdist[tind]=walkerdist[r1walk]
                        self.__copystructure(r1walk,tind)
                        if (self.verbose):
                            print("walker",r1walk,"cloned into",tind)
                            # update self.a2a array for final variance calculation (copy r1walk rows/columns to tind's rows/columns)
                            self.a2a[tind][r1walk] = 0
                            self.a2a[r1walk][tind] = 0
                            for i in range(0,self.nwalk):
                                self.a2a[tind][i] = self.a2a[r1walk][i]
                                self.a2a[i][tind] = self.a2a[i][r1walk]
         # done cloning and meging				
            if len(freewalkers) > 0:
                raise("Error! walkers left over after merging and cloning")

    # end of if nobalance == True
        if self.verbose :
            (newspread,wsum) = self.__calcspread(lpmin,dpower)                
            for i in range(0 , self.nwalk -1 ):
                walkerdist[i] = wsum [i]
            print("final variance check:",newspread)


    def __calcspread(self, lpmin, dpower):
        spread = 0
        wsum = np.zeros(self.nwalk)
        wtfac = np.zeros(self.nwalk)

         # determine weight factors                                                                                
        for i in range(self.nwalk):
            if self.walkerwt[i] > 0 and self.amp[i] > 0:
                wtfac[i] = math.log(self.walkerwt[i]/self.amp[i]) - lpmin
            else:
                wtfac[i] = 0
            if wtfac[i] < 0:
                wtfac[i] = 0

        for i in range(self.nwalk-1):
            if self.amp[i] > 0:
                for j in range(i+1,self.nwalk):
                    if self.amp[j] > 0:
                        d = self.a2a[i][j] ** dpower * wtfac[i]*wtfac[j];
                        spread += d * self.amp[i] * self.amp[j];
                        wsum[i] += d * self.amp[j];
                        wsum[j] += d * self.amp[i];
 
        return(spread,wsum)
 
    def __merge(self, i, j):
        if self.verbose:
            print("walker",i,"_","merged into",j)
        self.walkerwt[j] += self.walkerwt[i]
        self.walkerwt[i] = 0
        self.amp[i] = 0
        self.amp[j] = 1


    def __copystructure(self, a, b):
        # this function will copy the simulation details from walker a to walker b
        self.copy_struct[a] = b
        return a+b 

# if __name__ == '__main__':
    
#     a2a = np.random.rand(3, 3)
#     walkerwt = [ 1/3 ,1/3 , 1/3]
#     amp = [ 1, 1, 1]
#     nwalk = 3
#     mergedist = 0.25 # 2.a A

#     f = decision_maker ( a2a ,walkerwt, amp, nwalk, mergedist)
#     f.mergeclone ()
#     print ( f.nwalk )

#     for p in f.walkerwt :
#         print (p)
    
