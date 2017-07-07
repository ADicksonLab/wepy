a2a = np.random.rand(3, 3)
walkerwt = [ 1/3 ,1/3 , 1/3]
amp = [ 1, 1, 1]
nwalk = 3
mergedist = 0.25 # 2.a A

f = decision_maker ( a2a ,walkerwt, amp, nwalk, mergedist)
f.mergeclone ()
print ( f.nwalk )

for p in f.walkerwt :
    print (p)
