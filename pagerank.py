import numpy
#whether exit iteration
def check_stability(current,last,threshold):
    diff=current-last
    for val in diff:
        print(abs(val))
        if abs(val)>threshold:
            return False
    return True

#read M which is already initialized 
def pagerank(M,maxn,beta=0.85,threshold=0.001):
    r_last=r=numpy.ones((maxn,1))/maxn
    while True:
        r=beta*numpy.dot(M,r)+(1-beta)*1/maxn
        if(check_stability(r,r_last,threshold)):
            break
        r_last=r
    return r

#main()
#readfile
maxn=10
M=numpy.zeros((maxn,maxn))
with open('/Users/hanxinlei/tmp/adj_ijk.txt','r') as f:
    for line in f:
        if None==line: 
            continue
        x=int(line.strip().split()[0])
        y=int(line.strip().split()[1])        
        n=int(line.strip().split()[2])
        M[x][y]=n
d=numpy.ones((1,maxn))
d=numpy.dot(d,M)
#init M
for i in range(maxn):
    for j in range(maxn):
        if(d[0][j]>0):
            M[i][j]/=d[0][j]
#rank           
Rank=pagerank(M,maxn,threshold=0.0001)