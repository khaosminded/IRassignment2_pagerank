import numpy
from scipy.sparse import *

#readfile and convert to spares matrix
def get_Matrix_CSC(fname):
    M=numpy.loadtxt(fname,dtype=int)
    row=M[:,0]
    col=M[:,1]
    data=M[:,2]
    #convert to SQUARE sparse matrix
    m,n=row.max(),col.max()
    row=numpy.r_[row,max(m,n)]
    col=numpy.r_[col,max(m,n)]
    data=numpy.r_[data,0]
    M=coo_matrix((data,(row,col)),dtype=numpy.float64)
    M=M.tocsc()

    #init 1/Si 
    s=numpy.array(M.sum(axis=0))
    for i in range(len(s[0])):
        if s[0][i]!=0:
            for j in range(M.indptr[i],M.indptr[i+1]):
                M.data[j]/=s[0][i]
    return M

#whether exit iteration
def check_stability(current,last,threshold):
    diff=current-last
    for val in diff:
        if abs(val)>threshold:
            return False
    return True

#read M which is already initialized

def pagerank(M,maxn,beta=0.85,threshold=0.001):
    count=0
    r_last=r=numpy.ones((maxn,1))/maxn
    while True:
        count+=1
        r=beta*M.dot(r)+(1-beta)*1/maxn
        if(check_stability(r,r_last,threshold)):
            break
        r_last=r
    return r

#main()
fname='/Users/hanxinlei/Downloads/AdjacencyMatrix.txt'
#fname='/Users/hanxinlei/tmp/adj_ijk.txt'
M=get_Matrix_CSC(fname)
r=pagerank(M,M.shape[0],threshold=1.0e-04)
numpy.set_printoptions(threshold=numpy.nan)
print(r)

