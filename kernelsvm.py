##  Name : VISHRUT KRASHAK
##  ID   : 12085766 






##--------------------------------------------------------------------------
##
##  kernelsvm.py
##
##  Routines to generate a hard-margin kernel-based SVM.  The solution to
##  the dual form Lagrangian optimization problem:
##
##      Maximize:
##
##        W(L) = \sum_i L_i - 1/2 sum_i sum_j L_i L_j t_i t_j K(x_i, x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and L_i >= 0.
##
##  (where L is a vector of n Lagrange multipliers and K is a kernel
##  function taking two vectors as arguments and returning a scalar) is
##  found using the quadratic program solver "qp" from the convex
##  optimization package "cvxopt" (which has to be installed).
##  
##  Example run, generate weights and bias for a 2-input AND binary
##  classifier.
##
##  >>> Xs = makeBinarySequence(2)   ## Generate list [[0,0],[0,1]...]
##  >>> Ts = [-1,-1,-1,+1]           ## Desired response for a binary AND.
##  >>> stat,Ls = makeLambdas(xs,ts) ## Solve W(Ls) for Lagrange mults.
##                                   ## N.B., status == 'optimal' if a
##                                   ## solution has been found.
##  >>> b = makeB(Xs,Ts,Ls)          ## Find bias.
##  >>> classify([0,1],Xs,Ts,Ls,b)   ## Test classification.
##  >>> testClassifier(Xs,Ts,Ls,b)   ## Exhaustive test on all training
##                                   ## patterns.  See documentation below.
##  >>> plotContours(Xs,Ts,Ls,b)     ## Plot the decision boundary and the
##                                   ## +ve/-ve margins for a 2-d 
##                                   ## classification problem.
##
##  N.B., "makeLambdas", "makeB" and "classify" all have an optional
##  parameter, the kernel function K.  This defaults to a polynomial
##  kernel (x.y + 1.0)^2.  Any  function that accepts two vectors as 
##  arguments and returns a scalar can be used here.  If you need to
##  specify an alternative kernel, use keyword K (see 3-input XOR
##  example below, which uses a cubic kernel).  Definitions for 
##  polynomial, rbf and linear kernels are provided as examples.
##  
##
##--------------------------------------------------------------------------
##
##  Routines
##
##      makeLambdas  -- Generate the n Lagrange multipliers that
##                      represent the maximum-point of the dual
##                      optimization problem W(L).
##                      N.B., maximizing W(L) is a quadratic convex
##                      optimization problem, so the "qp" solver from
##                      "cvxopt" actaully does the work.  Most of what
##                      this routine does is simply setting up the
##                      arguments for the call to "qp".
##
##              Arguments:
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##
##              Returns:
##
##                 A 2-tuple (status,Ls).
##                 1) status:  The first element of the tuple is the 
##                    return status of the "qp" solver.  This will be the 
##                    string "optimal" if a solution has been found.  If 
##                    no solution can be found (say if an XOR problem is
##                    presented to the solver), status typically comes back
##                    as "unknown".
##                 2) Ls:  The second element of the tuple is a list of  
##                    the n Lagrange multipliers.  Element 0 of this list
##                    is the first multiplier and corresponds to Xs[0] and
##                    Ts[0], element 1 is the multiplier corresponding to 
##                    Xs[1] and Ts[1], etc.  These values are only 
##                    meaningful if the first element of the tuple, status
##                    has returned as "optimal".
##
##      --------------------------------------------------------------------
##
##      makeB  -- Given the set of training vectors, "Xs", the set of
##                training responses "Ts", and the set of Lagrange
##                multipliers for the problem "Ls", return the bias for
##                the classifier.
##
##              Arguments:
##
##                 Xs -- Inputs, as "Xs" in makeLambdas.
##
##                 Ts -- A list of desired outputs.  As "Ts" in
##                       makeLambdas.
##
##                 Ls -- A list of Lagrange multipliers, the
##                       solution to the contrained optimaztion
##                       of W(L) as returned by a call to
##                       makeLambdas.  N.B., if this argument is
##                       None (the default), this routine will call
##                       generateLambdas automatically.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##
##              Returns:
##
##                 A double, the bias, "b".
##
##      --------------------------------------------------------------------
##
##      classify -- Classify an input vector using the Lagrange
##                  multipliers for the problem, the set of training
##                  inputs, "Xs", the set of desired outputs, "Ts" and 
##                  bias "b", classify a vector "x".
##
##              Arguments:
##
##                 x -- An input vector to classify (a list of values).
##
##                 Xs -- A list of the training input vectors (hence a list
##                       of n-element lists).
##
##                 Ts -- A list of desired outputs.  As "Ts" in
##                       makeLambdas.
##
##                 Ls -- A list of Lagrange multipliers.
##
##                 b  -- The classifier bias, as generated by makeB.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##                 verbose -- Controls whether or not the routine
##                       prints details about the current classification to
##                       the terminal as well as returning a status
##                       value.  Defaults to True.
##
##              Returns:
##
##                 A classification, +1, -1 or 0 (which indicates an
##                 error in the classification, and shouldn't happen).
##
##      --------------------------------------------------------------------
##
##      testClassifier(Xs,Ts,Ls,b,K,verbose) --
##                  Test a classifier by checking to see if its response
##                  to every training input Xs[i] is the desired output
##                  Ts[i].
##
##              Arguments:
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##                 Ls -- A list of Lagrange multipliers.
##
##                 b  -- The classifier bias, as generated by makeB.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##                 verbose -- Controls whether or not the routine
##                       prints details of misclassifications to the
##                       terminal as well as returning a status
##                       value.  Defaults to True.
##
##
##              Returns:
##
##                 True/False
##
##      --------------------------------------------------------------------
##
##      polyKernel(x,y,p) --
##                  Polynomial kernel (x.y + 1)^p.
##
##              Arguments:
##
##                 x,y  -- n-element vectors.
##
##                 p    -- Exponent of the poly.
##
##              Returns:
##
##                 Scalar value of kernel for the 2 input vectors.
##
##      --------------------------------------------------------------------
##
##      rbfKernel(x,y,s2) --
##                  Radial basis function kernel exp(-||x-y||^2/2*sigma^2).
##
##              Arguments:
##
##                 x,y  -- n-element vectors.
##
##                 s2   -- Variance of the R.B.F. kernel.  Squared standard
##                         deviation, sigma^2.
##
##              Returns:
##
##                 Scalar value of kernel for the 2 input vectors.
##
##      --------------------------------------------------------------------
##
##      linKernel(x,y) --
##                  Linear (standard dot-product) kernel.  Supplied to
##                  allow the kernel-based SVM to work like the non-kernel
##                  based linsvm.py on linear problems like AND.
##
##              Arguments:
##
##                 x,y  -- n-element vectors.
##
##              Returns:
##
##                 Scalar value of kernel for the 2 input vectors.
##
##  ------------------------------------------------------------------------
##
##  Support routines
##
##
##      makeBinarySequence -- Generate a list of the 2^d d-element "vectors"
##                            comprising the complete binary sequence in
##                            d bits.
##
##              Argument:
##
##                 d -- Length of each vector in the output list.  Defaults
##                      to 2 (which will generate the 4 "vector" list
##                      [[0,0], [0,1], [1,0], [1,1]].
##
##              Returns:
##
##                 A list of 2^d elements, each of which is one vector in
##                 the binary sequence.  Example, a call with d=3 will
##                 return the 8-element list [[0,0,0], [0,0,1], [0,1,0],
##                 [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]].
##
##      
##      --------------------------------------------------------------------
##
##      makeP -- Generates the P matrix for a kernel SVM problem.  See the
##               comments associated with generateLambdas below for a
##               discussion of the form and role of the P matrix.
##
##
##--------------------------------------------------------------------------
##



from cvxopt import matrix,solvers
from math import exp




import matplotlib.pyplot as plt
import numpy as np  
from numpy import arange, meshgrid, zeros


## The code below imports dataset
## It then creates Xs as a 2-D array an stores the values of the numbers on the forst 2 columns
## An array Ts is created to store all the values of the 3rd column of the file dataset

dataset = np.loadtxt("dataset.txt")                     
Xs = dataset[0:100,0:2]  
Ts = dataset[0:100,2]


##--------------------------------------------------------------------------
##
##  polyKernel
##
##  Return the polynomial kernel (x.y + 1)^p.  Note default exponent == 2.
##

def polyKernel(v1,v2,p=2):
    assert len(v1) == len(v2)
    return (sum(map(lambda x,y: x*y, v1, v2)) + 1.0)**p


##--------------------------------------------------------------------------
##
##  rbfKernel
##
##  Return the radial basis function kernel exp(-||x-y||^2/2*sigma^2).  Note
##  default variance  (sigma2) = 1.0 (= sigma^2, where sigma =
##  standard deviation).
##
##  Typical problems not too sensitive to size of variance, but note, it
##  must be +ve.
##


## ------------------------------------------------------- CHANGES MADE -----------------------------------------------------

## Value of sigma changed to 0.2

##---------------------------------------------------------------------------------------------------------------------------

def rbfKernel(v1,v2,sigma2=0.2):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x,y: (x-y)*(x-y), v1,v2))  ## Squared mag of diff.
    return exp(-mag2/(2.0*sigma2))


##--------------------------------------------------------------------------
##
##  linKernel
##
##  Return the linear (i.e., simple dot-product) kernel x.y.
##

def linKernel(v1,v2):
    assert len(v1) == len(v2)
    return sum(map(lambda x,y: x*y, v1,v2))


##--------------------------------------------------------------------------
##
##  makeLambdas
##
##  Use the qp solver from the cvx package to find a list of Lagrange
##  multipliers (lambdas or L's) for an Xs/Ts problem, where Xs is a list
##  of input vectors (themselves represented as simple lists) and Ts a list
##  of desired outputs.
##
##
##  Note that we are trying to solve the problem:
##
##      Maximize:
##
##        W(L) =  \sum_i L_i
##                - 1/2 sum_i sum_j L_i * L_j * t_i * t_j * K(x_i,x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and   L_i >= 0.
##
##
##  but the "standard" quadratic programming problem is subtly different,
##  it attempts to *minimize* the following quadratic form:
##
##        f(y) = 1/2 y^t P y  +  q^t y
##
##  subject to:  G L <= h   and   A y = b, where P is an n x n 
##  symmetric matrix, G is an n x n matrix, A is a 1 x n
##  (row) vector, q is an n x 1 (column) vector, as are h and y.
##  N.B., Vector y is the solution being searched for.
##
##  To turn the W(l) constrained maximazation into a constrained
##  minimization of f, it suffices to set:
##
##          [-1.0]
##             .
##             .
##      q = [-1.0]      (n element column vector).
##          [-1.0]
##             .
##             .
##          [-1.0]
##
##
##          [-1.0,  0.0  ....  0.0]
##          [ 0.0, -1.0           ]
##          [             .       ]   (n x n matrix with -1.0 on
##      G = [             .       ]    main diagonal and 0.0's
##          [            0.0,     ]    everywhere else. i.e -I) 
##          [           -1.0,  0.0]
##          [            0.0, -1.0]
##
##
##
##      A = [ t_1, t_2, t_3, ... t_n],  a row vector with n elements
##                                      made using the t list input.
##
##
##      h = n element column vector of 0.0's.
##
##      b = [0.0], i.e., a 1 x 1 matrix containing 0.0.
##
##
##          [                    ]   (n x n matrix with elements 
##      P = [ t_i t_j K(x_i x_j) ]    t_i t_j x_i x_j).
##          [                    ]
##
##
##  The solution (if one exists) is returned by the "qp" solver as
##  a vector of elements.  The solver actually returns a dictionary
##  of values, this contains lots of information about the solution,
##  quality, etc.  But, from the point of view of this routine the
##  important part is the vector of "l" values, which is accessed
##  under the 'x' key in the returned dictionary.
##
##  N.B.  All the routines in the Cxvopt library are very sensitive to
##  the data types of their arguments.  In particular, all vectors,
##  matrices, etc., passed to "qp" must have elements that are
##  doubles.


## --------------------------------------------------------------------- CHANGES MADE -----------------------------------------------------------------
## h matrix modified to be of size 2*n rows and 1 column
## G matrix modified to be of size 2*n rows and n columns
## Passed on C as a parameter to makeLambdas

##-----------------------------------------------------------------------------------------------------------------------------------------------------


def makeLambdas(Xs,Ts,C,K=rbfKernel):
    "Solve constrained maximaization problem and return list of l's."
    P = makeP(Xs,Ts,K)            ## Build the P matrix.
    n = len(Ts)
    q = matrix(-1.0,(n,1))        ## This builds an n-element column 
                                  ## vector of -1.0's (note the double-
                                  ## precision constant).
    
                                  ## n-element column vector of zeros.
    h = matrix(0.0,(2*n,1))         
    h[n::] = C

                                ## These lines generate G, an 
                                 ## n x n matrix with -1.0's on its
                                  ## main diagonal.
                                 ##Generating an n x n matrix
                                 ##with +1.0's on its main diagonal

    G = matrix(0.0,(2*n,n))
    G[::(2*n + 1)] = -1.0
    G[(n)::(2*n+1)] = 1.0  
    
    A = matrix(Ts,(1,n),tc='d')   ## A is an n-element row vector of 
                                  ## training outputs.
    
    ##
    ## Now call "qp". Details of the parameters to the call can be
    ## found in the online cvxopt documentation.
    ##
    r = solvers.qp(P,q,G,h,A,matrix(0.0))  ## "qp" returns a dict, r.
    ##
    ## print r                     ## Dump entire result dictionary
    ##                             ## to terminal.
    ##
    ## Return results. Return a tuple, (Status,Ls).  First element is
    ## a string, which will be "optimal" if a solution has been found.
    ## The second element is a list of Lagrange multipliers for the problem,
    ## rounded to six decimal digits to remove algorithm noise.
    ##
    Ls = [round(l,6) for l in list(r['x'])] ## "L's" are under the 'x' key.
    return (r['status'],Ls)


##--------------------------------------------------------------------------
##
##  Find the bias for this kernel-based classifier.
##
##  The bias can be generated from any support vector Xs[n]:
##
##
##      b = Ts[n] - sum_i Ls[i] * Ts[i] * K(Xs[i],Xs[n])
##
##  because support vectors lie on the margin hyperplanes, hence
##  Ts[n]*y(Xs[n]) == 1 provided that Ls[n] != 0, where y(x) is the
##  discriminant function.
##
##                         Ts[n]*y(Xs[n]) == 1
##                               y(Xs[n]) == Ts[n]   (Ts[n]*Ts[n] == 1)
##                   dot(Ws,Xs)*Xs[n] + b == Ts[n]   (def of y(x))
##  (sum_i Ls[i]*Ts[i]*K(Xs[i],Xs[n])) + b == Ts[n]  (def of Ws -- weights)
##
##
##  It's numerically more stable to average over all support vectors.
##
##  N.B.  If no multipliers are supplied, this routine will call
##  makeLambdas to generate them.  If this fails, it will throw an
##  exception.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##


##  -------------------------------------------------------- CHANGES MADE ----------------------------------------------------------------

##  Default Kernel changed to rbf Kernel
##  Hard margin classifier changed to soft margin classifier by putting a condition of on lamda to be always less than the value of C

## ---------------------------------------------------------------------------------------------------------------------------------------



def makeB(Xs,Ts,C,Ls=None,K=rbfKernel):
   
    "Generate the bias given Xs, Ts and (optionally) Ls and K"
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C,K)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if (Ls[n] >= 1e-10) and (Ls[n] < C):   ## 1e-10 for numerical stability.
            sv_count += 1
            b_sum += Ts[n]
            for i in range(len(Ts)):
                if (Ls[i] >= 1e-10):
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i],Xs[n])                       # i think Ls[i] should be removed as given in lec 11
            
    return b_sum/sv_count


##--------------------------------------------------------------------------
##
##  Classify a single input vector using a trained nonlinear SVM.
##
##  If Lagrange multipliers not supplied, will build them from the
##  training set.  Ditto for bias.  By default uses a quadratic polynomial
##  kernel.  Takes a parameter, verbose, that prints out the input,
##  activation level and classification if set to True.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##


def classify(x,Xs,Ts,Ls=None,b=None,K=rbfKernel,verbose=True):
   
    "Classify an input x into {-1,+1} given support vectors, outputs and L." 
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C,K)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:  b = makeB(Xs,Ts,C,Ls,K)
    ## Do classification.  y is the "activation level".
    y = b
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10 :
            y += Ls[n] * Ts[n] * K(Xs[n],x)

    if verbose:
        print "%s %8.5f  --> " %(x, y),
        if y > 0.0: print "+1"
        elif y < 0.0: print "-1"
        else: print "0  (ERROR)"
    if y > 0.0: return +1
    elif y < 0.0: return -1
    else: return 0 


##--------------------------------------------------------------------------
##
##  Test a trained nonlinear SVM on all vectors from its training set.  The
##  kernel is quadratic polynomial by default.
##
##  If Lagrange multipliers not supplied, will build them from the
##  training set.  Ditto for bias.  By default uses a quadratic polynomial
##  kernel.  Takes a parameter, verbose, that prints out the input,
##  activation level and classification if set to True.
##
##  If no kernel is supplied, this routine will use the default
##  polynomial kernel  K(x,y) = (dot(x,y) + 1.0)^2.
##
##


def testClassifier(Xs,Ts,Ls=None,b=None,K=rbfKernel,verbose=True):
    
    "Test a classifier specifed by Lagrange mults, bias and kernel on all Xs/Ts pairs."
    assert len(Xs) == len(Ts)
    ## No Ls supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C,K)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print "Lagrange multipliers:",Ls
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:
        b = makeB(Xs,Ts,C,Ls,K)
        print "Bias:",b
    ## Do classification test.
    good = True
    for i in range(len(Xs)):
        c = classify(Xs[i],Xs,Ts,Ls,b,K=K)
        if c != Ts[i]:
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %\
                      (Xs[i],c,Ts[i])
            good = False
    return good


##--------------------------------------------------------------------------
##
##  Auxiliary routines.
##
##--------------------------------------------------------------------------
##

## Generate a list of binary vectors, [[0,0],[0,1],[1,0],[1,1]], etc.
## The argument is the length of the vectors (above = 2).
##
def makeBinarySequence(d=2):
    "Return a binary sequence of 2^d vectors, each of length d."
    s = []
    for i in range(2**d):
        v=[]
        for j in range(d-1,-1,-1):
            v.append((i>>j)&1)
        s.append(v)
    return s

## Make the P matrix for a nonlinear, kernel-based SVM problem.


##  Kernel changed to rbf Kernel


def makeP(xs,ts,K=rbfKernel):
    
    """Make the P matrix given the list of training vectors,
       desired outputs and kernel."""
    N = len(xs)
    assert N == len(ts)
    P = matrix(0.0,(N,N),tc='d')
    for i in range(N):
        for j in range(N):
            P[i,j] = ts[i] * ts[j] * K(xs[i],xs[j])
    return P


##--------------------------------------------------------------------------
##
##
##
## Test data
##
##


    
   





def plotContours(Xs,Ts,Ls,b,C,K=rbfKernel,labelPoints=False):

   
    "Plot contours of activation function for a 2-d classifier, e.g. 2-input XOR."
    assert len(Xs) == len(Ts)
    assert len(Xs[0]) == 2   ## Only works with a 2-d classifier.
    ## No Ls supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C,K=rbfkernel)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print "Lagrange multipliers:",Ls
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:
        b = makeB(Xs,Ts,C,Ls,K)
        print "Bias:",b
    ## Build activation level array.
    xs = arange(-1.5,2.61,0.05)                  #imp
    ys = arange(-1.5,2.61,0.05)     
    als = zeros((len(xs),len(ys)))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            testvector=[x,y]
            al = b
            for n in range(len(Ts)):
                if Ls[n] >= 1e-10:
                    al += Ls[n] * Ts[n] * K(Xs[n],testvector)
            als[j,i] = al   ## N.B. Matplotlib array/matrix indexing is row/col
                            ## with rows corresponding to x's and cols to y's.
    ## Plot contour lines at 0.0, +1.0 and -1.0.
    X,Y=meshgrid(xs,ys)
    
    CS = plt.contour(X,Y,als,levels=[-1.0,0.0,1.0],linewidths=(1,2,1),colors=('blue','green','red'))
    ## Plot the training points using red 7 blue circles.
    for i,t in enumerate(Ts):
        if t < 0: col = 'blue'
        else: col = 'red'
        if labelPoints:
            ## print "Plotting %s (%d) as %s"%(Xs[i],t,col)
            plt.text(Xs[i][0]+0.1,Xs[i][1],"%s: %d"%(Xs[i],t),color=col)
        plt.plot([Xs[i][0]],[Xs[i][1]],marker='o',color=col)
    ## Generate labels for contours if flag 'labelContours' is set to
    ## strings 'manual' or 'auto'.  Manual is manual labelling, auto is
    ## automatic labelling (which can mess up if hidden behind data
    ## points).
    
    
    plt.show()




##-------------------------------------------------------------------------------------------------------------------
## Generate plot for 2 different values of C

    
for i in [1,1000000]:
    C = i
    status,Ls = makeLambdas(Xs,Ts,C,K=rbfKernel)
    b = makeB(Xs,Ts,C,Ls,K=rbfKernel)
    plotContours(Xs,Ts,Ls,b,C)

   




