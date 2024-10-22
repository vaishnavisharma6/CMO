import numpy as np
import matplotlib.pyplot as plt
from oracles_updated import f1, f2, f3
sr_no = 24445
#-----------------------------------------------------------------------------------------------------------------------------
# plot function for Gradient descent
def plotfunction(y, a):
    N = np.arange(0, len(y))
    plt.figure(figsize = (10, 6))
    plt.plot(N, y, label = 'function value using gradient descent')
    plt.title('Gradient Descent with alpha{}'.format(a))
    plt.xlabel('Iteration #')
    plt.ylabel('Function value')
    plt.legend()
    plt.savefig('GD_{}.png'.format(a))
#------------------------------------------------------------------------------------------------------------------------------

def Newtonplot(x0, y):
    N = np.arange(0, len(y))
    plt.figure(figsize = (10, 6))
    plt.plot(N, y, label = 'function value using Newton method')
    plt.title('Newton method with initial x0{}:'.format(x0))
    plt.xlabel('Iteration #')
    plt.ylabel('Function value')
    plt.legend()
    plt.savefig('Newton_2c.png')

#------------------------------------------------------------------------------------------------------------------------------

def conjugate_gradient(A, b, iter):

    x0 = np.random.rand(np.shape(A)[1], 1)
    xk = x0

    if np.shape(A)[1] == np.shape(A)[0]:
        residual = np.dot(A, xk) - b
    else:
        residual = np.dot(np.dot(np.transpose(A), A),xk) - np.dot(np.transpose(A),b)

    delta = -1 * residual
    epsilon = 10**-5      #to avoid division by 0
    
    for i in range(0, iter):
        if np.shape(A)[1] == np.shape(A)[0]: #for symmetric matrix
            beta = -1 * (np.dot(np.transpose(residual),delta))//(np.dot(np.dot(np.transpose(delta),A),delta) + epsilon)
        else:                                # for rectangualr matrix
            beta = -1 * (np.dot(np.transpose(residual),delta))//(np.dot(np.dot(np.transpose(delta),np.dot(np.transpose(A), A)),delta) + epsilon)

        x = xk + beta* delta
        xk = x
        if np.shape(A)[1] == np.shape(A)[0]:
            residual = np.dot(A, xk) - b
            c = (np.dot(np.dot(np.transpose(residual),A),delta))//(np.dot(np.dot(np.transpose(delta),A),delta) + epsilon)
        else:
            residual = np.dot(np.dot(np.transpose(A), A),xk) - np.dot(np.transpose(A),b)
            c = (np.dot(np.dot(np.transpose(residual),np.dot(np.transpose(A), A)),delta))//(np.dot(np.dot(np.transpose(delta),np.dot(np.transpose(A),A)),delta) + epsilon)
       
        delta = c*delta - residual
    return(xk)


#-----------------------------------------------------------------------------------------------------------------------------------
   # frozen oracle -- fix it

def gradient_descent(a, x0, oracle, iter):
    x = []
    f = []
    x.append(x0)
    if oracle == 'f2':
        f.append(f2(np.transpose(x0), sr_no, 0))  
          # for question 2
    if oracle == 'f3':
        f.append(f3(np.transpose(x0), sr_no, 0))    # for question 3
    for i in range(0, iter):
        if oracle == 'f3':
            xi = x[-1] - (a  * f3(np.transpose(x[-1]), sr_no, 1))
            x.append(xi)
            f.append(f3(np.transpose(xi), sr_no, 0))
        else:
            xi = x[-1] - (a * f2(np.transpose(x[-1]), sr_no, 1))   
            x.append(xi) 
            f.append(f2(np.transpose(xi), sr_no, 0))
       
    plotfunction(f, a)   #plot function values for Gradient descent with alpha = a

    return(x, f)

#-------------------------------------------------------------------------------------------------------------------------------------

def Newton(x0, oracle, iter):
    xn = []
    fn = []
    xn.append(x0)

    if oracle == 'f2':
        fn.append(f2(np.transpose(x0), sr_no, 0))
    else:
        fn.append(f3(np.transpose(x0), sr_no, 0))    

    for i in range(0, iter):
        if oracle == 'f2':
           xi = xn[-1] - (f2(np.transpose(xn[-1]), sr_no, 2))
           xn.append(xi)
           fn.append(f2(np.transpose(xn[-1]), sr_no, 0))

        else:
           xi = xn[-1] - (f3(np.transpose(xn[-1]), sr_no, 2))  
           xn.append(xi)
           fn.append(f3(np.transpose(xn[-1]), sr_no, 0))

     
    
    #plot for comparison of newton and gradient descent method
    alpha = 10**-2
    xd, fd = gradient_descent(alpha, x0, 'f2', iter)
    N = np.arange(0, len(fd))
    plt.figure(figsize = (10,6))
    plt.plot(N, fd, label = 'function value using gradient descent')
    plt.plot(N, fn, label = 'function value using Newton method')
    plt.title('comparison with alpha{}'.format(alpha))
    plt.xlabel('Iteration #')
    plt.ylabel('Function value')
    plt.legend()
    plt.savefig('newton_gd_comp_2b.png')

    Newtonplot(x0, fn)  

    return(xn, fn)
#---------------------------------------------------------------------------------------------------------------------------------------

def mixed(x0, oracle, alpha, K):
    x = []
    f = []
    x.append(x0)

    if oracle == 'f2':
        f.append(f2(np.transpose(x0), sr_no, 0))
    else:
        f.append(f3(np.transpose(x0), sr_no, 0))

    for i in range(0, K):
        if oracle == 'f2':
            xi = x[-1] - (alpha * (f2(np.transpose(x[-1]), sr_no, 1)))
            x.append(xi)
            f.append(f2(np.transpose(x[-1]), sr_no, 0))

        else:
            xi = x[-1] - (alpha * (f3(np.transpose(x[-1]), sr_no, 1)))
            x.append(xi)
            f.append(f3(np.transpose(x[-1]), sr_no, 0))

    for i in range(K, 100):
        if oracle == 'f2':
            xi = x[-1] - (f2(np.transpose(x[-1]), sr_no, 2))
            x.append(xi)
            f.append(f2(np.transpose(x[-1]), sr_no, 0))
        else:
            xi = x[-1] - (f3(np.transpose(x[-1]), sr_no, 2))
            x.append(xi)
            f.append(f3(np.transpose(x[-1]), sr_no, 0))

    # plot values
    N = np.arange(0, len(f))
    plt.figure(figsize = (10, 6))
    plt.plot(N[0:K], f[0:K], 'o', label =  'GD, alpha = {}'.format(alpha))
    plt.plot(N[K:100], f[K:100], 'x')
    plt.xlabel('Iteration #')
    plt.ylabel('Function value')
    plt.title('Function values using GD(blue) and Newton method(orange)')
    plt.legend()
    plt.savefig('mixed_{}_{}.png'.format(K, alpha))

    return(x, f)
#---------------------------------------------------------------------------------------------------------------------------------------------    



def quasi_newton(x0, iter):
    xq = []
    xq1 = []
    fq = []
    fq1 = []
    c = 10
    xq.append(x0)
    xq1.append(x0)
    fq.append(f2(np.transpose(x0), sr_no, 0))
    fq1.append(f2(np.transpose(x0), sr_no, 0))
    
    for i in range(0, iter):
        update = c * np.dot(np.identity(5), f2(np.transpose(xq[-1]), sr_no, 1))  #find update
        update_rank1 = 0 # rank1 update
        xi = xq[-1] + (-1) * update
        xiq = xq1[-1] + (-1)*update_rank1
        xq.append(xi)
        xq1.append(xiq)
        fq.append(f2(np.transpose(xq[-1]), sr_no, 0))
        fq1.append(f2(np.transpose(xq1[-1]), sr_no, 0))
    alpha = 10**-2
    xd, fd = gradient_descent(alpha, x0, 'f2', iter)

    # plot for comparison of quasi-newton and gradient descent with different alpha values
    N = np.arange(0, len(fd))
    plt.figure(figsize = (10,6))
    plt.plot(N, fd, label = 'Gradient descent with alpha{}'.format(alpha))
    plt.plot(N, fq, label = 'Quasi-Newton method')
    plt.plot(N, fq1, label = 'Quasi-Newton rank 1 update')
    plt.title('comparison with alpha{}'.format(alpha))
    plt.xlabel('Iteration #')
    plt.ylabel('Function value')
    plt.legend()
    plt.savefig('qcomp{}.png'.format(alpha)) 

    return(xq[-1], xq1[-1], xd[-1], alpha)



#------------------------------------------------------------------------------------------------------------------------------------------
# function calls

# Question 1

iter = 3
A1,b1 = f1(sr_no, True)
xopt = conjugate_gradient(A1,b1, iter)
print('x* using CGD in question 1(b):', xopt)
A2,b2 = f1(sr_no, False)
opt = conjugate_gradient(A2,b2, iter) 
print('x* using CGD in question 1(d):', opt)

#------------------------------------------------------------------------------------------------------------------------------------------
# Question 2 

x0 = np.zeros(5)
iter = 100
a = 0.01

# # 2(a)    #Change a(alpha) values to different values to see different results
x, f = gradient_descent(a, x0, 'f2', iter)
print('Final x using GD with alpha = 0.01:', x[-1])

# #2(b)
xn, fn = Newton(x0, 'f2', iter)  
print('x* using Newton in question 2:', xn[-1])

# #2(c)
# #Newton with 5 random initial x0
for i in range(0, 5):
    x0 = np.random.rand(5)
    xn, fn = Newton(x0, 'f2', iter)
    print('Function optimal value with initial x0{}:'.format(x0),fn[-1])

#-------------------------------------------------------------------------------------------------------------------------------------------
#Question 3 (x0 = [1, 1, 1, 1, 1]^T, alpha = 0.01) for all parts
# 3(a) GD with alpha = 0.1

a = 0.01
iter = 100
x0 = np.ones((5))
x, f = gradient_descent(a, x0, 'f3', iter)
print('Best function value obtained using Gradient Descent with alpha{} in question 3(a):'.format(a), np.min(f))

#---------------------------------------------------------------------------------------------------
# 3(c)

xn, fn = Newton(x0, 'f3', iter)
print('Function values for first 10 iterations:', fn[0:10])

#---------------------------------------------------------------------------------------------------
# 3(d) Newton and Gradient descent both
K = 40
x, f = mixed(x0, 'f3', a, K)
print('Best function value in question 3d with K = {}, alpha = {}:'.format(K, a), np.min(f))

#--------------------------------------------------------------------------------------------------------------------------------------------
# Question 4
x0 = np.zeros(5)
iter = 100
xq, xq1, xd, alpha = quasi_newton(x0, iter)
print('x* using Quasi-Newton:', xq)
print('x* using Rank1 update:', xq1)
print('x* using Gradient descent with alpha = {}:'.format(alpha), xd)
#------------------------------------------------------------------------



