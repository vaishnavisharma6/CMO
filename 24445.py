import numpy as np
import cvxpy as cx
import matplotlib.pyplot as plt



#1: Convprob and projected gradient descent

def convprob(A, b):
    mat = np.dot(A, np.transpose(A))
    lb = np.linalg.pinv(mat) * b
    x =  np.transpose(A) * lb
    return(x)

#--------------------------------------------------------------------------------------------------
def projection(A, b, y):
    kinv = np.linalg.pinv(A * np.transpose(A))
    proj = y - np.dot((np.dot(np.transpose(A), kinv)), ((A*y)-b))
    return(proj)

#-------------------------------------------------------------------------------------------------

def projected_gradient(A, b, alpha, iter):
    x = []
    x.append(np.zeros((4,1)))
    for i in range(iter):
        y = x[-1] - alpha* 2 * x[-1]
        x.append(projection(A, b, y))
  
    X = []
    xopt = x[-1]
    for i in range(iter+1):
        X.append(np.linalg.norm((x[i]-xopt), 'fro'))    

    N = np.arange(len(x))
    plt.figure(figsize = (10,6))
    plt.plot(N, X, label = "xt-x*")
    plt.xlabel("# iteration")
    plt.ylabel("xt-x*")
    plt.title('alpha={}'.format(alpha))
    plt.legend()
    plt.savefig('diff_{}.png'.format(alpha))

    return(x[-1])

#-----------------------------------------------------------------------------------------------------
# 2: Support vector machines

X = np.loadtxt("Data.csv", delimiter = ",")
Y = np.loadtxt("Labels.csv", delimiter = ",")

N = np.shape(X)[0]
d = np.shape(X)[1]

w = cx.Variable(d)
b = cx.Variable()

objective = cx.Minimize(0.5* cx.norm(w, 2)**2)
constraints = [Y[i] * (X[i] @ w + b) >= 1 for i in range(N)]

problem = cx.Problem(objective, constraints)
problem.solve()

wopt = -w.value
bopt = b.value
print("w optimal:", wopt)
print("b optimal:", bopt)
print()
print("Optimal Objective value:", problem.value)
print()
#-----------------------------------------------------------------------------------------------------
#2. dual problem solution 
K = np.outer(Y, Y) * (X @ X.T)  
epsilon = 1e-6
reg = epsilon * np.eye(N)

lambda_ = cx.Variable(N)

objective1 = cx.Maximize(cx.sum(lambda_) - 0.5 * cx.quad_form(lambda_, (K+reg)))
constraints1 = [lambda_ >= 0, cx.sum(cx.multiply(lambda_, Y)) == 0]


problem = cx.Problem(objective1, constraints1)
problem.solve()


lambda_optimal = lambda_.value
dual_objective_value = problem.value

print("optimal dual objective:", dual_objective_value)
print("Optimal lambda values:", lambda_optimal)
print()

#-----------------------------------------------------------------------------------------------
#Plots for question 2
#line wTx+b = 0
X_pos = X[Y == 1]
X_neg = X[Y == -1]
x= []
y = []
xt0= np.linspace(-3,3,10)
xt1 = np.linspace(-3,3,10)
for i in range(10):
    x.append(xt0[i])
    yt = wopt[0]*xt0[i]+ wopt[1]*xt1[i] + bopt
    y.append(yt)

#points corresponding to active constraints

ACT = []
for i in range(N): 
    cvalue = np.dot(Y[i], (np.dot(wopt,X[i])+bopt))
    if (cvalue-1 <=1e-6):
        ACT.append(X[i])
 
plt.figure(figsize = (10, 6))
plt.plot(X_pos[:,0], X_pos[:,1], 'o', label = '1')
plt.plot(X_neg[:,0], X_neg[:,1], 's', label ='-1')
plt.plot(x, y, '-', label = 'line')
plt.plot(ACT[:][0], ACT[:][1], 'x', label = 'active constraints')
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.legend()
plt.savefig('labels.png')

#--------------------------------------------------------------------------------------------------------

#required function calls
#1
A = np.matrix([[2, -4, 2, -14], 
              [-1, 2, -2, 11], 
              [-1, 2, -1, 7]])

b = np.matrix([[10], 
              [-6], 
              [-5]])

x = convprob(A, b)
print("x* in 1st question:", x)

print()

alpha = 0.01
iter = 10
x = projected_gradient(A, b, alpha, iter)
print('x* using projected gradient:',x)