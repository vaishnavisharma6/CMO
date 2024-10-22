from CMO_A1 import f1,f2,f3,f4
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp


#1st question: To check convexity and coercevity
Sno = 24445

def isConvex(fname, interval):
   x = sp.Symbol('x')
   for i in interval:
      if fname =='f1':
         sd = sp.diff(f1(Sno, i), x, 2)
         if sd < 0:
            return False
      if fname == 'f2':
         sd = sp.diff(f2(Sno, i), x, 2)  
         if sd < 0:
            return False
   return True       

    
#------------------------------------------------------------
def isCoercive(fname):
    points = [np.random.randn(n) * 10**n for n in range(1, 6)]
    for i in points:
      if fname == 'f3':
        fx = f3(Sno,i)
        print(fx)
        if fx.any() < 0:
           return False
    return True    

def FindStationaryPoints(fname):
    a,b = -2, 2
    interval = [a,b]
    dict1 = {}
    for i in interval:
       if f3(Sno, a) * f3(Sno, b) >=0:
          return
       else:
          c = (a+b)//2
          if f3(Sno, c) == 0:
             dict1['root'] = c
          elif (f3(Sno, c)*f3(Sno,a)) < 0:
             b = c
          else:
             a = c
    # find minima, use gradient descent method defined in question 2
   
    return dict1

# Question 1

interval = [-2, 2]
result1 = isConvex('f1', interval)
result2 = isConvex('f2', interval)  
print('Is function f1 convex?:', result1)
print('Is function f2 convex?:',result2)
result3 = isCoercive('f3')
print('Is function f3 coercive?:',result3)
#----------------------------------------------------------------------------

# Question 2

#plot function for all gradient descents
def plot(method, gradl2,fdiff, fdiffratio, xdiffl2, xdiffratio):
    
    k = np.arange(0, len(gradl2))
    plt.figure(figsize=(10,5))
    plt.plot(k, gradl2, label = 'l2 norm of grad fx')
    plt.xlabel('Iterations')
    plt.ylabel('l2 norm of grad fx')
    plt.legend()
    plt.savefig('{}_fig2_1.png'.format(method))

    k = np.arange(0, len(fdiff))
    plt.figure(figsize = (10,5))
    plt.plot(k, fdiff, label = 'function difference')
    plt.xlabel('Iterations')
    plt.ylabel('fx-fT')
    plt.legend()
    plt.savefig('{}_fig2_2.png'.format(method))

    k = np.arange(0, len(fdiffratio))
    plt.figure(figsize = (10,5))
    plt.plot(k, fdiffratio, label = 'Ratio')
    plt.xlabel('Iterations')
    plt.ylabel('fxk-fT/fxk-1 -fT')
    plt.legend()
    plt.savefig('{}_fig2_3.png'.format(method))

    k = np.arange(0, len(xdiffl2))
    plt.figure(figsize = (10,5))
    plt.plot(k, xdiffl2, label = 'x differences')
    plt.xlabel('Iterations')
    plt.ylabel('l2 norm of xk - xT')
    plt.legend()
    plt.savefig('{}_fig2_4.png'.format(method))
   
    
    k = np.arange(0, len(xdiffratio))
    plt.figure(figsize = (10, 5))
    plt.plot(k, xdiffratio, label = 'ratio2')
    plt.xlabel('Iterations')
    plt.ylabel('xk-xT/xk-1 - xT')
    plt.legend()
    plt.savefig('{}_fig2_4.png'.format(method))
   
   
def ConstantGradientDescent(alpha, initialx):
    x = []
    f = []
    gradl2 = []
    fdiff2 = []
    fdiff = []
    xdiffl2 = []
    xdiff2 = []
    T = 10000
    x.append(initialx)
    for i in range(T):
        last = x[-1]
        fx, gradfx = f4(Sno, last)
        f.append(fx)
        gradl2.append(np.linalg.norm(gradfx)**2)
        dir = -1 * gradfx
        x.append(last + (alpha*dir))
    for i in range(0, T):
       fdiff.append(f[i] - f[-1])
       xdiffl2.append(np.linalg.norm(x[i]-x[-1])**2)
    for i in range(1,T):   
       fdiff2.append(f[i] - f[-1])
       xdiff2.append(np.linalg.norm(x[i] - x[-1])**2)
    fdiff2.append(0)
    xdiff2.append(0)
    fdiffratio = np.divide(fdiff2,fdiff)
    xdiffratio = np.divide(xdiff2, xdiffl2)
    
    plot('const',gradl2,fdiff, fdiffratio, xdiffl2, xdiffratio)

    return(x[-1], f[-1])   
#--------------------------------------------------------------


def DiminishingGradientDescent(Initialalpha, initialx):
  x = []
  f = []
  gradl2 = []
  fdiff2 = []
  fdiff = []
  xdiffl2 = []
  xdiff2 = []
  T = 10000
  x.append(initialx)
  for i in range(T):
    last = x[-1]
    fx, gradfx = f4(Sno, last)
    f.append(fx)
    gradl2.append(np.linalg.norm(gradfx)**2)
    dir = -1 * gradfx
    x.append(last + ((Initialalpha/(i+1)) * dir))
  for i in range(0, T):
      fdiff.append(f[i] - f[-1])
      xdiffl2.append(np.linalg.norm(x[i]-x[-1])**2)
  for i in range(1,T):   
      fdiff2.append(f[i] - f[-1])
      xdiff2.append(np.linalg.norm(x[i] - x[-1])**2)
  fdiff2.append(0)
  xdiff2.append(0)
  fdiffratio = np.divide(fdiff2,fdiff)
  xdiffratio = np.divide(xdiff2, xdiffl2)

  plot('dim',gradl2,fdiff, fdiffratio, xdiffl2, xdiffratio)

  return(x[-1], f[-1])





#----------------------------------------------------------------

def ExactLineSearch(initialx):
  x = []
  f = []
  gradl2 = []
  fdiff2 = []
  fdiff = []
  xdiffl2 = []
  xdiff2 = []
  x.append(initialx)
  T = 10000
  for i in range(T):
    last = x[-1]
    fx, gradfx = f4(Sno, last)
    f.append(fx)
    gradl2.append(np.linalg.norm(gradfx)**2)
  
    dir = -1 * gradfx
    A = (1/2) * np.gradient(gradfx, x[-1])
 
    x.append(last + ((-1)* ((np.transpose(gradfx)) * dir)//(np.transpose(dir)*dir)))
  for i in range(0, T):
    fdiff.append(f[i] - f[-1])
    xdiffl2.append(np.linalg.norm(x[i]-x[-1])**2)
  for i in range(1,T):   
      fdiff2.append(f[i] - f[-1])
      xdiff2.append(np.linalg.norm(x[i] - x[-1])**2)
  fdiff2.append(0)
  xdiff2.append(0)
  fdiffratio = np.divide(fdiff2,fdiff)
  xdiffratio = np.divide(xdiff2, xdiffl2)

  plot('exact', gradl2,fdiff, fdiffratio, xdiffl2, xdiffratio)
  
  return(x[-1], f[-1])




#Question 2
initialx = np.transpose(np.zeros(5))
alpha = 10**-5
x, y = ConstantGradientDescent(alpha, initialx)     #Constant alpha
print(x)
print(y)

InitialAlpha = 10**-3
x, y = DiminishingGradientDescent(InitialAlpha, initialx)       #Diminishing alpha
print(x)
print(y)


x,y = ExactLineSearch(initialx)
print(x)
print(y)

#-------------------------------------------------------------------------------
# Question 3
def perturbed(f, alpha, sigma, initialx, interval):
   x = []
   T = 10000
   x.append(initialx)
   for i in range(T):
      last = x[-1]
      fx, gradfx = f4(Sno, last)
      f.append(fx)
      dir = -1 * gradfx
      noise = np.random.normal(0, sigma, size = x[0].shape)
      x.append(last + (alpha*(dir+noise)))
   return
#--------------------------------------------------------------------------------
# Question 4
def plot(method, fat, fbt, diff):
    k = np.arange(0, len(fat))
    plt.figure(figsize=(10,5))
    plt.plot(k, fat, label = 'f(at)')
    plt.xlabel('Iterations')
    plt.ylabel('f(at)')
    plt.legend()
    plt.savefig('{}_fig4_1.png'.format(method))

    k = np.arange(0, len(fbt))
    plt.figure(figsize = (10,5))
    plt.plot(k, fbt, label = 'f(bt)')
    plt.xlabel('Iterations')
    plt.ylabel('f(bt)')
    plt.legend()
    plt.savefig('{}_fig4_2.png'.format(method))

    k = np.arange(0, len(diff))
    plt.figure(figsize = (10,5))
    plt.plot(k, diff, label = 'Ratio')
    plt.xlabel('Iterations')
    plt.ylabel('bt - at')
    plt.legend()
    plt.savefig('{}_fig4_3.png'.format(method))



#--------------------------------------------------------------------------
def f5(x):
   return(x*(x-1)*(x-3)*(x+2))

def golden(x):
  p = 1 - ((1+math.sqrt(5))//2)
  T = 18
  fat = [f5(x[0])]
  fbt = [f5(x[-1])]
  diff = []
  xl = []
  for i in range(T):
    x1 = p*x[0] + ((1-p)*x[-1])
    x2 = ((1-p)*x[0]) + p*x[-1]
    I = [x[0], x1, x2, x[-1]]


    if f5(I[1]) <= f5(I[2]):
      x3 = p*x1 + (1-p)*x2
      I = [x[0], x3, x1, x2]
      fat.append(f5(I[0]))
      fbt.append(f5(I[-1]))
      diff.append(I[-1]-I[0])
    else:
      x3 = p*x1 + (1-p)*x[-1]
      I = [x1, x2, x3, x[-1]]
      fat.append(f5(I[0]))
      fbt.append(f5(I[-1]))
      diff.append(I[-1]-I[0])
  plot('golden', fat, fbt, diff) 

  return(I)
  
def fibo(x):
  T = 18
  F = [0,1,1]
  fat = [f5(x[0])]
  fbt = [f5(x[-1])]
  diff = []
  xl = []

  #Define fibonacci sequence
  for i in range(3,T+1):
    F.append(F[i-1] + F[i-2])

  for t in range(3, T):
    p = 1 - (F[T-t+1]//F[T-t+2])
    x1 = p*x[0] + (1-p)*x[-1]
    x2 = (1-p)*x[0] + p*x[-1]
    I = [x[0], x1, x2, x[-1]]
  

    if f5(I[1]) <= f5(I[2]):
      x3 = p*I[1] + (1-p)*I[2]
      I = [x[0], x3, x1, x2]
      fat.append(f5(I[1]))
      fbt.append(f5(I[2]))
      diff.append(I[-1]-I[0])

    else:
      x3 = p*I[1] + (1-p)*I[-1]
      I = [x1, x2, x3, x[-1]]
      fat.append(f5(I[1]))
      fbt.append(f5(I[2]))
      diff.append(I[-1]-I[0])
  plot('fibo', fat, fbt, diff) 
  return(I) 
   
#function calls
interval = [1, 3]
I1 = golden(interval)
I2 = fibo(interval)

print(I1)
print(I2) 