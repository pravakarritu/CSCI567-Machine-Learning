import numpy as np
import json
import collections
import matplotlib.pyplot as plt

# 5.1
def total_squared_err(x, y, w, n):
	total_err = 0
	for i in range(n):
		total_err += (np.dot(w.T, x[i]) - y[i])**2
	return total_err

# 5.2
def grad_desc(X, y, step, n, d):
	w = np.zeros(d)
	F = np.zeros(20)
	for i in range(20):
		w = w - step*calc_grad(X, y, w, n)
		F[i] = total_squared_err(X, y, w, n)
	return w, F

#5.2
def calc_grad(X, y, w, n):
	grad = 0
	for i in range(n):
		grad += 2*(np.dot(w.T, X[i]) - y[i])*X[i].T
	return grad

#5.3
def stoc_grad_desc(X, y, step, n, d):
	w = np.zeros(d)
	F = np.zeros(1000)
	for i in range(1000):
		rand = np.random.randint(0, d+1)
		w = w - step*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand].T
		F[i] = total_squared_err(X, y, w, n)
	return w, F

def main():
    #==================Problem Set 5.1=======================
	d = 100 # dimensions of data
	n = 1000 # number of data points
	X = np.random.normal(0,1, size=(n,d))
	w_true = np.random.normal(0,1, size=(d,1))
	y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
	w_ls = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
	print("F(w_ls)", total_squared_err(X, y, w_ls, n))

	w = np.zeros((d,1))
	print("F(w_0)", total_squared_err(X, y, w, n))

	X = np.random.normal(0,1, size=(n,d))
	y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
	print("F(w_ls)", total_squared_err(X, y, w_ls, n))

	#==================Problem Set 5.2=======================
	stepsize1 = 0.00005
	w, f1 = grad_desc(X, y, stepsize1, n, d)

	stepsize2 = 0.0005
	w, f2 = grad_desc(X, y, stepsize2, n, d)

	stepsize3 = 0.0007
	w, f3 = grad_desc(X, y, stepsize3, n, d)

	plt.cla()
	plt.xlabel('Num iterations')
	plt.ylabel('F(w) [total squared error]')
	plt.plot(np.array(range(20)), f1, label=stepsize1)
	plt.plot(np.array(range(20)), f2, label=stepsize2)
	plt.plot(np.array(range(20)), f3, label=stepsize3)
	plt.legend()
	plt.title("Gradient descent")
	plt.show()

	#==================Problem Set 5.3=======================
	stepsize1 = 0.0005
	w, f1 = stoc_grad_desc(X, y, stepsize1, n, d)

	stepsize2 = 0.005
	w, f2 = stoc_grad_desc(X, y, stepsize2, n, d)

	stepsize3 = 0.01
	w, f3 = stoc_grad_desc(X, y, stepsize3, n, d)

	plt.cla()
	plt.xlabel('Num iterations')
	plt.ylabel('F(w) [total squared error]')
	plt.plot(np.array(range(1000)), f1, label=stepsize1)
	plt.plot(np.array(range(1000)), f2, label=stepsize2)
	plt.plot(np.array(range(1000)), f3, label=stepsize3)
	plt.legend()
	plt.title("Stochastic Gradient descent")
	plt.show()

if __name__ == "__main__":
	main()