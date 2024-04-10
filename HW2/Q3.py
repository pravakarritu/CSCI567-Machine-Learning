import numpy as np
import matplotlib.pyplot as plt

####### 3.2 #######
def l2error(w, x, y, lamb, d):
    total_err = 0
    for i in range(d):
        total_err += (np.dot(w.T, x[i])-y[i])**2 + lamb*np.linalg.norm(w)**2
    return total_err

def norm_err(x, y, w):
    return np.linalg.norm(x.dot(w)-y) / np.linalg.norm(y)

def stoc_grad_desc(X, y, step, d):
    w = np.zeros(d)
    for i in range(1000000):
        rand = np.random.randint(0, d)
        w = w - step*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand]
    print(str(step)+" done")
    return norm_err(X, y, w)

def stoc_grad_desc_graph(X, y, step, d):
    w = np.zeros(d)
    f = np.zeros(1000000)
    for i in range(1000000):
        rand = np.random.randint(0, d)
        w = w - step*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand].T
        f[i] = norm_err(X, y, w)
    return f

def stoc_grad_desc_norm(X, y, step, d):
    w = np.zeros(d)
    f = np.zeros(1000000)
    for i in range(1000000):
        rand = np.random.randint(0, d)
        w = w - step*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand].T
        f[i] = np.linalg.norm(w)
    return f

def stoc_grad_desc_graph_test(X, y, step, d):
    c = 0
    w = np.zeros(d)
    f = np.zeros(10000)
    for i in range(1000000):
        rand = np.random.randint(0, d)
        w = w - step*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand].T
        if i%100==0:
            f[c] = norm_err(X, y, w)
            c+=1
    return f

def stoc_grad_desc_rad(X, y, d, r):
    w = np.random.normal(0, 1, size=(d, 1))
    w = r * (w / np.linalg.norm(w))
    for i in range(1000000):
        rand = np.random.randint(0, d)
        w = w - 0.00005*2*(np.dot(w.T, X[rand]) - y[rand])*X[rand].T
        if i%100000==0:
            print(i)
    return norm_err(X, y, w)

def main():
    # Given code to generate training and test data
    train_n = 100
    test_n = 1000
    d = 100

    print("3.1")
    ####### 3.1 #######
    train_error = 0
    test_error = 0
    for i in range(10):
        X_train = np.random.normal(0,1, size=(train_n,d))
        w_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
        w_train = np.linalg.inv(X_train).dot(y_train)
        train_error += np.linalg.norm(X_train.dot(w_train)-y_train) / np.linalg.norm(y_train)

        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
        test_error += np.linalg.norm(X_test.dot(w_train)-y_test) / np.linalg.norm(y_test)

    train_error /= 10
    test_error /= 10
    print("Average training error over 10 trials", train_error)
    print("Average testing error over 10 trials", test_error)

    print("3.2")
    ####### 3.2 #######
    lamb_vals = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    train_err = [0,0,0,0,0,0,0]
    test_err = [0,0,0,0,0,0,0]
    for i in range(10):
        X_train = np.random.normal(0,1, size=(train_n,d))
        w_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
        for j in range(len(lamb_vals)):
            w_train = np.linalg.inv((X_train.T).dot(X_train)+lamb_vals[j]*np.identity(train_n)).dot(X_train.T).dot(y_train)
            train_err[j] += norm_err(X_train, y_train, w_train)
            test_err[j] += norm_err(X_test, y_test, w_train)
    for i in range(len(train_err)):
        train_err[i] /= 10
        test_err[i] /= 10
    plt.plot(lamb_vals, train_err, label="Training")
    plt.plot(lamb_vals, test_err, label="Test")
    plt.xlabel('Lambda')
    plt.ylabel('Error rate')
    plt.title('Normalized l2 error')
    plt.legend()
    plt.show()

    print("3.3")
    ####### 3.3 #######
    step = [0.00005, 0.0005, 0.005]
    train_err = [0,0,0]
    test_err = [0,0,0]
    for i in range(10):
        X_train = np.random.normal(0,1, size=(train_n,d))
        w_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
        for j in range(len(step)):
            train_err[j] += stoc_grad_desc(X_train, y_train, step[j], d)
            test_err[j] += stoc_grad_desc(X_test, y_test, step[j], d)
    for i in range(len(train_err)):
        train_err[i] /= 10
        test_err[i] /= 10
    print("Train err", train_err)
    print("Test err", test_err)

    print("3.4")
    ####### 3.4 #######
    X_train = np.random.normal(0,1, size=(train_n,d))
    w_true = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
    err = stoc_grad_desc_graph(X_train, y_train, 0.00005, d)
    plt.plot(range(1000000), err, label='0.00005')
    err = stoc_grad_desc_graph(X_train, y_train, 0.005, d)
    plt.plot(range(1000000), err, label='0.005')
    plt.axhline(y=norm_err(X_train, y_train, w_true))
    plt.xlabel('Iteration number t')
    plt.ylabel('Normalized training error of w at t')
    plt.title('Normalized training error of SGD')
    plt.legend()
    plt.show()
    plt.cla()

    step = [0.00005, 0.005]
    
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
    for s in step:
        err = stoc_grad_desc_graph_test(X_test, y_test, s, d)
        plt.plot(range(0, 1000000, 100), err, label=s)
        print(str(s)+" done")
    plt.xlabel('Iteration number t')
    plt.ylabel('Normalized test error of w at t')
    plt.title('Normalized test error of SGD')
    plt.legend()
    plt.show()
    plt.cla()

    for s in step:
        err = stoc_grad_desc_norm(X_test, y_test, s, d)
        plt.plot(range(1000000), err, label=s)
        print(str(s)+" done")
    plt.xlabel('Iteration number t')
    plt.ylabel('l2 norm of w at t')
    plt.legend()
    plt.title('l2 norm of SGD')
    plt.show()
    plt.cla()

    print("3.5")
    ####### 3.5 #######
    radius = [0, 0.1, 0.5, 1, 10, 20, 30]
    train_err = [0 for i in range(7)]
    test_err = [0 for i in range(7)]
    for i in range(10):
        X_train = np.random.normal(0,1, size=(train_n,d))
        w_true = np.random.normal(0,1, size=(d,1))
        y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
        X_test = np.random.normal(0,1, size=(test_n,d))
        y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
        for j in range(len(radius)):
            train_err[j] += stoc_grad_desc_rad(X_train, y_train, d, radius[j])
            test_err[j] += stoc_grad_desc_rad(X_test, y_test, d, radius[j])
            print(str(radius[j])+ " done")
    for i in range(len(train_err)):
        train_err[i] /= 10
        test_err[i] /= 10
    plt.plot(radius, train_err, label="Training")
    plt.plot(radius, test_err, label="Test")
    plt.xlabel('Radius r')
    plt.ylabel('Average normalized error over 10 trials')
    plt.title('SGD with radius')
    plt.legend()
    plt.show()

if __name__ == "__main__":
	main()
