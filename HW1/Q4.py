import numpy as np
import json
import collections
import matplotlib.pyplot as plt

def data_processing(data):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False):
	train_set, valid_set, test_set = data['train'], data['valid'], data['test']
	Xtrain = train_set[0]
	ytrain = train_set[1]
	Xval = valid_set[0]
	yval = valid_set[1]
	Xtest = test_set[0]
	ytest = test_set[1]

	Xtrain = np.array(Xtrain)
	Xval = np.array(Xval)
	Xtest = np.array(Xtest)

	ytrain = np.array(ytrain)
	yval = np.array(yval)
	ytest = np.array(ytest)
	
	# We load data from json here and turn the data into numpy array
	# You can further perform data transformation on Xtrain, Xval, Xtest

	# Min-max
	def min_max(x):
		for i in range(len(x)): 
			if (np.linalg.norm(x[i])!=0):
				x[i] = (x[i] - np.amin(x[i])) / (np.amax(x[i]) - np.amin(x[i]))
		return x

	# Min-Max scaling
	if do_minmax_scaling:
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		Xtrain = min_max(Xtrain)
		Xtest = min_max(Xtest)

	# Normalization
	def normalization(x):
		#####################################################
		#				 YOUR CODE HERE					    #
		#####################################################
		for i in range(len(x)): 
			if (np.linalg.norm(x[i])!=0):
				x[i] = x[i] / (np.linalg.norm(x[i]))
		return x
	
	if do_normalization:
		Xtrain = normalization(Xtrain)
		Xval = normalization(Xval)
		Xtest = normalization(Xtest)

	return Xtrain, ytrain, Xval, yval, Xtest, ytest


def compute_l2_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	dists = np.zeros((len(X), len(Xtrain)))
	for i in range(len(X)):
		for j in range(len(Xtrain)):
			dists[i][j] = np.linalg.norm(X[i]-Xtrain[j])
	return dists


def compute_cosine_distances(Xtrain, X):
	"""
	Compute the distance between each test point in X and each training point
	in Xtrain.
	Inputs:
	- Xtrain: A numpy array of shape (num_train, D) containing training data
	- X: A numpy array of shape (num_test, D) containing test data.
	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Cosine distance between the ith test point and the jth training
	  point.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	dists = np.zeros((len(X), len(Xtrain)))
	for i in range(len(X)):
		for j in range(len(Xtrain)):
			x_norm = np.linalg.norm(X[i])
			train_norm = np.linalg.norm(Xtrain[i])
			if (x_norm==0 or train_norm==0):
				dists[i][j] = 1
			else:
				dists[i][j] = 1 - (np.dot(X[i], Xtrain[i]) / (x_norm*train_norm))
	return dists


def predict_labels(k, ytrain, dists):
	"""
	Given a matrix of distances between test points and training points,
	predict a label for each test point.
	Inputs:
	- k: The number of nearest neighbors used for prediction.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  gives the distance betwen the ith test point and the jth training point.
	Returns:
	- ypred: A numpy array of shape (num_test,) containing predicted labels for the
	  test data, where y[i] is the predicted label for the test point X[i].
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	ypred = np.zeros(len(dists))
	k_labels = np.zeros(k)
	for i in range(len(dists)):
		sorted = np.argsort(dists[i])
		for j in range(k):
			k_labels[j] = ytrain[sorted[j]]
		counter = collections.Counter(k_labels)
		label, top_ct = counter.most_common(1)[0]
		ties = np.zeros(k)
		ties[0] = label
		ct = 1
		for t in counter.most_common():
			if t[0] != label and top_ct == t[1]:
				ties[ct] = t[0]
				ct+=1	
			elif t[1] < top_ct:
				break
		ypred[i] = np.min(ties[:ct])
	return ypred


def compute_error_rate(y, ypred):
	"""
	Compute the error rate of prediction based on the true labels.
	Inputs:
	- y: A numpy array with of shape (num_test,) where y[i] is the true label
	  of the ith test point.
	- ypred: A numpy array with of shape (num_test,) where ypred[i] is the
	  prediction of the ith test point.
	Returns:
	- err: The error rate of prediction (scalar).
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	cnt = 0
	for i in range(len(y)):
		if y[i] != ypred[i]:
			cnt += 1
	return cnt/len(y)


def find_best_k(K, ytrain, dists, yval):
	"""
	Find best k according to validation error rate.
	Inputs:
	- K: A list of ks.
	- ytrain: A numpy array of shape (num_train,) where ytrain[i] is the label
	  of the ith training point.
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the distance between the ith test point and the jth training
	  point.
	- yval: A numpy array with of shape (num_val,) where y[i] is the true label
	  of the ith validation point.
	Returns:
	- best_k: The k with the lowest error rate.
	- validation_error: A list of error rate of different ks in K.
	- best_err: The lowest error rate we get from all ks in K.
	"""
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	validation_error = [compute_error_rate(yval, predict_labels(k, ytrain, dists)) for k in K]
	best_err = min(validation_error)
	best_k = K[np.argmin(validation_error)]
	plt.plot(K, validation_error)
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.title('Error rate on validation set for each k')
	plt.show()
	return best_k, validation_error, best_err

def main():
	input_file = 'disease.json'
	output_file = 'knn_output.txt'

	#==================Problem Set 1.1=======================

	with open(input_file) as json_data:
		data = json.load(json_data)

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.1")
	print()

	#==================Problem Set 1.2=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=False, do_normalization=True)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using normalization")
	print()

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_with_transformation(data, do_minmax_scaling=True, do_normalization=False)

	dists = compute_l2_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.2 when using minmax_scaling")
	print()

	#==================Problem Set 1.3=======================

	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)
	dists = compute_cosine_distances(Xtrain, Xval)

	# Compute validation accuracy when k=4
	k = 4
	ypred = predict_labels(k, ytrain, dists)
	err = compute_error_rate(yval, ypred)
	print("The validation error rate is", err, "in Problem Set 1.3, which use cosine distance")
	print()

	#==================Problem Set 1.4=======================
	# Compute distance matrix
	Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing(data)

	#======performance of different k in training set=====
	K = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
	#####################################################
	#				 YOUR CODE HERE					    #
	#####################################################
	train_err = np.zeros(len(K))
	for i in range(len(K)):
		dists = compute_l2_distances(Xtrain, Xtrain)
		ypred = predict_labels(K[i], ytrain, dists)
		train_err[i] = compute_error_rate(ytrain, ypred)
	plt.plot(K, train_err)
	plt.xlabel('k')
	plt.ylabel('Error rate')
	plt.title('Error rate on training set for each k')
	plt.show()
	
	#==========select the best k by using validation set==============
	dists = compute_l2_distances(Xtrain, Xval)
	best_k, validation_error, best_err = find_best_k(K, ytrain, dists, yval)

	#===============test the performance with your best k=============
	dists = compute_l2_distances(Xtrain, Xtest)
	ypred = predict_labels(best_k, ytrain, dists)
	test_err = compute_error_rate(ytest, ypred)
	print("In Problem Set 1.4, we use the best k = ", best_k, "with the best validation error rate", best_err)
	print("Using the best k, the final test error rate is", test_err)
	#====================write your results to file===================
	f=open(output_file, 'w')
	for i in range(len(K)):
		f.write('%d %.3f' % (K[i], validation_error[i])+'\n')
	f.write('%s %.3f' % ('test', test_err))
	f.close()

if __name__ == "__main__":
	main()
