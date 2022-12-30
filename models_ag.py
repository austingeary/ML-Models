from collections import Counter
import random
import data_funcs_ag as funcs


class KNN:
    def __init__(self):
        self.x = None
        self.y = None
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        return None
        
    def predict(self, x, k, output='classify'):
        assert output in ['classify','regress']
        #Finding distances is O(MN) time complexity where M is the number of data points, 
        #and N is the number of features in each train_point
        distance_label = [
            (funcs.get_distance(x, train_point), train_label)
            for train_point, train_label in zip(self.x, self.y)
        ]
        #Sorting the distances is O(Mlog(M)) time complexity
        neighbors = sorted(distance_label)[:k]
        if output == 'classify':
            neighbor_labels = [label for dist, label in neighbors]
            return Counter(neighbor_labels).most_common()[0][0]
        elif output == 'regress':
            return sum(label for _, label in neighbors) / k

class LinearRegression:
    def __init__(self, iterations=100, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate=learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = 0

        for _ in range(self.iterations):
            gradient_bias, gradient_weights = (
            self.compute_gradients(X, y, n_samples, n_features))
            self.update_params(gradient_bias, gradient_weights)
        return None

    def compute_gradients(self, X, y, n_samples, n_features):
        gradient_bias = 0
        gradient_weights = [0] * n_features
        #Loop through each point in dataset
        for i in range(n_samples):
            #Predict Y value based on X values and current weights
            pred = self.predict_point(X[i])
            #Error
            error = 2 * (y[i] - pred)
            #Loop through all features of this point 
            for j in range(n_features):
                #The gradient in the j dimension is the sum of
                #the difference between logistic prediction and actual class (0, 1)
                #multiplied by the feature value, divided by the number of data points.
                gradient_weights[j] += (error * X[i][j]) / n_samples
            #Similar idea for the constant beta, but without the feature multiplication
            gradient_bias += error / n_samples
        return gradient_bias, gradient_weights

    def predict(self, X):
        preds = []
        for point in X:
            preds.append(self.predict_point(point))
        return preds

    def predict_point(self, point):
        linear_pred = funcs.regress_eval(point, self.weights, self.bias)
        return linear_pred

    def update_params(self, gradient_bias, gradient_weights):
        self.bias += gradient_bias * self.learning_rate
        for i in range(len(gradient_weights)):
            self.weights[i] += (gradient_weights[i] * self.learning_rate)
        return None

class LogisticRegression:
    def __init__(self, iterations=100, learning_rate=0.01, batch_size=None, output='classify'):
        self.iterations = iterations
        self.learning_rate=learning_rate
        self.batch_size = batch_size
        self.output = output
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = 0

        for _ in range(self.iterations):
            gradient_bias, gradient_weights = (
            self.compute_gradients(X, y, n_samples, n_features))
            self.update_params(gradient_bias, gradient_weights)
        return None

    def compute_gradients(self, X, y, n_samples, n_features):
        gradient_bias = 0
        gradient_weights = [0] * n_features
        if self.batch_size:
            #Create a mini-batch
            for _ in range(self.batch_size):
                #Select random datapoint
                i = random.randint(0,n_samples-1)
                point = X[i]
                #Predict the class of this point based on current weights (betas)
                pred = self.predict_point(point)
                #Loop through all features of this point 
                for j, feature in enumerate(point):
                    #The gradient in the j dimension is the sum of
                    #the difference between logistic prediction and actual class (0, 1)
                    #multiplied by the feature value, divided by the number of data points.
                    gradient_weights[j] += (
                        (pred - y[i]) * (feature / self.batch_size))
                #Similar idea for the constant beta, but without the feature multiplication
                gradient_bias += (pred - y[i]) / self.batch_size
        else:
            #Loop through each point in dataset
            for i, point in enumerate(X):
                #Predict the class of this point based on current weights (betas)
                pred = self.predict_point(point)
                #Loop through all features of this point 
                for j, feature in enumerate(point):
                    #The gradient in the j dimension is the sum of
                    #the difference between logistic prediction and actual class (0, 1)
                    #multiplied by the feature value, divided by the number of data points.
                    gradient_weights[j] += (
                        (pred - y[i]) * (feature / n_samples))
                #Similar idea for the constant beta, but without the feature multiplication
                gradient_bias += (pred - y[i]) / n_samples
        return gradient_bias, gradient_weights

    def predict(self, X):
        preds = []
        for point in X:
            preds.append(self.predict_point(point))
        return preds

    def predict_point(self, point):
        linear_pred = funcs.regress_eval(point, self.weights, self.bias)
        logistic_pred = funcs.sigmoid(linear_pred)
        if self.output == 'classify':
            return int(round(logistic_pred,0))
        elif self.output == 'regress':
            return logistic_pred

    def update_params(self, gradient_bias, gradient_weights):
        self.bias -= gradient_bias * self.learning_rate
        for i in range(len(gradient_weights)):
            self.weights[i] -= (gradient_weights[i] * self.learning_rate)
        return None