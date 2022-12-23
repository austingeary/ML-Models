from collections import Counter
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

class LogisticRegression:
    def __init__(self, iterations=100, learning_rate=0.01):
        self.x = None
        self.y = None
        self.iterations = iterations
        self.learning_rate=learning_rate
        
    def fit(self, x, y):
        self.x = x
        self.y = y
        return None

    def predict(self):
        return None