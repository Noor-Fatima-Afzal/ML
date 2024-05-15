from statistics import mode

class KNN:

    def __init__(self, k):
        #set value of neighbors to consider
        self.k = k

    def fit(self, X, y):
        # simply store the training data, features in x and ground truth label i.e. class is either 1 or 2 . 
        self.X = X
        self.y = y

    def predict(self, x):
        # x is test point whose class label is to be predicted
        # now compute the euclidean distance of x from each and every sample present in the training dataset
        distances = []
        for i in range(len(self.X)):
            distance = self._euclidean_distance(x, self.X[i])
            distances.append((distance, self.y[i])) # append the distance and the class label of the sample
        
        # now sort according to the distances computed and extract the nearest k number of neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        #labels of k nearest neighbors
        print(f"Distances and Class labels of  {self.k}-nearest neighbors are, {neighbors}")

        #now extract the class labels of the k nearest neighbors in list labels
        labels = []
        for neighbor in neighbors:
            labels.append(neighbor[1])

        #now find the most frequently occuring element
        #most_common_label = max(set(labels), key=labels.count)
        most_common_label = mode(labels)
        return most_common_label

    def _euclidean_distance(self, x1, x2):
        sum_of_squares = 0
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            sum_of_squares += diff * diff

        return sum_of_squares ** 0.5


X = [[1, 70], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [1, 1, 1, 0, 0]

knn = KNN(5)
knn.fit(X, y)

x = [4, 7]
print("The predicted label for the test sample is ",knn.predict(x))


