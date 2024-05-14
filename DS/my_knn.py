from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance=0.0
    for i in range (len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(data, test_row, num_neighbors):
    distance = list()
    for row in data:
        dist = euclidean_distance(test_row, row)
        distance.append((row, dist))
    distance.sort(key=lambda tup: tup[1]) #based on the second element of the tuple and sec elemet is distance of the row from the test_row
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distance[i][0]) # 0 because we want the row not the distance
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(data, test_row, num_neighbors):
    neighbors = get_neighbors(data, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors] # last column of the row is the output value
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Test distance function
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]

pred = predict_classification(dataset, [6.3487,3.3483], 3)
print(pred)
