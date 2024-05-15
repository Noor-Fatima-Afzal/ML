import random
def kmeans(data, k, max_iter):
    random.seed(9001)
    centroids = []
    for _ in range(k):
        centroids.append([random.randint(0, 100), random.randint(0, 100)])

    labels = []
    for i in range(len(data)):
        distances = []
        for j in range(k):
            distances.append(distance(data[i], centroids[j]))
        labels.append(distances.index(min(distances)))

    for _ in range(max_iter):
        new_centroids = []
        for i in range(k):
            new_centroids.append([0, 0])
            for j in range(len(data)):
                if labels[j] == i:
                    new_centroids[i][0] += data[j][0]
                    new_centroids[i][1] += data[j][1]
            new_centroids[i][0] /= len(data[labels == i])
            new_centroids[i][1] /= len(data[labels == i])

        for i in range(k):
            if new_centroids[i] != centroids[i]:
                centroids = new_centroids

    return labels

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


data = [[10, 10], [50, 50], [30, 30], [70, 70], [90, 90]]

labels = kmeans(data, 4, 100)

print(labels)