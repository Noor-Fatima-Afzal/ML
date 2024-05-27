class selection_sort:
    def __init__(self, data):
        self.data = data

    def sort(self):
        for i in range(len(self.data)):
            min_index = i
            for j in range(i+1, len(self.data)):
                if self.data[j] < self.data[min_index]:
                    min_index = j
            self.data[i], self.data[min_index] = self.data[min_index], self.data[i]
        return self.data