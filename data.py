class TestData:
    def __init__(self):
        self.data = []
        self.labels = []
    
    def add(self, data, label):
        self.data.append(data)
        self.labels.append(label)


def get_data(n = 5000):
    training_n = int(5000 * 0.7)
    test_n = n - training_n

    training_data = TestData()
    for _ in range(training_n):
        training_data.add(n, n*2)
    
    test_data = TestData()    
    for _ in range(test_n):
        n = randint(0, 10000)
        test_data.add(n, n*2)

    #base = [randint(0, 10000) for _ in range(test_n)]

    return training_data, test_data#, base