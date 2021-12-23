class TestData:
    def __init__(self):
        self.data = []
        self.labels = []
    
    def add(self, data, label):
        self.data.append(data)
        self.labels.append(label)
