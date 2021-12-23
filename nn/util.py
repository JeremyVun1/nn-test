import matplotlib.pyplot as plt

def avg(m, new, n):
    d = (new - m) / n
    m += d

    return m

def plot_graph(*data):
    plt.clf()
    #plt.rcParams["figure.figsize"] = (4, 2)
    plt.rcParams['toolbar'] = 'None'
    for d in data:
        plt.plot(d)
    plt.pause(0.001)