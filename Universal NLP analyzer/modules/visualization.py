import matplotlib.pyplot as plt
from collections import Counter

def word_freq_plot(tokens):
    freq = Counter(tokens)
    fig = plt.figure()
    plt.bar(freq.keys(), freq.values())
    plt.xticks(rotation=90)
    return fig