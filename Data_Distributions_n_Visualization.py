import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions():
    distributions = {
        "Normal": np.random.normal(loc=0, scale=1, size=1000),
        "Uniform": np.random.uniform(low=-1, high=1, size=1000),
        "Exponential": np.random.exponential(scale=1, size=1000),
        "Poisson": np.random.poisson(lam=3, size=1000),
        "Binomial": np.random.binomial(n=10, p=0.5, size=1000)
    }
    
    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(distributions.items()):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data, bins=30, kde=True)
        plt.title(name)
    plt.tight_layout()
    plt.show()

plot_distributions()
