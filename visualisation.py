import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


def plot_1d(model, fname, title=None):
    plt.clf()
    name = ['First', 'Second', 'Third']
    count = 0
    for mean, var in zip(model.means_, model.covariances_):
        try:
            value = np.random.normal(loc=mean[0], scale=var[0], size=1000)
        except ValueError:
            value = np.random.normal(loc=np.mean(mean), scale=np.mean(var), size=1000)
        sns.distplot(value, kde_kws={'clip': (-500, 500)}, label=name[count] + ' Pattern')
        count += 1
    if title:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig('./results/' + fname + '.png')
    plt.show()
