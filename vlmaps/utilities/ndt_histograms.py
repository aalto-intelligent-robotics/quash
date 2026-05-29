import numpy as np
import matplotlib as plt
import argparse

# create histograms from ndt maps

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./ndt_historgrams.py")
    parser.add_argument(
        '--item', '-i',
        dest="item",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path', '-p',
        dest="path",
        type=str,
        required=True,
        default=""
    )
    FLAGS, unparsed = parser.parse_known_args()
    item = FLAGS.item
    path = FLAGS.path

data = np.genfromtxt(path+item, delimiter=", ")
data = data[~np.isnan(data)]
print(data.shape)
print(data)

plt.hist(data, bins=100)
plt.title(item)
plt.show()
