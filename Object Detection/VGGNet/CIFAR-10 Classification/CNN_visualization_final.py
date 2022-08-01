from sklearn.manifold import TSNE
import numpy as np
import os
from keras.datasets import cifar10


features = np.load(os.path.join(out_dir, 'fc1_features.npy'))
tsne = TSNE().fit_transform(features)

np.save(os.path.join(out_dir, 'fc1_features_tsne_default.npy'), tsne)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

import matplotlib.pyplot as plt
from PIL import Image

_, (x_test, y_test) = cifar10.load_data()

y_test = np.asarray(y_test)

plt.figure(figsize = (16,12))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(len(classes)):
    y_i = y_test == i
    plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label=classes[i], marker='x', linewidth=2)
plt.legend(loc=4)
plt.gca().invert_yaxis()
plt.savefig(os.path.join(out_dir, "feature_space_visualization_num_24.jpg"), bbox_inches='tight')
plt.show()