# %%
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# %%
for im in range(310):
    img = Image.open(f"data_images/img_{im}.png")
    np_img = numpy.array(img)
    normed_matrix = normalize(np_img, axis=1, norm='l1')
    for i in range(normed_matrix.shape[0]):
        for j in range(normed_matrix.shape[1]):
            if normed_matrix[i][j] > 0.033:
                normed_matrix[i][j] = 1
    plt.imsave(f'processed_images/img_{im}.png', normed_matrix, cmap='gray_r')

# %%
