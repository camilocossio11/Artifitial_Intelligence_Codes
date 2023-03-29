#%%
import umap
import pandas as pd
import common_functions as commons
import seaborn
#%%

def umap_transformation(X,dimension):
    if dimension == 2:
        fit = umap.UMAP(metric='euclidean',
                        n_components=dimension,
                        #random_state=0,
                        n_neighbors=15,
                        min_dist=0.4)
        features_low = fit.fit_transform(X)
        features_low = pd.DataFrame(features_low,columns=['Feature_1','Feature_2'])
        features_low = commons.minmax_norm(features_low)
        seaborn.scatterplot(x=features_low.columns.tolist()[0],
                            y=features_low.columns.tolist()[1],
                            data=features_low).set(title='Dataset preview')
        features_low.to_excel('Compressed.xlsx',index=False)
    elif dimension == 3:
        fit = umap.UMAP(metric='euclidean',
                        n_components=dimension,
                        #random_state=0,
                        n_neighbors=15,
                        min_dist=0.4)
        features_low = fit.fit_transform(X)
        features_low = pd.DataFrame(features_low,columns=['Feature_1','Feature_2','Feature_3'])
        features_low = commons.minmax_norm(features_low)
        features_low.to_excel('Compressed_3D.xlsx',index=False)


# %%
X = pd.read_excel('Normalized_data.xlsx')
umap_transformation(X,3)
# %%
