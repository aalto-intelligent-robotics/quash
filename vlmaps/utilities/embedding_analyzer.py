import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# colormap = [
#     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
#     '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
#     '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
#     '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
#     '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
#     '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
#     '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b',
#     '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
#     '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d',
#     '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476'
# ]

# colormap = [
#     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
#     '#393b79', '#6b6ecf', '#bd9e39', '#843c39', '#ad494a',
#     '#e7969c', '#31a354', '#f7b6d2', '#c5b0d5', '#ff9896',
#     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#ffbb78',
#     '#2ca02c', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
#     '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
#     '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
#     '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
#     '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b'
# ]

colormap = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#6b6ecf', '#bd9e39', '#843c39', '#ad494a',
    '#e7969c', '#31a354', '#f7b6d2', '#c5b0d5', '#aec7e8',
    '#ffbb78', '#98df8a', '#ff9896', '#c7c7c7', '#9edae5',
    '#5254a3', '#9c9ede', '#637939', '#8ca252', '#b5cf6b',
    '#8c6d31', '#d6616b', '#c49c94', '#cedb9c', '#dbdb8d',
    '#e7ba52', '#e7cb94'
]

def analyze(embeddings, ax, name):
    print("*"*80)
    print(name)
    print(embeddings.shape)
    print("raw: min, mean, max")
    print(embeddings.shape, np.min(embeddings), np.mean(embeddings), np.max(embeddings))

    embeddings_sum = np.linalg.norm(embeddings, axis=1)
    embeddings_sum = embeddings_sum[~np.isinf(embeddings_sum)]
    print(embeddings_sum.shape, np.all(np.abs(1 - embeddings_sum) < 0.1), np.min(embeddings_sum), np.mean(embeddings_sum), np.max(embeddings_sum))

    embeddings_mean = np.mean(embeddings, axis=0)
    print("means: min, mean, max")
    print(embeddings_mean.shape, np.min(embeddings_mean), np.mean(embeddings_mean), np.max(embeddings_mean))

    ax.plot(embeddings_mean)

def main():
    ensemble = np.load("/home/user/code/vlmaps/embedding-example/5LpN3gDmAk7_embedding_openseg_ensemble.data.npy")
    ens_lab = np.load("/home/user/code/vlmaps/embedding-example/5LpN3gDmAk7_semantic.data.npy")
    distill = np.load("/home/user/code/vlmaps/embedding-example/5LpN3gDmAk7_embedding_openseg_distill.data.npy")
    fusion = np.load("/home/user/code/vlmaps/embedding-example/5LpN3gDmAk7_embedding_openseg_fusion.data.npy")
    vlmapsdata = np.load("/home/user/code/vlmaps/embedding-example/0_openseg.data.npy")
    vlmaps = vlmapsdata[:, 3, :]
    vl_sem = vlmapsdata[:, 0, 0]

    vlmapsdata2 = np.load("/home/user/code/vlmaps/embedding-example/0_lseg.data.npy")
    vlmaps2 = vlmapsdata2[:, 3, :]
    vl_sem2 = vlmapsdata2[:, 0, 0]

    pca = PCA(n_components=2)
    pca.fit(vlmaps)
    pca2_vl = pca.transform(vlmaps)

    pca2 = PCA(n_components=2)
    pca2.fit(vlmaps2)
    pca2_vl2 = pca2.transform(vlmaps2)

    pca3 = PCA(n_components=2)
    pca3.fit(ensemble)
    pca_ens = pca3.transform(ensemble)
    pca_dis = pca3.transform(distill)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    # 1
    labels = np.unique(vl_sem)
    print("unique labels", len(labels), labels)
    print("cmap", len(colormap))
    legend = []
    i = 0
    for label in labels:
        idx = vl_sem == label
        pt = pca2_vl[idx]
        ax1.scatter(pt[:, 0], pt[:, 1], c=colormap[i])
        legend.append(str(label))
        i += 1
    ax1.set_title("OpenSeg")
    ax1.legend(legend)

    # 2
    labels2 = np.unique(vl_sem2)
    print("unique labels2", len(labels2), labels2)
    legend = []
    i = 0
    for label in labels2:
        idx = vl_sem2 == label
        pt = pca2_vl2[idx]
        ax2.scatter(pt[:, 0], pt[:, 1], c=colormap[i])
        legend.append(str(label))
        i += 1
    ax2.set_title("LSeg")

    # 3
    labels3 = np.unique(ens_lab)
    print("unique labels3", len(labels3), labels3)
    legend = []
    i = 0
    for label in labels2:
        idx = ens_lab == label
        pt_e = pca_dis[idx]
        ax3.scatter(pt_e[:, 0], pt_e[:, 1], c=colormap[i])
        legend.append(str(label))
        i += 1
    ax3.set_title("Ensemble")


    plt.show()
    exit()



    # RAW
    #print("raw: min, mean, max")
    #print(os.shape, np.min(os), np.mean(os), np.max(os))
    #(3891409, 512) -5468.0 0.0367 14620.0
    #print(vl.shape, np.min(vl), np.mean(vl), np.max(vl))
    #(198658, 512) -4.032070159912109 0.03189411294059693 9.017416000366211

    # oss = np.linalg.norm(os, axis=1)
    # oss = oss[~np.isinf(oss)]
    # print(oss.shape, np.all(np.abs(1 - oss) < 0.1), np.min(oss), np.mean(oss), np.max(oss))
    # sum, axis=1
    #(3891409,) False -3.297 18.8 22030.0
    # norm, axis=1
    #(3890728,) False 1.715 17.27 255.6

    # vls = np.linalg.norm(vl, axis=1)
    # print(vls.shape, np.all(np.abs(1 - vls) < 0.1), np.min(vls), np.mean(vls), np.max(vls))
    # sum, axis=1
    #(198658,) False -5.408788680764701 16.32978582558561 30.059583618480247
    # norm, axis=1
    #(198658,) False 2.2091787065546958e-26 12.507721786001955 14.293267872394374

    # MEANS
    # osm = np.mean(os, axis=0)
    # vlm = np.mean(vl, axis=0)
    # print("means: min, mean, max")
    # print(osm.shape, np.min(osm), np.mean(osm), np.max(osm))
    # #(512,) -3.592 0.03683 10.25
    # print(vlm.shape, np.min(vlm), np.mean(vlm), np.max(vlm))
    #(512,) -2.5859770575014394 0.031894112940596644 6.670617666958258

    fig, ax = plt.subplots()
    analyze(ensemble, ax, "ensemble")
    analyze(distill, ax, "distill")
    analyze(fusion, ax, "fusion")
    analyze(vlmaps, ax, "vlmaps")
    fig.legend(["ensemble", "distill", "fusion"])
    ax.set_yscale("log")
    plt.show()

if __name__ == "__main__":
    main()
