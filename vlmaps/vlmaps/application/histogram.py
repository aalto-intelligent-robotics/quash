from typing import Tuple, Dict, List
from vlmaps.map.vlmap import VLMap
import hydra
from omegaconf import DictConfig
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import click
import random
from sklearn.neighbors import KernelDensity
from diptest import diptest
from scipy.signal import find_peaks
from enum import Enum
from collections import Counter
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator


class DiptestResult(Enum):
    VeryLikelyUnimodal = 0
    LikelyUnimodal = 1
    Unknown = 2
    LikelyMultimodal = 3
    VeryLikelyMultimodal = 4

    def __str__(self):
        if self == DiptestResult.VeryLikelyUnimodal:
            return "Very Likely Unimodal"
        elif self == DiptestResult.LikelyUnimodal:
            return "Likely Unimodal"
        elif self == DiptestResult.Unknown:
            return "Unknown"
        elif self == DiptestResult.LikelyMultimodal:
            return "Likely Multimodal"
        elif self == DiptestResult.VeryLikelyMultimodal:
            return "Very Likely Multimodal"
        else:
            return "Unknown"


def get_cats(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    df = pd.read_csv(path, delimiter="\t")
    labels = pd.unique(df["mpcat40index"])
    labels.sort()  # type: ignore
    id_to_name = {}
    name_to_id = {}

    for label in labels:
        mask = df["mpcat40index"] == label
        hits = df[mask]
        names = hits["mpcat40"]
        unames = pd.unique(names)
        assert len(unames) == 1
        name = unames[0]
        id_to_name[label] = name
        name_to_id[name] = label

    return id_to_name, name_to_id


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    divisor = np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2)
    if divisor == 0:
        divisor = np.float32(1e-6)
    distance = np.dot(a, b) / divisor
    if distance > 1:
        distance = 1
    if distance < -1:
        distance = -1
    return distance


def get_eps(vals, min_samples) -> float:
    # Choose k = min_samples - 1
    k = min_samples - 1
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vals)
    distances, _ = nbrs.kneighbors(vals)
    k_distances = np.sort(distances[:, -1])  # distance to kth neighbor

    # Find elbow
    x = np.arange(len(k_distances))
    knee = KneeLocator(x, k_distances, curve="convex", direction="increasing")

    # Robust handling of knee result
    if knee.knee is not None:
        elbow_index = knee.knee
        if isinstance(elbow_index, (list, np.ndarray)):
            elbow_index = elbow_index[0]
        if elbow_index >= len(k_distances):
            elbow_index = len(k_distances) - 1
        eps = float(k_distances[elbow_index])
    else:
        # Fallback: pick a high percentile as a conservative guess
        eps = float(np.percentile(k_distances, 90))

    if eps <= 0 or eps == float("inf"):
        eps = 1

    return eps

def get_modality_score(vals: np.ndarray, verbose: bool = False):
    num_vals = vals.shape[0]

    # DISTS
    ################################################################################
    distsl = []
    for i in range(0, num_vals):
        for j in range(i + 1, num_vals):
            d = cosine_distance(vals[i, :], vals[j, :])
            distsl.append(d)
    dists = np.array(distsl)

    # PCA + Empirical p-value
    ################################################################################
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vals)
    mean_vec = np.mean(vals, axis=0)
    mean_pca = pca.transform(mean_vec.reshape(1, -1))

    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(proj)
    log_densities = kde.score_samples(proj)
    test_density = kde.score_samples(mean_pca.reshape(1, -1))[0]

    p_value = np.mean(log_densities <= test_density)
    if verbose:
        print(f"p_value: {p_value}")

    # GMM + BIC
    ################################################################################
    lowest_bic = np.inf
    for n_components in range(1, 10):
        if n_components >= num_vals:
            continue
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag").fit(vals)
        bic = gmm.bic(vals)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
    gmm_score = best_gmm.n_components
    if verbose:
        print(f"GMM: Optimal number of components: {gmm_score}")

    # histogram KDE
    ################################################################################
    # Assume vals is a NumPy array of shape (n_samples, n_features)
    # Step 1: Reduce to 1D using PCA
    pca = PCA(n_components=1)
    vals_1d = pca.fit_transform(vals).flatten()

    # Step 2: Kernel Density Estimation
    x_grid = np.linspace(vals_1d.min(), vals_1d.max(), 1000).reshape(-1, 1)

    bandwidth = 0.3 * np.std(vals_1d)
    # bandwidth = "scott"
    # bandwidth = "silverman"
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(vals_1d.reshape(-1, 1))
    log_density = kde.score_samples(x_grid)
    density = np.exp(log_density)
    _, p_value = diptest(vals_1d)
    if p_value >= 0.75:
        dip_result = "Very likely Unimodal"
        dip_score = 1
    elif p_value < 0.75 and p_value >= 0.6:
        dip_result = "Likely Unimodal"
        dip_score = 1
    elif p_value < 0.6 and p_value >= 0.4:
        dip_result = "Unknown"
        dip_score = 1.5
    elif p_value < 0.4 and p_value >= 0.25:
        dip_result = "Likely Multimodal"
        dip_score = 2
    else:
        dip_result = "Very likely Multimodal"
        dip_score = 2

    if verbose:
        print(f"Dip test p-value: {p_value:.4f}")

    # Optional: Count number of peaks in KDE curve
    peaks, _ = find_peaks(density)
    kde_score = len(peaks)
    if verbose:
        print(f"Number of peaks in KDE: {kde_score}")

    # DBSCAN
    ################################################################################
    #D = euclidean_distances(vals)
    D = cosine_distances(vals)
    #D = 1 - D
    dbscan = DBSCAN(eps=0.025, min_samples=5, metric="precomputed")
    labels = dbscan.fit_predict(D)
    unique_labels = np.unique(labels)
    dbscan_score = unique_labels.shape[0]
    if verbose:
        print(f"DBSCAN labels: {dbscan_score} - {unique_labels}")

    # Score
    ################################################################################
    score = (gmm_score + dbscan_score + kde_score + dip_score) / 4
    if score >= 1.75:
        overall_result = "Likely Multimodal"
    else:
        overall_result = "Likely Unimodal"
    if verbose:
        print(f"Score: {score} -> {overall_result}")
    return score, proj, mean_pca, dists, x_grid, density, vals_1d, D


def visualize(vlmap: VLMap):
    # ANALYZE
    idx = 1
    keys = list(vlmap.grid_histogram.keys())
    histograms = None
    if Path("histogram-analysis.data").exists():
        with open("histogram-analysis.data", "rb") as f:
            histograms = pickle.load(f)

    while True:
        id = keys[idx]
        vals = vlmap.grid_histogram[id]
        vals = np.array(vals)
        num_vals = vals.shape[0]

        print(f"{id}: {vals.shape}")

        # # GMM + BIC
        # lowest_bic = np.inf
        # for n_components in range(1, 10):
        #     if n_components >= num_vals:
        #         continue
        #     gmm = GaussianMixture(n_components=n_components, covariance_type="full").fit(vals)
        #     bic = gmm.bic(vals)
        #     if bic < lowest_bic:
        #         lowest_bic = bic
        #         best_gmm = gmm
        # print(f"GMM: Optimal number of components: {best_gmm.n_components}")

        # # COSINE DISTANCES
        # distsl = []
        # for i in range(0, num_vals):
        #     for j in range(i + 1, num_vals):
        #         d = cosine_distance(vals[i, :], vals[j, :])
        #         distsl.append(d)
        # dists = np.array(distsl)

        # # PCA
        # pca = PCA(n_components=2)
        # proj = pca.fit_transform(vals)
        # mean_vec = np.mean(vals, axis=0)
        # mean_pca = pca.transform(mean_vec.reshape(1, -1))

        # # Diptest
        # pca = PCA(n_components=1)
        # vals_1d = pca.fit_transform(vals).flatten()
        # x_grid = np.linspace(vals_1d.min(), vals_1d.max(), 1000).reshape(-1, 1)

        # bandwidth = 0.25 * np.std(vals_1d)
        # # bandwidth = "scott"
        # # bandwidth = "silverman"
        # kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(vals_1d.reshape(-1, 1))
        # log_density = kde.score_samples(x_grid)
        # density = np.exp(log_density)
        # _, p_value = diptest(vals_1d)
        # if p_value >= 0.75:
        #     dip_result = "Very likely Unimodal"
        # elif p_value < 0.75 and p_value >= 0.6:
        #     dip_result = "Likely Unimodal"
        # elif p_value < 0.6 and p_value >= 0.4:
        #     dip_result = "Unknown"
        # elif p_value < 0.4 and p_value >= 0.25:
        #     dip_result = "Likely Multimodal"
        # else:
        #     dip_result = "Very likely Multimodal"

        # print(f"Dip test p-value: {p_value:.4f} -> {dip_result}")

        # DBSCAN
        # D = cosine_distances(vals)
        D = euclidean_distances(vals)
        dbscan = DBSCAN(eps=get_eps(vals, 5), min_samples=5, metric="precomputed")
        labels = dbscan.fit_predict(D)
        pca = PCA(n_components=2)
        vals_2d = pca.fit_transform(vals)
        unique_labels = np.unique(labels)
        print(f"DSCAN clusters (orig): {unique_labels.shape[0]}")
        colors = ["r", "b", "g", "c", "m", "y", "k"]

        # PCAD DBSCAN
        pca = PCA(n_components=0.95)  # keep 95% variance
        X_reduced = pca.fit_transform(vals)
        min_samples = pca.n_components_ + 1
        D_reduced = cosine_distances(X_reduced)
        db = DBSCAN(eps=get_eps(vals, min_samples), min_samples=min_samples, metric='precomputed')
        reduced_labels = db.fit_predict(D_reduced)
        reduced_unique_labels = np.unique(reduced_labels)
        print(f"DSCAN clusters (reduced): {reduced_unique_labels.shape[0]}")

        score, proj, mean_pca, dists, x_grid, density, vals_1d, D = get_modality_score(vals, verbose=True)

        # VIZ
        fig, ax = plt.subplots(1, 5)
        ax[0].scatter(proj[:, 0], proj[:, 1], c="b")
        ax[0].scatter(mean_pca[:, 0], mean_pca[:, 1], c="r")
        ax[0].set_title("PCA")

        ax[1].hist(dists)
        ax[1].set_title("Cosine distances")

        ax[2].plot(x_grid, density, label="KDE")
        ax[2].hist(vals_1d, bins=50, density=True, alpha=0.4, label="Histogram")
        ax[2].set_title("1D Projection KDE + Histogram")
        ax[2].legend()

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax[3].scatter(
                vals_2d[mask, 0],
                vals_2d[mask, 1],
                c=[color],
                label=f"Cluster {label}" if label != -1 else "Noise",
            )
        ax[3].set_title("DBSCAN orig dim")

        for label, color in zip(reduced_unique_labels, colors):
            mask = labels == label
            ax[4].scatter(
                vals_2d[mask, 0],
                vals_2d[mask, 1],
                c=[color],
                label=f"Cluster {label}" if label != -1 else "Noise",
            )
        ax[4].set_title(f"DBSCAN 95% var dim ({X_reduced.shape[1]})")
        plt.show()

        click.echo("← → r p q:", nl=False)
        c = click.getchar()
        click.echo()
        if c == "q":
            break
        elif c == "\x1b[D":
            idx -= 1
            if idx < 0:
                idx = len(keys)
        elif c == "\x1b[C":
            idx += 1
            if idx > len(keys):
                idx = 0
        elif c == "r":
            idx = random.choice(keys)
        elif c == "p":
            if histograms is not None:
                print("0: VeryLikelyUnimodal")
                print("1: LikelyUnimodal")
                print("2: Unknown")
                print("3: LikelyMultimodal")
                print("4: VeryLikelyMultimodal")
                try:
                    choice = int(input(": "))
                except ValueError:
                    print("Unknown choice")
                    exit()
                val = DiptestResult(choice)
                print(val)
                ch_keys = histograms[val]
                idx = random.choice(ch_keys)

def analyze_score(vlmap: VLMap):
    keys = list(vlmap.grid_histogram.keys())
    dipresults = []
    dipdict: Dict[DiptestResult, List[int]] = {}
    dipdict[DiptestResult.VeryLikelyUnimodal] = []
    dipdict[DiptestResult.LikelyUnimodal] = []
    dipdict[DiptestResult.Unknown] = []
    dipdict[DiptestResult.LikelyMultimodal] = []
    dipdict[DiptestResult.VeryLikelyMultimodal] = []
    for i, key in enumerate(tqdm(keys)):
        if key == -1:
            continue
        vals = vlmap.grid_histogram[key]
        vals = np.array(vals)

        # diptest requires 3 entries
        if vals.shape[0] <= 3:
            continue

        try:
            score, _, _, _, _, _, _, _ = get_modality_score(vals, verbose=False)
        except Exception as e:
            print(e)
            continue

        if score >= 1.75:
            dip_result = DiptestResult.LikelyMultimodal
        else:
            dip_result = DiptestResult.LikelyUnimodal
        dipresults.append(dip_result)

    # Count occurrences of each enum
    enum_counts = Counter(dipresults)
    out = dict(enum_counts)

    # Display the counts
    for key, value in out.items():
        print(f"{key}: {value}")

    with open("histogram-analysis.data", "wb") as f:
        pickle.dump(dipdict, f)


def analyze_dip(vlmap: VLMap):
    keys = list(vlmap.grid_histogram.keys())
    dipresults = []
    dipdict: Dict[DiptestResult, List[int]] = {}
    dipdict[DiptestResult.VeryLikelyUnimodal] = []
    dipdict[DiptestResult.LikelyUnimodal] = []
    dipdict[DiptestResult.Unknown] = []
    dipdict[DiptestResult.LikelyMultimodal] = []
    dipdict[DiptestResult.VeryLikelyMultimodal] = []
    for i, key in enumerate(tqdm(keys)):
        if key == -1:
            continue
        vals = vlmap.grid_histogram[key]
        vals = np.array(vals)

        # diptest requires 3 entries
        if vals.shape[0] <= 3:
            continue

        pca = PCA(n_components=1)
        vals_1d = pca.fit_transform(vals).flatten()

        # Step 3: Dip Test for Unimodality
        _, p_value = diptest(vals_1d)

        if p_value >= 0.75:
            dip_result = DiptestResult.VeryLikelyUnimodal
            dipdict[DiptestResult.VeryLikelyUnimodal].append(i)
        elif p_value < 0.75 and p_value >= 0.6:
            dip_result = DiptestResult.LikelyUnimodal
            dipdict[DiptestResult.LikelyUnimodal].append(i)
        elif p_value < 0.6 and p_value >= 0.4:
            dip_result = DiptestResult.Unknown
            dipdict[DiptestResult.Unknown].append(i)
        elif p_value < 0.4 and p_value >= 0.25:
            dip_result = DiptestResult.LikelyMultimodal
            dipdict[DiptestResult.LikelyMultimodal].append(i)
        else:
            dip_result = DiptestResult.VeryLikelyMultimodal
            dipdict[DiptestResult.VeryLikelyMultimodal].append(i)

        dipresults.append(dip_result)

    # Count occurrences of each enum
    enum_counts = Counter(dipresults)
    out = dict(enum_counts)

    # Display the counts
    for key, value in out.items():
        print(f"{key}: {value}")

    with open("histogram-analysis-dip.data", "wb") as f:
        pickle.dump(dipdict, f)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="histogram_cfg.yaml",
)
def main(config: DictConfig):
    # LOAD
    print("histogram")
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    id = config.scene_id
    vlmap = VLMap(config.map_config, data_dir=data_dirs[id])
    vlmap.load_map_override(data_dirs[id], config.direct_path_path)
    print("read map from:", vlmap.map_save_path)

    choice = input("(V)isualize or (A)nalyze: ")
    if choice.lower() == "v":
        visualize(vlmap)
    if choice.lower() == "a":
        analyze_score(vlmap)
        #analyze_dip(vlmap)
    else:
        print("Unknown choice")
        exit()


if __name__ == "__main__":
    main()
