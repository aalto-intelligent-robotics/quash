import argparse

from clip_embeddings.embeddingCreator import EmbeddingCreator
import numpy as np
from tqdm import tqdm
from scipy import stats as st, special, spatial
import matplotlib.pyplot as plt
import math
#import torch
#from geomloss import SamplesLoss

def distance(a, b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

def getEmbeddings(list, creator):
    f = open(list, "r")
    lines = f.readlines()
    l = len(lines)
    embeddings = np.zeros((l, 512))
    idx = 0
    for line in tqdm(lines):
        line = line.rstrip().lstrip()
        query = creator.get_text_embedding(line)
        embeddings[idx, :] = query
    f.close()

    return embeddings

def getMean(embeddings):
    mean = np.mean(embeddings, axis=0)
    return mean

def bin(p, q):
    pmn = np.min(p)
    qmn = np.min(q)
    pmx = np.max(p)
    qmx = np.max(q)
    mn = pmn if pmn < qmn else qmn
    mx = pmx if pmx > qmx else qmx
    numpoints = np.size(p) if np.size(p) > np.size(q) else np.size(q)
    k = int(np.ceil(np.sqrt(numpoints)))
    (hp, _) = np.histogram(p, bins=k, range=(mn, mx), density=True)
    (hq, _) = np.histogram(q, bins=k, range=(mn, mx), density=True)

    hp = np.where(hp == 0, hp + 1e-6, hp)
    hq = np.where(hq == 0, hq + 1e-6, hq)

    # print("p", p)
    # print("q", q)
    # print("hp", hp)
    # print("hq", hq)

    return hp, hq

def binned_run(p, q, f):
    assert(p.shape[1] == q.shape[1])
    l = p.shape[1]
    vals = np.zeros((l))
    for i in range(l):
        hp, hq = bin(p[:,i], q[:,i])
        vals[i] = f(hp, hq)
    return vals

def fit_normal(p):
    mean = np.mean(p, axis=0)
    cov = np.cov(p, rowvar=False)



    # print("fit normal")
    # print("m", mean)
    # print("c", cov)

    return mean, cov

def parametric_kld(p,q):
    (p_mean, p_cov) = fit_normal(p)
    (q_mean, q_cov) = fit_normal(q)
    n = np.size(p_mean)
    # https://stanford.edu/~jduchi/projects/general_notes.pdf

    print("p_cov shape", p_cov.shape, "rank", np.linalg.matrix_rank(p_cov))
    print("q_cov shape", q_cov.shape, "rank", np.linalg.matrix_rank(q_cov))

    # det is nonzero only for invertible
    t1 = np.log(np.linalg.det(q_cov)/np.linalg.det(p_cov))
    t2 = np.trace(np.linalg.inv(q_cov) @ p_cov)
    t3 = (q_mean - p_mean).T @ np.linalg.inv(q_cov) @ (q_mean - p_mean)

    print("t1 shape", t1.shape, "rank", np.linalg.matrix_rank(t1))
    print("t2 shape", t2.shape, "rank", np.linalg.matrix_rank(t2))
    print("t3 shape", t3.shape, "rank", np.linalg.matrix_rank(t3))
    return 0.5*(t1 - n + t2 + t3)

def aggregate_dimensions(vals):
    return np.linalg.norm(vals, ord=2)

def binned_kld(p, q):
    vals = binned_run(p, q, kullback_leibler)
    return aggregate_dimensions(vals)

def binned_jsd(p, q):
    vals = binned_run(p, q, jensen_shannon)
    return aggregate_dimensions(vals)

def binned_bha(p, q):
    vals = binned_run(p, q, bhattacharyya)
    return aggregate_dimensions(vals)

def binned_wsd(p, q):
    vals = binned_run(p, q, wasserstein)
    return aggregate_dimensions(vals)

def kullback_leibler(p, q):
    if (p.shape != q.shape):
        return -1
    return np.sum(p*np.log(p/q))

def jensen_shannon(p, q):
    m = 0.5*(p+q)
    return 0.5 * (kullback_leibler(p,m) + kullback_leibler(q,m))

def bhattacharyya(p, q):
    if(p.shape != q.shape):
        return -1
    return -np.log(np.sum(np.sqrt(p * q)))

def wasserstein(p,q):
    if (p.shape != q.shape):
        return -1
    return np.mean(np.abs((np.sort(p)-np.sort(q))))

def makePDF(m):
    mn = np.min(m)
    m = (m - mn)+1e-5
    mx = np.max(m)
    m = m * (1/mx)
    m = m / np.sum(m)
    return m

def distPrint(p, title):
    mn = np.min(p)
    mx = np.max(p)
    s = np.sum(p)
    print(title, "shape", p.shape, "min", mn, "max", mx, "sum", s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./wasserstein_test.py")
    parser.add_argument(
        '--a', '-a',
        dest="a",
        type=str
    )
    parser.add_argument(
        '--b', '-b',
        dest="b",
        type=str
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cpu or cuda.",
    )
    FLAGS, unparsed = parser.parse_known_args()
    a = FLAGS.a
    b = FLAGS.b
    device = FLAGS.device


    runtype = 0

    if(runtype == 0 or runtype == 1):
        creator = EmbeddingCreator(device)
        embeddings_a = getEmbeddings(a, creator)
        embeddings_b = getEmbeddings(b, creator)

        if runtype == 0:
            dist_a = embeddings_a
            dist_b = embeddings_b
        if runtype == 1:
            mean_a = getMean(embeddings_a)
            mean_b = getMean(embeddings_b)
            dist_a = mean_a
            dist_b = mean_b
    else:
        dim = 500
        cnt = 100
        dist_a = np.random.normal(0, 1, (cnt+1, dim))
        dist_b = np.random.normal(0, 1, (cnt-1, dim))

    distPrint(dist_a, "dist_a")
    distPrint(dist_b, "dist_b")


    # wsd = stats.wasserstein_distance(dist_a, dist_b)
    # print("Wasserstein distance :", wsd)

    # jsd = spatial.distance.jensenshannon(dist_a, dist_b)**2
    # print("Jensen-Shannon divergence :", jsd)

    # kld = special.kl_div(dist_a, dist_b)
    # print("Kullback-Leibler divergence :", np.sum(kld))

    wsd2 = binned_wsd(dist_a, dist_b)
    print("Binned Wasserstein distance:", wsd2)

    jsd2 = binned_jsd(dist_a, dist_b)
    print("Binned Jensen-Shannon divergence:", jsd2)

    kld2 = binned_kld(dist_a, dist_b)
    print("Binned Kullback-Leibler divergence:", kld2)

    bhd = binned_bha(dist_a, dist_b)
    print("Binned Bhattacharyya distance:", bhd)

    # wsd2 = parametric_wsd(dist_a, dist_b)
    # print("Parametric Wasserstein distance:", wsd2)

    # jsd2 = parametric_jsd(dist_a, dist_b)
    # print("Parametric Jensen-Shannon divergence:", jsd2)

    kld2 = parametric_kld(dist_a, dist_b)
    print("Parametric Kullback-Leibler divergence:", kld2)

    # bhd = parametric_bha(dist_a, dist_b)
    # print("Parametric Bhattacharyya distance:", bhd)


