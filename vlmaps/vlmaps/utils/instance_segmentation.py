import numpy as np
from tqdm import tqdm
import numba as nb

@nb.njit
def printNumRegions(regions):
    uqr = np.unique(regions)
    regs = len(uqr)
    print("got", regs, "regions")
    print(uqr)
    return regs

@nb.njit
def progress(i_, l_):
    if (i_ % 100 == 0):
        p = (i_/l_)*100
        print(p, "%")

@nb.njit(nb.types.UniTuple(nb.int64[:],2)(nb.int64[:]))
def unique_with_counts(arr):
    unique_elements = []
    element_counts = {}

    for elem in arr:
        if elem in element_counts:
            element_counts[elem] += 1
        else:
            element_counts[elem] = 1
            unique_elements.append(elem)

    unique = np.array(unique_elements, dtype=np.int64)
    counts = np.array([element_counts[elem] for elem in unique_elements], dtype=np.int64)
    return unique, counts


#@nb.njit(nb.types.UniTuple(nb.int64[:], 2)(nb.int64[:,:,:]))
@nb.njit
def unique_with_counts_3d(arr):
    arr = arr.reshape(-1)
    return unique_with_counts(arr)


#@nb.njit((nb.int64[:, :, :], nb.int64[:, :, :]), parallel=True, fastmath=True)
@nb.njit(parallel=True, fastmath=True)
def checkSemantics(regions_, grid_):
    uq_regions = np.unique(regions_)
    # for region in tqdm(uq_regions, leave=False, desc="check semantics"):
    c = 0
    l = len(uq_regions)

    #print(l, "regions to check")
    lin_grid = grid_.reshape(-1)
    lin_regions = regions_.reshape(-1)
    # for region in uq_regions:
    for i in nb.prange(len(uq_regions)):
        region = uq_regions[i]
        # if (c % 100 == 0):
        #     p = c/l*100
        #     print(p, "%")
        # c += 1

        idx = (lin_regions == region)
        sem = lin_grid[idx]
        # sem = np.take(grid_, idx)
        # us, cs = np.unique(sem, return_counts=True)
        us, cs = unique_with_counts(sem)

        if (us.size > 1 and region >= 0):
            print(str(region) + ": MIXED SEMANTICS!!!!!")
        # else:
            # print(str(region) + ": all ok")


@nb.njit(nb.int64[:,:,:](nb.int64,nb.int64,nb.int64,nb.boolean[:,:,:],nb.int64[:,:,:],nb.int64[:,:,:],nb.int64))
def grow(maxx, maxy, maxz, occupied, regions, grid, grow_range):
    # for x in tqdm(range(0, maxx), leave=False, desc="grow"):
    for x in range(0, maxx):
        # progress(x, maxx)
        for y in range(0, maxy):
            for z in range(0, maxz):
                if (not occupied[x, y, z]):
                    continue
                current_region = regions[x, y, z]
                current_label = grid[x, y, z]

                for dx in range(-1*grow_range, grow_range+1):
                    for dy in range(-1*grow_range, grow_range+1):
                        for dz in range(-1*grow_range, grow_range+1):
                            # indentity, not neighbor
                            if (dx == 0 and dy == 0 and dz == 0):
                                continue
                            # check bounds
                            if (not ((grid.shape[0] > x+dx and x+dx >= 0) and (grid.shape[1] > y+dy and y+dy >= 0) and (grid.shape[2] > z+dz and z+dz >= 0))):
                                continue
                            # check occupied
                            if (not occupied[x + dx, y + dy, z + dz]):
                                continue
                            # check labels
                            comparison_label = grid[x + dx, y + dy, z + dz]
                            # comparison_region = regions[x + dx, y + dy, z + dz]
                            if (comparison_label == current_label):
                                regions[x+dx, y+dy, z+dz] = current_region
                                # regions[regions == comparison_region] = current_region
    return regions


@nb.njit(parallel=True)
def join(maxx, maxy, maxz, occupied, regions, grid, join_range):
    # for y in tqdm(range(0, maxy), leave=False, desc="join"):
    for y in nb.prange(0, maxy):
        # progress(y,maxy)
        for x in nb.prange(0, maxx):
            for z in nb.prange(0, maxz):
                if (not occupied[x, y, z]):
                    continue
                current_region = regions[x, y, z]
                current_label = grid[x, y, z]
                for dx in nb.prange(-1*join_range, join_range+1):
                    for dy in nb.prange(-1*join_range, join_range+1):
                        for dz in nb.prange(-1*join_range, join_range+1):
                            # indentity, not neighbor
                            if (dx == 0 and dy == 0 and dz == 0):
                                continue
                            # check bounds
                            if (not ((grid.shape[0] > x+dx and x+dx >= 0) and (grid.shape[1] > y+dy and y+dy >= 0) and (grid.shape[2] > z+dz and z+dz >= 0))):
                                continue
                            # check occupied
                            if (not occupied[x + dx, y + dy, z + dz]):
                                continue
                            # check labels
                            comparison_label = grid[x + dx, y + dy, z + dz]
                            region = regions[x + dx, y + dy, z + dz]
                            if (comparison_label == current_label and region != current_region):

                                # idx = (regions == region).reshape(-1)
                                # reg = np.copy(regions).reshape(-1)
                                # reg[idx] = current_region
                                # regions = reg

                                idx = regions == region
                                # regions[idx] = current_region
                                for i in nb.prange(idx.shape[0]):
                                    for j in nb.prange(idx.shape[1]):
                                        for k in nb.prange(idx.shape[2]):
                                            if (idx[i, j, k]):
                                                regions[i, j, k] = current_region
    return regions

@nb.njit(parallel=True, fastmath=True)
def removeSmallRegions(regions, instanceMinSize):
    orig_shape = regions.shape
    regions = regions.reshape(-1)

    #unique, counts = np.unique(regions, return_counts=True)
    unique, counts = unique_with_counts(regions)

    #regionSizes = zip(unique, counts)
    rem = 0
    #for i, (reg, size) in tqdm(enumerate(regionSizes), leave=False, desc="remove small regions"):
    for i in nb.prange(len(unique)):
        #progress(i,len(unique))
        reg = unique[i]
        size = counts[i]
        if (size < instanceMinSize):
            idx = regions == reg
            regions[idx] = -1
            rem += idx.sum()
            # for a in nb.prange(idx.shape[0]):
            #     for b in nb.prange(idx.shape[1]):
            #         for c in nb.prange(idx.shape[2]):
            #             regions[a,b,c] = -1
            #             rem += 1

    # print("removed", rem, "small instances")
    regions = regions.reshape(orig_shape)

    return regions

@nb.njit(parallel=True)
def renumberRegions(regions):
    orig_shape = regions.shape
    regions = regions.reshape(-1)

    ulist = np.unique(regions)
    # updatelist = []
    for i in nb.prange(len(ulist)):
        reg = ulist[i]
        if (reg == -1):
            continue
        idx = np.where(ulist == reg)
        idx = idx[0]
        # updatelist.append((reg, idx))
        regions[regions == reg] = idx

    regions = regions.reshape(orig_shape)
    return regions

@nb.njit(parallel=True)
def instancesFromRegions(maxx, maxy, maxz, indices, regions, instances):
    #for x in tqdm(range(0, maxx), leave=False, desc="output convert"):
    for x in nb.prange(0, maxx):
        for y in nb.prange(0, maxy):
            for z in nb.prange(0, maxz):
                idx = indices[x, y, z]
                if (idx >= 0):
                    instances[idx] = regions[x, y, z]
    return instances

def instance_segmentation(semantics, pos, instanceMinSize, grow_range=1, join_range=1):
    # initialize
    instances = np.full_like(semantics, -1, dtype=np.int32)

    # print("labels:")
    us, cs = np.unique(semantics, return_counts=True)
    # print(np.stack((us, cs)).T)

    #! preprocess
    minx = np.min(pos[:, 0])
    miny = np.min(pos[:, 1])
    minz = np.min(pos[:, 2])
    pos[:, 0] = pos[:, 0] - minx
    pos[:, 1] = pos[:, 1] - miny
    pos[:, 2] = pos[:, 2] - minz
    maxx = np.max(pos[:, 0])
    maxy = np.max(pos[:, 1])
    maxz = np.max(pos[:, 2])

    # copy to 3d grid
    grid = np.full((maxx+1, maxy+1, maxz+1), -1)
    occupied = np.full((maxx+1, maxy+1, maxz+1), False)
    regions = np.full((maxx+1, maxy+1, maxz+1), -1)
    indices = np.full((maxx+1, maxy+1, maxz+1), -1)
    #for i in tqdm(range(pos.shape[0]), leave=False, desc="Create grid"):
    for i in range(pos.shape[0]):
        xyz = pos[i]
        grid[xyz[0], xyz[1], xyz[2]] = semantics[i]
        occupied[xyz[0], xyz[1], xyz[2]] = True
        indices[xyz[0], xyz[1], xyz[2]] = i

    # create regions
    # idx = 0
    # for x in tqdm(range(0, maxx), desc="Create initial regions", leave=False):
    #     for y in range(0, maxy):
    #         for z in range(0, maxz):
    #             oc = occupied[x, y, z]
    #             if (oc):
    #                 regions[x, y, z] = idx
    #                 idx += 1
    init_inidices = np.array(range((maxx+1)*(maxy+1)*(maxz+1)))
    init_inidices = init_inidices.reshape(maxx+1, maxy+1, maxz+1)
    regions = init_inidices
    regions[occupied == False] = -1

    # _ = printNumRegions(regions)

    ############################################################################
    # region growing
    it = 0
    prev = 0
    while (it < 10):
        # print("*"*40)
        # print("it", it)

        # print("grow")
        regions = grow(maxx, maxy, maxz, occupied, regions, grid, grow_range)
        # print("="*20)

        # print("join")
        regions = join(maxx, maxy, maxz, occupied, regions, grid, join_range)
        # print("="*20)

        # print("check semantics")
        # checkSemantics(regions, grid)
        # print("="*20)

        it += 1
        # regs = printNumRegions(regions)

        # print("="*20)
        uqr = np.unique(regions)
        regs = len(uqr)
        if (regs == prev):
            # print("region growing done")
            # print("="*20)
            break
        prev = regs

    ############################################################################
    # remove too small instances
    # print("remove small regions")
    regions = removeSmallRegions(regions, instanceMinSize)
    # _ = printNumRegions(regions)
    # print("="*20)

    # print("check semantics")
    # checkSemantics(regions, grid)
    # print("="*20)

    ############################################################################
    # renumber
    # print("renumber regions")
    regions = renumberRegions(regions)
    # _ = printNumRegions(regions)
    # print("="*20)

    # print("check semantics")
    # checkSemantics(regions, grid)
    # print("="*20)

    ############################################################################
    # for (reg, idx) in updatelist:
    #     regions[regions == reg] = idx

    # back to semantics
    # print("instances from regions")
    instances = instancesFromRegions(maxx, maxy, maxz, indices, regions, instances)
    # _ = printNumRegions(regions)
    # _ = printNumRegions(instances)
    # print("="*20)

    # print("check semantics")
    # checkSemantics(instances, semantics)
    # print("="*20)

    ############################################################################
    # print("DONE!")
    return instances
