
import os
import numpy as np
from tqdm import tqdm

def select_files_with_same_name(folder1, folder2, folder3):
    files_list = []
    # Get list of file names in each folder
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))
    # Get the common file names among all three folders
    common_files = files1.intersection(files2, files3)
    # Iterate through common file names and create 3-tuples
    for file_name in common_files:
        file_path1 = os.path.join(folder1, file_name)
        file_path2 = os.path.join(folder2, file_name)
        file_path3 = os.path.join(folder3, file_name)
        files_list.append((file_path1, file_path2, file_path3))
    return files_list

def main():
    p1 = os.environ['BASE_DIR'] + "/comp/orig/gundam"
    p2 = os.environ['BASE_DIR'] + "/comp/orig/marvin"
    p3 = os.environ['BASE_DIR'] + "/comp/orig/local"

    files = select_files_with_same_name(p1, p2, p3)

    all_eq = 0
    a_b_eq = 0
    a_c_eq = 0
    b_c_eq = 0
    cnt = 0

    for file in tqdm(files):
        a = np.load(file[0])
        b = np.load(file[1])
        c = np.load(file[2])

        a_b = np.array_equal(a, b)
        a_c = np.array_equal(a, c)
        b_c = np.array_equal(b, c)


        if (a_b):
            a_b_eq += 1
        if (a_c):
            a_c_eq += 1
        if (b_c):
            b_c_eq += 1
        if (a_b and a_c and b_c):
            all_eq += 1
        cnt += 1

    print("gundam/marvin", a_b_eq, cnt, a_b_eq/cnt)
    print("gundam/local", a_c_eq, cnt, a_c_eq/cnt)
    print("marvin/local", b_c_eq, cnt, b_c_eq/cnt)
if __name__=="__main__":
    main()

