import os
import re

def main():
    base_path = "/home/user/data/datasets/Datasets_HSN/ADE20K_2021_17_01/images"

    # Dictionary to store filenames
    filenames = {"train": set(), "val": set()}

    # Regex pattern to match the filenames: "ADE_train_{num}.jpg"
    pattern_jpg = re.compile(r"ADE_(train|val)_(\d+)\.jpg")

    # Regex pattern to match the corresponding .pkl file: "ADE_{split}_{num}_{variation}.pkl"
    pattern_pkl = re.compile(r"ADE_(train|val)_(\d+)_(ade_full|ade_all)\.pkl")

    # Dictionary to store missing information for each variation
    variation_status = {
        "ade_full": {"missing": 0, "found": set()},
        "ade_all": {"missing": 0, "found": set()}
    }

    # Traverse the directory and its subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jpg"):
                match_jpg = pattern_jpg.match(file)
                if match_jpg:
                    category = match_jpg.group(1)  # Either "train" or "validation"
                    num = int(match_jpg.group(2))  # The number in the filename
                    filenames[category].add(num)

                    # Check if corresponding .pkl files with variations exist
                    for variation in ["ade_full", "ade_all"]:
                        pkl_filename = os.path.join(root, f"ADE_{category}_{match_jpg.group(2)}_{variation}.pkl")
                        if os.path.exists(pkl_filename):
                            variation_status[variation]["found"].add(num)
                        else:
                            variation_status[variation]["missing"] += 1

    # Check if all numbers from 1 to max are found for both categories
    for category, nums in filenames.items():
        if nums:
            max_num = max(nums)
            missing = set(range(1, max_num + 1)) - nums
            if missing:
                print(f"Missing {category} .jpg files: {len(missing)}")
            else:
                print(f"All {category} .jpg files are present from 1 to {max_num}.")
        else:
            print(f"No {category} .jpg files found.")

    # Report missing or present .pkl files for each variation
    for variation, status in variation_status.items():
        if status["missing"] > 0:
            print(f"{status['missing']} {variation} .pkl files are missing.")
        else:
            if status["found"]:
                print(f"All {variation} .pkl files are present.")
            else:
                print(f"No {variation} .pkl files found.")

if __name__ == "__main__":
    main()
















# import os
# import re

# def main():
#     base_path = "/home/user/data/datasets/Datasets_HSN/ADE20K_2021_17_01/images"

#     # Dictionary to store filenames
#     filenames = {"train": set(), "val": set()}

#     # Regex pattern to match the filenames: "ADE_train_{num}.jpg"
#     pattern_jpg = re.compile(r"ADE_(train|val)_(\d+)\.jpg")

#     # Regex pattern to match the corresponding .pkl file: "ADE_{split}_{num}_{variation}.pkl"
#     pattern_pkl = re.compile(r"ADE_(train|val)_(\d+)_(ade_full|ade_all)\.pkl")

#     # Missing corresponding .pkl files
#     missing_pkl_files = []

#     # Traverse the directory and its subdirectories
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             if file.endswith(".jpg"):
#                 match_jpg = pattern_jpg.match(file)
#                 if match_jpg:
#                     category = match_jpg.group(1)  # Either "train" or "validation"
#                     num = int(match_jpg.group(2))  # The number in the filename
#                     filenames[category].add(num)

#                     # Check if corresponding .pkl file with variation exists
#                     found_pkl = False
#                     for variation in ["ade_full", "ade_all"]:
#                         pkl_filename = os.path.join(root, f"ADE_{category}_{num}_{variation}.pkl")
#                         if os.path.exists(pkl_filename):
#                             found_pkl = True
#                             break

#                     if not found_pkl:
#                         missing_pkl_files.append(f"ADE_{category}_{num}_*.pkl (any variation)")

#     # Check if all numbers from 1 to max are found for both categories
#     for category, nums in filenames.items():
#         if nums:
#             max_num = max(nums)
#             missing = set(range(1, max_num + 1)) - nums
#             if missing:
#                 print(f"Missing {category} .jpg files: {sorted(missing)}")
#             else:
#                 print(f"All {category} .jpg files are present from 1 to {max_num}.")
#         else:
#             print(f"No {category} .jpg files found.")

#     # Report missing .pkl files
#     if missing_pkl_files:
#         print("\nMissing corresponding .pkl files (with variations):")
#         # for pkl_file in missing_pkl_files:
#         #     print(pkl_file)
#     else:
#         print("\nAll .jpg files have corresponding .pkl files (with variations).")

# if __name__ == "__main__":
#     main()


















# import os
# import re

# def main():
#     base_path = "/home/user/data/datasets/Datasets_HSN/ADE20K_2021_17_01/images"

#     # Dictionary to store filenames
#     filenames = {"train": set(), "val": set()}

#     # Regex pattern to match the filenames: "ADE_train_{num}" or "ADE_val_{num}"
#     pattern = re.compile(r"ADE_(train|val)_(\d+)\.jpg")

#     # Missing corresponding .pkl files
#     missing_pkl_files = []

#     # Traverse the directory and its subdirectories
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             if file.endswith(".jpg"):
#                 match = pattern.match(file)
#                 if match:
#                     category = match.group(1)  # Either "train" or "val"
#                     num = int(match.group(2))  # The number in the filename
#                     filenames[category].add(num)

#                     # Check if corresponding .pkl file exists
#                     pkl_filename = os.path.join(root, file.replace(".jpg", ".pkl"))
#                     if not os.path.exists(pkl_filename):
#                         missing_pkl_files.append(pkl_filename)

#     # Check if all numbers from 1 to max are found for both categories
#     for category, nums in filenames.items():
#         if nums:
#             max_num = max(nums)
#             missing = set(range(1, max_num + 1)) - nums
#             if missing:
#                 print(f"Missing {category} files: {sorted(missing)}")
#             else:
#                 print(f"All {category} files are present from 1 to {max_num}.")
#         else:
#             print(f"No {category} files found.")

#     # Report missing .pkl files
#     if missing_pkl_files:
#         print("Missing corresponding .pkl files:")
#         for pkl_file in missing_pkl_files:
#             print(pkl_file)
#     else:
#         print("All .jpg files have corresponding .pkl files.")

# if __name__ == "__main__":
#     main()
