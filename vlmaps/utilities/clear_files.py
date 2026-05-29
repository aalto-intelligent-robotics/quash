import argparse
import os

scenes = [
    "5LpN3gDmAk7_1",
    "gTV8FGcVJC9_1",
    "jh4fc5c5qoQ_1",
    "JmbYfDe2QKZ_1",
    "JmbYfDe2QKZ_2",
    "mJXqzFtmKg4_1",
    "ur6pFq6Qu1A_1",
    "UwV83HsGsw3_1",
    "Vt2qJdWjCF2_1",
    "YmJkqBEsHnH_1",
]
encoders = ["vlmaps_lseg", "vlmaps_openseg"]


def main(yes, dryrun):
    if not yes:
        inp = input("This will clear all created files, are you sure, y/n?\n")
        if inp != "y" and inp != "yes":
            exit(0)

    data_dir = os.environ["DATA_DIR"]
    vlmaps_dir = os.environ["VLMAPS_DIR"]
    mapsdir = data_dir + "/vlmaps_dataset/"
    parse_dir = vlmaps_dir + "/data/mapdata/"
    analysis_dir = vlmaps_dir + "/data/mapdata/analysis/"

    for scene in scenes:
        dir = mapsdir + scene + "/vlmap"
        for root, dirs, files in os.walk(dir):
            for file in files:
                f = root + "/" + file
                if dryrun:
                    print(f)
                else:
                    os.remove(f)

    for root, dirs, files in os.walk(analysis_dir):
        for file in files:
            f = root + file
            if dryrun:
                print(f)
            else:
                os.remove(f)

    for encoder in encoders:
        pdir = parse_dir + encoder
        for root, dirs, files in os.walk(pdir):
            for file in files:
                f = root + "/" + file
                if dryrun:
                    print(f)
                else:
                    os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./clear_files.py")
    parser.add_argument(
        "--yes", "-y", dest="yes", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--dryrun",
        "-d",
        dest="dryrun",
        action="store_true",
        required=False,
        default=False,
    )
    FLAGS, unparsed = parser.parse_known_args()
    yes = FLAGS.yes
    dryrun = FLAGS.dryrun
    main(yes, dryrun)
