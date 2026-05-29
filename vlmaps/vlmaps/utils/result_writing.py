import os
from pathlib import Path
import datetime
from wonderwords import RandomWord


def new_name() -> str:
    r = RandomWord()
    adj = r.word(include_parts_of_speech=["adjectives"])
    noun = r.word(include_parts_of_speech=["noun"])
    return f"{adj}_{noun}"


def createNamedExperiment(path: str, is_density: bool, encoder_name: str, verbose: bool = False):
    exists = True
    exp_name = ""
    suffix = "_density" if is_density else ""
    while exists:
        exp_name = new_name()
        path += f"/{exp_name}_{encoder_name}{suffix}"
        exists = Path(path).exists()

    if verbose:
        print(f"Starting experiment: {exp_name}")
        print(f"Path: {path}")
    return path


def createDirs(file):
    path = os.path.dirname(os.path.abspath(file))
    Path(path).mkdir(parents=True, exist_ok=True)


def safeOpen(file, mode):
    createDirs(file)
    return open(file, mode)


def writeClassificationHeader(file):
    f = safeOpen(file, "a")
    f.write(
        ("map;"
        "method;"
        "accuracy;"
        "macro_averaged_precision;"
        "micro_averaged_precision;"
        "macro_averaged_recall;"
        "micro_averaged_recall;"
        "macro_averaged_f1;"
        "micro_averaged_f1;"
        "macro_iou;"
        "micro_iou;"
        "macro_specificity;"
        "micro_specificity;"
        "macro_NPV;"
        "micro_NPV;"
        "macro_balanced_accuracy;"
        "micro_balanced_accuracy;"
        "prompt_engineering;"
        "postprocessing;"
        "median;"
        "2d;"
        "density;"
        "config;"
        "prompt;"
        "synonyms;"
        "antonyms;"
        "classifier;"
        "encoder;"
        "metadata;"
        "\n")
    )
    f.close()


def getFile(path, file, suffix, force=False, exist_ok=False):
    os.makedirs(path, exist_ok=True)
    fileName = path + "/" + file + "." + suffix
    file_exists = os.path.exists(fileName)
    if file_exists and not exist_ok:
        if not force:
            dt = datetime.datetime.now()
            dt_str = dt.strftime("_%d_%m_%Y_%H_%M_%S")
            fileName = path + "/" + file + dt_str + "." + suffix
        else:
            os.remove(fileName)
    return fileName


def addField(line, value):
    line += str(value) + ";"
    return line


def writeClassificationLine(file, cl, metadata = None):
    f = safeOpen(file, "a")
    for i, d in enumerate(cl):
        if metadata is not None:
            meta = metadata[i]
        else:
            meta = ["", "", "", "", "", "","", "", "", "", "", ""]
        if d:
            line = ""
            line = addField(line, d[0]) # map
            line = addField(line, d[1]) # method
            line = addField(line, d[2]) # accuracy
            line = addField(line, d[3]) # macro_averaged_precision
            line = addField(line, d[4]) # micro_averaged_precision
            line = addField(line, d[5]) # macro_averaged_recall
            line = addField(line, d[6]) # micro_averaged_recall
            line = addField(line, d[7]) # macro_averaged_f1
            line = addField(line, d[8]) # micro_averaged_f1
            line = addField(line, d[9]) # macro_iou
            line = addField(line, d[10]) # micro_iou
            line = addField(line, d[11]) # macro_specificity
            line = addField(line, d[12]) # micro_specificity
            line = addField(line, d[13]) # macro_NPV
            line = addField(line, d[14]) # micro_NPV
            line = addField(line, d[15]) # macro_balanced_accuracy
            line = addField(line, d[16]) # micro_balanced_accuracy

            if len(meta) > 0:
                line = addField(line, meta[0]) # prompt_engineering
            if len(meta) > 1:
                line = addField(line, meta[1]) # postprocessing
            if len(meta) > 2:
                line = addField(line, meta[2]) # median
            if len(meta) > 3:
                line = addField(line, meta[3]) # 2d
            if len(meta) > 4:
                line = addField(line, meta[4]) # density
            if len(meta) > 5:
                line = addField(line, meta[5]) # config
            if len(meta) > 6:
                line = addField(line, meta[6]) # prompts
            if len(meta) > 7:
                line = addField(line, meta[7]) # synonyms
            if len(meta) > 8:
                line = addField(line, meta[8]) # antonyms
            if len(meta) > 9:
                line = addField(line, meta[9]) # classifier
            if len(meta) > 10:
                line = addField(line, meta[10]) # encoder
            if len(meta) > 11:
                line = addField(line, meta[11]) # encoder

            line = line.rstrip(";")
            line += "\n"
            f.write(line)
    f.close()


def writeString(file, str, appendBreak=True):
    f = safeOpen(file, "a")
    if appendBreak:
        str += "\n"
    f.write(str)
    f.close()


def print_classification_line(cl):
    for i in range(len(cl)):
        d = cl[i]
        if d:
            print(f"map:                      {str(d[0])}")
            print(f"method:                   {str(d[1])}")
            print(f"accuracy:                 {str(d[2])}")
            # print(f"macro averaged precision: {str(d[3])}")
            # print(f"macro averaged recall:    {str(d[5])}")
            # print(f"macro averaged f1:        {str(d[7])}")
            # print(f"macro iou:                {str(d[9])}")
            print(f"micro averaged precision: {str(d[4])}")
            print(f"micro averaged recall:    {str(d[6])}")
            print(f"micro averaged f1:        {str(d[8])}")
            print(f"micro iou:                {str(d[10])}")
            # print(f"macro_specificity: {str(d[11])}")
            # print(f"micro_specificity: {str(d[12])}")
            # print(f"macro_NPV: {str(d[13])}")
            # print(f"micro_NPV: {str(d[14])}")
            # print(f"macro_balanced_accuracy: {str(d[15])}")
            # print(f"micro_balanced_accuracy: {str(d[16])}")
