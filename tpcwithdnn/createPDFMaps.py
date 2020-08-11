import pickle
import gzip
import sys
import subprocess
import math
import pandas as pd
from root_pandas import to_root, read_root
from RootInteractive.Tools.makePDFMaps import makePdfMaps


def makePDFMapFromFile(inputFileStr):
    """
    Create a pdf map from histogram in inputfile and store it in a tree.
    :param inputFileStr: input file string to file containing nd histogram
    """
    outputFileStr = inputFileStr.replace("NDHistos", "PDFMaps").replace(".gzip", ".root")
    print("Input file: " + inputFileStr)
    print("Output file: " + outputFileStr)


    with gzip.open(inputFileStr, 'rb') as inputFile:
        histo = pickle.load(inputFile)

    dimI = 0
    slices = ((0, histo['H'].shape[0], 1, 0),
              (0, histo['H'].shape[1], 1, 0),
              (0, histo['H'].shape[2], 1, 0),
              (0, histo['H'].shape[3], 1, 0),
              (0, histo['H'].shape[4], 1, 0))
    print("Create pdf maps for " + histo['name'] + " (startBin = " + str(slices[0][0]) + ", stopBin = " + str(slices[0][1]) + ", step = " + str(slices[0][2]) + ", grouping = " + str(slices[0][3]) + ")")
    dfPDFMap = makePdfMaps(histo, slices, dimI)
    # set the index name to retrieve the name of the variable of interest later
    dfPDFMap.index.name = histo['name']
    dfPDFMap.to_root(outputFileStr, key=histo['name'], mode='w', store_index=True)


def createPDFMapsFromList(inputList):
    print("List of input files: " + inputList)
    inputFileList = subprocess.run("cat " + inputList, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8").split()
    for file in inputFileList:
        makePDFMapFromFile(file)


def mergePDFMaps(mapDirectory = "."):
    """
    Merges pdf maps for mean maps 1.0, 0.9 and 1.1
    """
    dfmerged = pd.DataFrame()
    for factor in [1.0, 1.1, 0.9]:
        print("Merging pdf maps for mean: %.1f" % factor)
        inputFiles = subprocess.run("ls " + mapDirectory + "/outputPDFMaps_mean%.1f_*.root" % factor, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8").split()

        df = read_root(inputFiles[0], columns="*Bin*")
        df['fsector'] = df['phiBinCenter'] / math.pi * 9
        df['meanMap'] = factor
        for file in inputFiles:
            dftemp = read_root(file, ignore="*Bin*")
            variable = dftemp.index.name
            for col in list(dftemp.keys()):
                df[variable + '_' + col] = dftemp[col]
        dfmerged = dfmerged.append(df, ignore_index = True)

    outputFile = mapDirectory + "/outputPDFMaps.root"
    print("Writing merged pdf maps (means 0.9, 1.0 and 1.1) to file: " + outputFile)
    dfmerged.to_root(outputFile, key='pdfmaps', mode='w', store_index=False)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 createPDFMaps.py <action> = [create, createFromList, merge] [<inputFile>]")
        sys.exit()
    action = sys.argv[1]
    print("Action: " + action)

    if action == 'create':
        if len(sys.argv) < 3:
            print("Usage: python3 createPDFMaps.py create <inputFile>")
            sys.exit()
        inputFile = sys.argv[2]
        makePDFMapFromFile(inputFile)
    elif action == 'createFromList':
        if len(sys.argv) < 3:
            print("Usage: python3 createPDFMaps.py createFromList <inputFile>")
            sys.exit()
        inputFile = sys.argv[2]
        createPDFMapsFromList(inputFile)
    elif action == 'merge':
        mergePDFMaps()
    else:
        print("Usage: python3 createPDFMaps.py <action> = [create, createFromList, merge] <inputFile>")
        sys.exit()
