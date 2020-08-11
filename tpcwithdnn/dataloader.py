# pylint: disable=fixme, pointless-string-statement
import numpy as np

def loaddata_original(inputdata, indexev):

    vecRPosFile = inputdata + "data/Pos/" + str(0) + '-vecRPos.npy'
    vecPhiPosFile = inputdata + "data/Pos/" + str(0) + '-vecPhiPos.npy'
    vecZPosFile = inputdata + "data/Pos/" + str(0) + '-vecZPos.npy'
    scMeanFile = inputdata + "data/Mean/"+ str(indexev[1]) + '-vecMeanSC.npy'
    scRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomSC.npy'
    distRMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistR.npy'
    distRRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistR.npy'
    distRPhiMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistRPhi.npy'
    distRPhiRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistRPhi.npy'
    distZMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistZ.npy'
    distZRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistZ.npy'

    vecRPos = np.load(vecRPosFile)
    vecPhiPos = np.load(vecPhiPosFile)
    vecZPos = np.load(vecZPosFile)
    vecMeanSC = np.load(scMeanFile)
    vecRandomSC = np.load(scRandomFile)
    vecMeanDistR = np.load(distRMeanFile)
    vecRandomDistR = np.load(distRRandomFile)
    vecMeanDistRPhi = np.load(distRPhiMeanFile)
    vecRandomDistRPhi = np.load(distRPhiRandomFile)
    vecMeanDistZ = np.load(distZMeanFile)
    vecRandomDistZ = np.load(distZRandomFile)

    return [vecRPos, vecPhiPos, vecZPos,
            vecMeanSC, vecRandomSC,
            vecMeanDistR, vecRandomDistR,
            vecMeanDistRPhi, vecRandomDistRPhi,
            vecMeanDistZ, vecRandomDistZ]


def loaddata_derivativesRefMean(inputdata, selopt, opt_pred):
    """
    Here below we define the preselections on the input data for the training.
    Three options are currently implemented.
    selopt == 0 selects only points with positive z position
    selopt == 1 selects only points with negative z position
    selopt == 2 uses all data with no selections

    Selection of distortion variable to be returned (multi-dimensional output not yet implemented:
    opt_pred[0] == 1: dr
    opt_pred[1] == 1: drphi
    opt_pred[2] == 1: dz

    """
    zPosFile = inputdata + "data/Pos/" + str(0) + '-vecZPos.npy'
    refMeanSCPlusFile = inputdata + "data/Mean/" + str(9) + '-vecMeanSC.npy'
    refMeanSCMinusFile = inputdata + "data/Mean/" + str(18) + '-vecMeanSC.npy'

    if selopt == 0:
        arrSelectionZ = np.load(zPosFile) > 0
    elif selopt == 1:
        arrSelectionZ = np.load(zPosFile) < 0
    elif selopt == 2:
        arrSelectionZ = np.load(zPosFile) == np.load(zPosFile)

    arrDerRefMeanSC = np.load(refMeanSCPlusFile)[arrSelectionZ] - np.load(refMeanSCMinusFile)[arrSelectionZ]

    dim_distortions = sum(opt_pred)
    arraysize = arrDerRefMeanSC.size
    matDerRefMeanDist = np.empty((dim_distortions, arraysize))
    indexfillDist = 0
    if opt_pred[0] == 1:
        refMeanDistRPlusFile = inputdata + "data/Mean/" + str(9) + '-vecMeanDistR.npy'
        refMeanDistRMinusFile = inputdata + "data/Mean/" + str(18) + '-vecMeanDistR.npy'
        matDerRefMeanDist[indexfillDist, :] = np.load(refMeanDistRPlusFile)[arrSelectionZ] - np.load(refMeanDistRMinusFile)[arrSelectionZ]
        indexfillDist = indexfillDist + 1
    if opt_pred[1] == 1:
        refMeanDistRPhiPlusFile = inputdata + "data/Mean/" + str(9) + '-vecMeanDistRPhi.npy'
        refMeanDistRPhiMinusFile = inputdata + "data/Mean/" + str(18) + '-vecMeanDistRPhi.npy'
        matDerRefMeanDist[indexfillDist, :] = np.load(refMeanDistRPhiPlusFile)[arrSelectionZ] - np.load(refMeanDistRPhiMinusFile)[arrSelectionZ]
        indexfillDist = indexfillDist + 1
    if opt_pred[2] == 1:
        refMeanDistZPlusFile = inputdata + "data/Mean/" + str(9) + '-vecMeanDistZ.npy'
        refMeanDistZMinusFile = inputdata + "data/Mean/" + str(18) + '-vecMeanDistZ.npy'
        matDerRefMeanDist[indexfillDist, :] = np.load(refMeanDistZPlusFile)[arrSelectionZ] - np.load(refMeanDistZMinusFile)[arrSelectionZ]

    return arrDerRefMeanSC, matDerRefMeanDist


def loaddata(inputdata, indexev, selopt_input, selopt_output):

    """ Here we define the functionalties to load the files from the input
    directory which is set in the database. Here below the description of
    the input files:
        - 0-vecZPos.npy, 0-vecRPos.npy, 0-vecPhiPos.npy contains the
        position of the FIXME. There is only one of these files for each
        folder, therefore for each bunch of events
        Input features for training:
        - vecMeanSC.npy: average space charge in each bin of r, rphi and z.
        - vecRandomSC.npy: fluctuation of the space charge.
        Output from the numberical calculations:
        - vecMeanDistR.npy average distorsion along the R axis in the same
          grid. It represents the expected distorsion that an electron
          passing by that region would have as a consequence of the IBF.
        - vecRandomDistR.npy are the correponding fluctuations.
        - All the distorsions along the other directions have a consistent
          naming choice.

    """

    [_, _, vecZPos,
     vecMeanSC, vecRandomSC,
     vecMeanDistR, vecRandomDistR,
     vecMeanDistRPhi, vecRandomDistRPhi,
     vecMeanDistZ, vecRandomDistZ] = loaddata_original(inputdata, indexev)

    """
    Here below we define the preselections on the input data for the training.
    Three options are currently implemented.
    selopt_input == 0 selects only clusters with positive z position
    selopt_input == 1 selects only clusters with negative z position
    selopt_input == 2 uses all data with no selections

    """
    if selopt_input == 0:
        vecMeanSC_ = vecMeanSC[vecZPos >= 0]
        vecFluctuationSC_ = vecMeanSC[vecZPos >= 0] - vecRandomSC[vecZPos >= 0]
    elif selopt_input == 1:
        vecMeanSC_ = vecMeanSC[vecZPos < 0]
        vecFluctuationSC_ = vecMeanSC[vecZPos < 0] - vecRandomSC[vecZPos < 0]
    elif selopt_input == 2:
        vecMeanSC_ = vecMeanSC
        vecFluctuationSC_ = vecMeanSC  - vecRandomSC

    """
    selopt_output == 0 selects only clusters with positive z position
    selopt_output == 1 selects only clusters with negative z position
    selopt_output == 2 uses all data with no selections

    """
    if selopt_output == 0:
        vecFluctuationDistR_ = \
                vecMeanDistR[vecZPos >= 0] - vecRandomDistR[vecZPos >= 0]
        vecFluctuationDistRPhi_ = \
                vecMeanDistRPhi[vecZPos >= 0] - vecRandomDistRPhi[vecZPos >= 0]
        vecFluctuationDistZ_ = \
                vecMeanDistZ[vecZPos >= 0] - vecRandomDistZ[vecZPos >= 0]
    elif selopt_output == 1:
        vecFluctuationDistR_ = \
                vecMeanDistR[vecZPos < 0] - vecRandomDistR[vecZPos < 0]
        vecFluctuationDistRPhi_ = \
                vecMeanDistRPhi[vecZPos < 0] - vecRandomDistRPhi[vecZPos < 0]
        vecFluctuationDistZ_ = \
                vecMeanDistZ[vecZPos < 0] - vecRandomDistZ[vecZPos < 0]
    elif selopt_output == 2:
        vecFluctuationDistR_ = vecMeanDistR - vecRandomDistR
        vecFluctuationDistRPhi_ = vecMeanDistRPhi - vecRandomDistRPhi
        vecFluctuationDistZ_ = vecMeanDistZ - vecRandomDistZ

    return [vecMeanSC_, vecFluctuationSC_, vecFluctuationDistR_,
            vecFluctuationDistRPhi_, vecFluctuationDistZ_]


def loadtrain_test(inputdata, indexev, selopt_input, selopt_output,
                   grid_r, grid_rphi, grid_z, opt_train, opt_pred):

    [vecMeanSC, vecFluctuationSC, vecFluctuationDistR,
     vecFluctuationDistRPhi, vecFluctuationDistZ] = \
        loaddata(inputdata, indexev, selopt_input, selopt_output)
    dim_input = sum(opt_train)
    dim_output = sum(opt_pred)
    x_ = np.empty((grid_rphi, grid_r, grid_z, dim_input))
    y_ = np.empty((grid_rphi, grid_r, grid_z, dim_output))

    indexfillx = 0
    if opt_train[0] == 1:
        x_[:, :, :, indexfillx] = \
                vecMeanSC.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1
    if opt_train[1] == 1:
        x_[:, :, :, indexfillx] = \
                vecFluctuationSC.reshape(grid_rphi, grid_r, grid_z)
        indexfillx = indexfillx + 1

    if sum(opt_pred) > 1:
        print("MULTI-OUTPUT NOT IMPLEMENTED YET")
        return 0
    indexfilly = 0
    if opt_pred[0] == 1:
        y_[:, :, :, indexfilly] = \
                vecFluctuationDistR.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    if opt_pred[1] == 1:
        y_[:, :, :, indexfilly] = \
                vecFluctuationDistRPhi.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    if opt_pred[2] == 1:
        y_[:, :, :, indexfilly] = \
                vecFluctuationDistZ.reshape(grid_rphi, grid_r, grid_z)
        indexfilly = indexfilly + 1
    #print("DIMENSION INPUT TRAINING", x_.shape)
    #print("DIMENSION OUTPUT TRAINING", y_.shape)

    return x_, y_


def loaddata_applyND(inputdata, indexev, selopt, opt_pred):

    """
    Here below we define the preselections on the input data for the training.
    Three options are currently implemented.
    selopt == 0 selects only points with positive z position
    selopt == 1 selects only points with negative z position
    selopt == 2 uses all data with no selections

    Selection of distortion variable to be returned (multi-dimensional output not yet implemented:
    opt_pred[0] == 1: dr
    opt_pred[1] == 1: drphi
    opt_pred[2] == 1: dz

    """
    arrRPosFile = inputdata + "data/Pos/" + str(0) + '-vecRPos.npy'
    arrPhiPosFile = inputdata + "data/Pos/" + str(0) + '-vecPhiPos.npy'
    arrZPosFile = inputdata + "data/Pos/" + str(0) + '-vecZPos.npy'
    scMeanFile = inputdata + "data/Mean/"+ str(indexev[1]) + '-vecMeanSC.npy'
    scRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomSC.npy'

    if selopt == 0:
        arrSelectionZ = np.load(arrZPosFile) > 0
    elif selopt == 1:
        arrSelectionZ = np.load(arrZPosFile) < 0
    elif selopt == 2:
        arrSelectionZ = np.load(arrZPosFile) == np.load(arrZPosFile)

    arrRPos = np.load(arrRPosFile)[arrSelectionZ]
    arrPhiPos = np.load(arrPhiPosFile)[arrSelectionZ]
    arrZPos = np.load(arrZPosFile)[arrSelectionZ]

    arrMeanSC = np.load(scMeanFile)[arrSelectionZ]
    arrFluctuationSC = arrMeanSC - np.load(scRandomFile)[arrSelectionZ]

    dim_distortions = sum(opt_pred)
    arraysize = arrMeanSC.size
    matMeanDist = np.empty((dim_distortions, arraysize))
    matFluctuationDist = np.empty((dim_distortions, arraysize))
    indexfillDist = 0
    if opt_pred[0] == 1:
        distRMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistR.npy'
        distRRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistR.npy'
        matMeanDist[indexfillDist, :] = np.load(distRMeanFile)[arrSelectionZ]
        matFluctuationDist[indexfillDist, :] = matMeanDist[indexfillDist, :] - np.load(distRRandomFile)[arrSelectionZ]
        indexfillDist = indexfillDist + 1
    if opt_pred[1] == 1:
        distRPhiMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistRPhi.npy'
        distRPhiRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistRPhi.npy'
        matMeanDist[indexfillDist, :] = np.load(distRPhiMeanFile)[arrSelectionZ]
        matFluctuationDist[indexfillDist, :] = matMeanDist[indexfillDist, :] - np.load(distRPhiRandomFile)[arrSelectionZ]
        indexfillDist = indexfillDist + 1
    if opt_pred[2] == 1:
        distZMeanFile = inputdata + "data/Mean/" + str(indexev[1]) + '-vecMeanDistZ.npy'
        distZRandomFile = inputdata + "data/Random/" + str(indexev[0]) + '-vecRandomDistZ.npy'
        matMeanDist[indexfillDist, :] = np.load(distZMeanFile)[arrSelectionZ]
        matFluctuationDist[indexfillDist, :] = matMeanDist[indexfillDist, :] - np.load(distZRandomFile)[arrSelectionZ]


    return  arrRPos, arrPhiPos, arrZPos, \
            arrMeanSC, arrFluctuationSC, \
            matMeanDist, matFluctuationDist
