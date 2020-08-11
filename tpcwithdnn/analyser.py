import math

import numpy as np
import pandas as pd
from keras.models import model_from_json
from symmetrypadding3d import symmetryPadding3d
from dataloader import loaddata_applyND
from RootInteractive.InteractiveDrawing.bokeh.bokehDrawSA import *
from root_pandas import to_root, read_root
from ROOT import TH1F, TH2F, TLine, TLegend, TLatex, TFile, TCanvas, \
    gPad  # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, TTree, TObjArray  # pylint: disable=import-error, no-name-in-module


def getUnitChargeResponse(nphi, nr, nz, model_files, input_dir, input_sc_fluc, input_sc_mean, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'symmetryPadding3d': symmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], output_pred[1], output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for distName in dist_names:
        column_names = np.append(column_names, ['flucDist' + distName + 'Pred'])

    # load position arrays
    arrRPos, arrPhiPos, arrZPos, \
    _, _, \
    _, _ = \
        loaddata_applyND(input_dir, [0,0], 0, output_pred)

    # define input densities, create data frame and fill with prediction
    arrFluctuationSC = np.full(arrRPos.size, input_sc_fluc)
    arrMeanSC = np.full(arrRPos.size, input_sc_mean)

    df = pd.DataFrame({column_names[0]: arrPhiPos,
                        column_names[1]: arrRPos,
                        column_names[2]: arrZPos,
                        column_names[3]: arrFluctuationSC,
                        column_names[4]: arrMeanSC})

    input_single = np.empty((1, nphi, nr, nz, 2))
    input_single[0, :, :, :, 0] = arrMeanSC.reshape(nphi, nr, nz)
    input_single[0, :, :, :, 1] = arrFluctuationSC.reshape(nphi, nr, nz)

    matFluctuationDistPredict_group = loaded_model.predict(input_single)
    matFluctuationDistPredict = np.empty((sum(output_pred), arrRPos.size))
    for indexDist in range(sum(output_pred)):
        matFluctuationDistPredict[indexDist, :] = matFluctuationDistPredict_group[0, :, :, :, indexDist].flatten()
        df[column_names[5 + indexDist]] = matFluctuationDistPredict[indexDist, :]

    return df


def getLineChargeResponse(nphi, nr, nz, model_files, input_dir, iphi, ir, input_sc_fluc, input_sc_mean, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'symmetryPadding3d': symmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], output_pred[1], output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for distName in dist_names:
        column_names = np.append(column_names, ['flucDist' + distName + 'Pred'])

    # load position arrays
    arrRPos, arrPhiPos, arrZPos, \
    _, _, \
    _, _ = \
        loaddata_applyND(input_dir, [0,0], 0, output_pred)

    # define input densities, create data frame and fill with prediction
    input_single = np.zeros((1, nphi, nr, nz, 2))
    input_single[0, iphi, ir, :, 0] = input_sc_mean
    input_single[0, iphi, ir, :, 1] = input_sc_fluc
    arrMeanSC = input_single[0, :, :, :, 0].flatten()
    arrFluctuationSC = input_single[0, :, :, :, 1].flatten()

    df = pd.DataFrame({column_names[0]: arrPhiPos,
                        column_names[1]: arrRPos,
                        column_names[2]: arrZPos,
                        column_names[3]: arrFluctuationSC,
                        column_names[4]: arrMeanSC})

    matFluctuationDistPredict_group = loaded_model.predict(input_single)
    matFluctuationDistPredict = np.empty((sum(output_pred), arrRPos.size))
    for indexDist in range(sum(output_pred)):
        matFluctuationDistPredict[indexDist, :] = matFluctuationDistPredict_group[0, :, :, :, indexDist].flatten()
        df[column_names[5 + indexDist]] = matFluctuationDistPredict[indexDist, :]

    return df


def getPointChargeResponse(nphi, nr, nz, model_files, input_dir, iphi, ir, iz, input_sc_fluc, input_sc_mean, output_pred):
    # load model
    json_file = open("%s.json" % model_files, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = \
        model_from_json(loaded_model_json, {'symmetryPadding3d': symmetryPadding3d})
    loaded_model.load_weights("%s.h5" % model_files)

    # define data frame column names
    dist_names = np.array(['R', 'Rphi', 'Z'])[np.array([output_pred[0], output_pred[1], output_pred[2]]) > 0]
    column_names = np.array(['phi', 'r', 'z',
                             'flucSC', 'meanSC'])
    for distName in dist_names:
        column_names = np.append(column_names, ['flucDist' + distName + 'Pred'])

    # load position arrays
    arrRPos, arrPhiPos, arrZPos, \
    _, _, \
    _, _ = \
        loaddata_applyND(input_dir, [0,0], 0, output_pred)

    # define input densities, create data frame and fill with prediction
    input_single = np.zeros((1, nphi, nr, nz, 2))
    input_single[0, iphi, ir, iz, 0] = input_sc_mean
    input_single[0, iphi, ir, iz, 1] = input_sc_fluc
    arrMeanSC = input_single[0, :, :, :, 0].flatten()
    arrFluctuationSC = input_single[0, :, :, :, 1].flatten()

    df = pd.DataFrame({column_names[0]: arrPhiPos,
                        column_names[1]: arrRPos,
                        column_names[2]: arrZPos,
                        column_names[3]: arrFluctuationSC,
                        column_names[4]: arrMeanSC})

    matFluctuationDistPredict_group = loaded_model.predict(input_single)
    matFluctuationDistPredict = np.empty((sum(output_pred), arrRPos.size))
    for indexDist in range(sum(output_pred)):
        matFluctuationDistPredict[indexDist, :] = matFluctuationDistPredict_group[0, :, :, :, indexDist].flatten()
        df[column_names[5 + indexDist]] = matFluctuationDistPredict[indexDist, :]

    return df


def makePlotsInteractive():
    df = read_root("outputPDFMaps.root")
    df['sector'] = df['phiBinCenter'] / math.pi * 9

    tooltips = [("r", "(@rBinCenter)"), ("phi", "(@phiBinCenter)"), ("sec", "(@sector)"), ("z", "(@zBinCenter)"), ("deltaSC", "(@deltaSCBinCenter)"), ("meanMap", "(@meanMap)")]
    figureLayout: str = '((0),(1),x_visible=1,y_visible=1,plot_height=200,plot_width=1200)'

    # diff vs phi and deltaSC, z < 5
    figureArray = [
        [['sector'], ['flucDistRDiff_means'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
        [['sector'], ['flucDistRDiff_rmsd'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
    ]
    widgetParams = [
        ['range', ['sector']],
        ['range', ['rBinCenter']],
        ['range', ['deltaSCBinCenter']],
        ['range', ['meanMap']]
    ]
    widgetLayout = [[0, 1], [2, 3]]
    output_file("figures/figDistRDiff_phi-deltaSC.html")
    bokehDrawSA.fromArray(df, "flucDistRDiff_meansOK==1 & flucDistRDiff_entries>50 & zBinCenter<5", figureArray, widgetParams, layout=figureLayout, tooltips=tooltips, widgetLayout=widgetLayout)

    # diff vs r and deltaSC, z < 5, sector 9
    figureArray = [
        [['rBinCenter'], ['flucDistRDiff_means'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
        [['rBinCenter'], ['flucDistRDiff_rmsd'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
    ]
    widgets = 'slider.phiBinNumber(90,100,1,95,95),slider.deltaSCBinCenter(-0.2,0.2,0.01,-0.1,0.1),slider.meanMap(0.9,1.1,0.1,1.0,1.0)'
    output_file("figures/figDistRDiff_r-deltaSC.html")
    bokehDrawSA.fromArray(df, "flucDistRDiff_meansOK==1 & flucDistRDiff_entries>50 & zBinCenter<5 & phiBinNumber>89 & phiBinNumber<101", figureArray, widgets, layout=figureLayout, tooltips=tooltips, legend=False)

    # diff vs z and deltaSC, sector 9
    figureArray = [
        [['zBinCenter'], ['flucDistRDiff_means'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
        [['zBinCenter'], ['flucDistRDiff_rmsd'], {'color': "red", "size": 4, "colorZvar": "deltaSCBinCenter"}],
    ]
    widgets = 'slider.phiBinNumber(90,100,1,95,95),slider.deltaSCBinCenter(-0.2,0.2,0.01,-0.1,0.1),slider.meanMap(0.9,1.1,0.1,1.0,1.0)'
    output_file("figures/figDistRDiff_z-deltaSC.html")
    bokehDrawSA.fromArray(df, "flucDistRDiff_meansOK==1 & flucDistRDiff_entries>50 & phiBinNumber>89 & phiBinNumber<101 & rBinCenter<90", figureArray, widgets, layout=figureLayout, tooltips=tooltips, legend=False)


if __name__ == '__main__':
    makePlotsInteractive()
