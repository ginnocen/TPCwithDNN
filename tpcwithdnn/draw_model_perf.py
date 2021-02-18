# pylint: disable=too-many-locals, too-many-statements, fixme
import datetime
from ROOT import TFile, TCanvas, TLegend, TLatex, TPaveText # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kBlue, kGreen, kRed, kOrange # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond # pylint: disable=import-error, no-name-in-module
from ROOT import kOpenSquare, kOpenCircle, kOpenTriangleUp, kOpenDiamond # pylint: disable=import-error, no-name-in-module
from ROOT import kDarkBodyRadiator # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, gPad # pylint: disable=import-error, no-name-in-module

def setup_canvas(hist_name):
    canvas = TCanvas(hist_name, hist_name, 0, 0, 800, 800)
    canvas.SetMargin(0.13, 0.05, 0.12, 0.05)
    canvas.SetTicks(1, 1)

    leg = TLegend(0.36, 0.75, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.SetTextFont(42)
    leg.SetMargin(0.1)
    leg.SetHeader("Train setup: #it{N}_{ev}^{training}, #it{n}_{#it{#varphi}}" +\
                  " #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}", "C")

    return canvas, leg

def setup_hist(x_label, y_label, htemp):
    #htemp = gPad.GetPrimitive("th")
    htemp.GetXaxis().SetTitle(x_label)
    htemp.GetYaxis().SetTitle(y_label)
    htemp.GetXaxis().SetTitleOffset(1.1)
    htemp.GetYaxis().SetTitleOffset(1.5)
    htemp.GetXaxis().CenterTitle(True)
    htemp.GetYaxis().CenterTitle(True)
    htemp.GetXaxis().SetTitleSize(0.045)
    htemp.GetYaxis().SetTitleSize(0.045)
    htemp.GetXaxis().SetLabelSize(0.035)
    htemp.GetYaxis().SetLabelSize(0.035)

def add_alice_text():
    tex = TLatex(0.53, 0.57, "#scale[0.8]{ALICE work in progress}")
    tex.SetNDC()
    tex.SetTextFont(42)
    return tex

def add_cut_desc(cut):
    txt = TPaveText(0.65, 0.6, 0.91, 0.75, "NDC")
    # txt.SetFillColor(kWhite)
    txt.SetFillStyle(0)
    txt.SetBorderSize(0)
    txt.SetTextAlign(32) # middle,right
    # txt.SetTextFont(42) # helvetica
    txt.SetTextSize(0.03)
    for cut_var in cut:
        if cut_var == 'sector':
            txt.AddText("%s %d" % (cut_var, int(round(cut[cut_var][0]))))
        else:
            txt.AddText("%.2f < %s < %.2f" % (cut[cut_var][0], cut_var, cut[cut_var][1]))
    txt.AddText("20 epochs")
    return txt

def draw_model_perf():
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTextFont(42)
    gStyle.SetTitleFont(42)
    gStyle.SetLabelFont(42)
    gStyle.SetPalette(kDarkBodyRadiator)

    # pdf_dir = "trees/phi90_r17_z17_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1" \
    #           "_useSCFluc1_pred_doR1_dophi0_doz0"
    # nevs = [500, 1000, 2000, 5000]
    trees_dir = "/mnt/temp/mkabus/val-20201209/trees"
    suffix = "filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1" \
             "_dophi0_doz0/"
    pdf_dir_90 = "%s/phi90_r17_z17_%s" % (trees_dir, suffix)
    pdf_dir_180 = "%s/phi180_r33_z33_%s" % (trees_dir, suffix)

    filename = "model_perf_90-180"
    file_formats = ["png"] # "pdf" - lasts long

    nevs_90 = [10000, 18000] # 5000
    nevs_180 = [18000]
    nevs = nevs_90 + nevs_180
    pdf_files_90 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_90, nev) for nev in nevs_90]
    pdf_files_180 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_180, nev) for nev in nevs_180]
    pdf_file_names = pdf_files_90 + pdf_files_180

    grans = [90, 90, 180]

    colors = [kBlue, kOrange, kGreen, kRed]
    markers = [kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond]
    markers2 = [kOpenSquare, kOpenCircle, kOpenTriangleUp, kOpenDiamond]

    var_name = "flucDistRDiff"
    y_vars = ["rmsd", "means"]
    y_labels = ["#it{RMSE} (cm)", "Mean (cm)"]
    x_vars = ["rBinCenter", "fsector"]
    x_vars_short = ["r"] #, "fsector"]
    x_labels = ["#it{r} (cm)", "fsector"] # TODO: what units?

    # "r_rmsd": 33, 195.0, 245.5, 20, # 83.5, 254.5, 200,
    # "r_rmsd": "33, 83.5, 110, 200, 0.000, 0.06",
    hist_strs = {"r_rmsd": "33, 83.5, 245.5, 200, 0.000, 0.06",
            "fsector_rmsd": "90, -1.0, 19, 200, 0.00, 0.1",
            "r_means": "33, 83.5, 245.5, 200, -0.06, 0.06",
            "fsector_means": "90, -1.0, 19, 200, -0.07, 0.01",
            "r_rmsd_means": "33, 83.5, 245.5, 200, -0.06, 0.06"}

    # gran_desc = "#it{n}_{#it{#varphi}}#times#it{n}_{#it{r}}#times#it{n}_{#it{z}}"
    date = datetime.date.today().strftime("%Y%m%d")

    # flucDistR_entries>50
    # deltaSCBinCenter>0.0121 && deltaSCBinCenter<0.0122
    # deltaSCBinCenter>0.020 && deltaSCBinCenter<0.023
    # rBinCenter > 200.0 && deltaSCBinCenter>0.04 && deltaSCBinCenter<0.057
    cuts = {'r': {'#it{z}': (0.0, 5.0), 'sector': (9.00, 9.05), # '#it{r}': (0.0, 110.0),
                  '(#it{<#rho>}_{SC} - #it{#rho}_{SC})': (0.06, 0.07)},
            'fsector': {'#it{z}': (0.0, 5.0), '#it{r}': (86.0, 86.1),
                       '(#it{<#rho>}_{SC} - #it{#rho}_{SC})': (0.00, 0.05)}}
    cut_r = "zBinCenter > %.2f && zBinCenter < %.2f" % cuts['r']['#it{z}'] +\
            " && fsector > %.2f  && fsector < %.2f" % cuts['r']['sector'] +\
            " && deltaSCBinCenter > %.2f && deltaSCBinCenter < %.2f" %\
                cuts['r']['(#it{<#rho>}_{SC} - #it{#rho}_{SC})'] +\
            " && %s_rmsd > 0.0" % var_name
            # " && rBinCenter > %.2f && rBinCenter < %.2f" % cuts['r']['#it{r}'] +\
    cut_fsector = "zBinCenter > %.2f && zBinCenter < %.2f" % cuts['fsector']['#it{z}'] +\
                  " && rBinCenter > %.2f && rBinCenter < %.2f" % cuts['fsector']['#it{r}'] +\
                  " && deltaSCBinCenter > %.2f && deltaSCBinCenter < %.2f" %\
                      cuts['fsector']['(#it{<#rho>}_{SC} - #it{#rho}_{SC})'] +\
                  " && %s_rmsd > 0.0" % var_name
    cut_r = "zBinCenter > %.2f && zBinCenter < %.2f" % cuts['r']['#it{z}'] +\
            " && abs(phiBinCenter - 3.1) < 2.9" +\
            " && %s_rmsd > 0.0" % var_name +\
            " && deltaSCBinCenter > %.2f && deltaSCBinCenter < %.2f" %\
                cuts['r']['(#it{<#rho>}_{SC} - #it{#rho}_{SC})']
                  # " && rBinCenter > %.2f && rBinCenter < %.2f" % cuts['r']['#it{r}'] +\
                  # " && fsector > %.2f  && fsector < %.2f" % cuts['r']['sector'] +\
    cuts_list = [cut_r, cut_fsector]

    # for y_var, y_label in zip(y_vars, y_labels):
    y_label = "#it{RMSE} and #it{#mu} (cm)"
    canvas, leg = setup_canvas("perf_%s" % y_label)
    pdf_files = [TFile.Open(pdf_file_name, "read") for pdf_file_name in pdf_file_names]
    trees = [pdf_file.Get("pdfmaps") for pdf_file in pdf_files]
    variables = zip(x_vars, x_vars_short, x_labels, cuts_list)
    for x_var, x_var_short, x_label, cut in variables:
        hist_str = hist_strs["%s_%s_%s" % (x_var_short, y_vars[0], y_vars[1])]
        # pdf_files = [TFile.Open(pdf_file_name, "read") for pdf_file_name in pdf_file_names]
        # trees = [pdf_file.Get("pdfmaps") for pdf_file in pdf_files]
        hists = []
        styles = enumerate(zip(nevs, colors, markers, markers2, trees, grans))
        for ind, (nev, color, marker, marker2, tree, gran) in styles:
            tree.SetMarkerColor(color)
            tree.SetMarkerStyle(marker)
            tree.SetMarkerSize(2)
            #same_str = "" if ind == 0 else "same"
            same_str = "prof" if ind == 0 else "profsame"
            gran_str = "180#times33#times33" if gran == 180 else "90#times17#times17"
            hist_def = ">>th_%d(%s)" % (ind, hist_str) # if ind == 0 else "" 
            tree.Draw("%s_%s:%s%s" % (var_name, y_vars[0], x_var, hist_def), cut, same_str)
            hist = tree.GetHistogram()
            setup_hist(x_label, y_label, hist)
            hists.append(hist)
            print('hist type: {} hists last type: {}'.format(type(hist), type(hists[-1])))
            leg.AddEntry(tree, "%d, %s" % (nev, gran_str), "P")
            #gPad.Update()
            #tree.SetMarkerStyle(marker2)
            #tree.Draw("%s_%s:%s:%s" % (var_name, y_vars[1], x_var, prof_var), cut, "profsame")
            #leg.AddEntry(tree, "%d, %s" % (nev, gran_str), "P")
            # gPad.Update()

        #setup_frame(x_label, y_label, hist)
        print('hists last type: {}'.format(type(hists[-1])))
        for hist in hists:
            print('current hist type: {}'.format(type(hist)))
            hist.Draw()
        leg.Draw()
        tex = add_alice_text()
        tex.Draw()
        txt = add_cut_desc(cuts[x_var_short])
        txt.Draw()
        for ff in file_formats:
            canvas.SaveAs("%s_%s_%s_%s_%s.%s" % (date, filename, x_var_short, y_vars[0], y_vars[1], ff))
    for pdf_file in pdf_files:
        pdf_file.Close()

def main():
    draw_model_perf()

if __name__ == "__main__":
    main()
