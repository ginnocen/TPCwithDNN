"""
Plot the results of model prediction validation.
"""
# pylint: disable=too-many-locals, too-many-statements, fixme
import datetime
from ROOT import TFile, TCanvas, TLegend, TLatex, TPaveText # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kBlue, kGreen, kRed, kOrange, kWhite # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond # pylint: disable=import-error, no-name-in-module
from ROOT import kOpenSquare, kOpenCircle, kOpenTriangleUp, kOpenDiamond # pylint: disable=import-error, no-name-in-module
from ROOT import kDarkBodyRadiator # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT # pylint: disable=import-error, no-name-in-module

def setup_canvas(hist_name):
    canvas = TCanvas(hist_name, hist_name, 0, 0, 800, 800)
    canvas.SetMargin(0.13, 0.05, 0.12, 0.05)
    canvas.SetTicks(1, 1)

    leg = TLegend(0.3, 0.15, 0.75, 0.4)
    leg.SetNColumns(3)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.SetTextFont(42)
    leg.SetMargin(0.2)
    leg.SetHeader("Train setup: #it{N}_{ev}^{training}, #it{n}_{#it{#varphi}}" +\
                  " #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}", "C")

    return canvas, leg

def setup_hist(x_label, y_label, htemp):
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
    tex = TLatex(0.52, 0.75, "#scale[0.8]{ALICE work in progress}")
    tex.SetTextSize(0.04)
    tex.SetNDC()
    return tex

def add_cut_desc(cuts, x_var):
    txt = TPaveText(0.5, 0.75, 0.9, 0.89, "NDC")
    txt.SetFillColor(kWhite)
    txt.SetBorderSize(0)
    txt.SetTextAlign(12) # left, middle
    txt.SetTextSize(0.03)
    txt.AddText(cuts["deltaSC"]["desc"](cuts["deltaSC"]["%s_lim" % x_var]))
    txt.AddText("%s, 20 epochs" % cuts["z"]["desc"](cuts["z"]["%s_lim" % x_var]))
    #for cut_var in cuts:
        #if cut_var == "sector":
        #    txt.AddText("%s %d" % (cut_var, int(round(cut[cut_var][0]))))
        #if cut_var not in ("fsector", "phi", "r"):
        #    txt.AddText(cuts[cut_var]["desc"](cuts[cut_var]["%s_lim" % x_var]))
    #txt.AddText("20 epochs")
    return txt

def draw_model_perf():
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTextFont(42)
    gStyle.SetTitleFont(42)
    gStyle.SetLabelFont(42)
    gStyle.SetPalette(kDarkBodyRadiator)

    trees_dir = "/mnt/temp/mkabus/val-20201209/trees"
    suffix = "filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1" \
             "_dophi0_doz0/"
    pdf_dir_90 = "%s/phi90_r17_z17_%s" % (trees_dir, suffix)
    pdf_dir_180 = "%s/phi180_r33_z33_%s" % (trees_dir, suffix)

    filename = "model_perf_90-180"
    file_formats = ["png"] # "pdf" - lasts long

    nevs_90 = [10000, 18000] # 5000
    nevs_180 = [10000, 18000]
    nevs = nevs_90 + nevs_180
    pdf_files_90 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_90, nev) for nev in nevs_90]
    pdf_files_180 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_180, nev) for nev in nevs_180]
    pdf_file_names = pdf_files_90 + pdf_files_180

    grans = [90, 90, 180, 180]

    colors = [kBlue, kOrange, kGreen, kRed]
    markers_list = [(kFullSquare, kOpenSquare), (kFullCircle, kOpenCircle),
                    (kFullTriangleUp, kOpenTriangleUp), (kFullDiamond, kOpenDiamond)]

    var_name = "flucDistRDiff"
    y_vars = ["rmsd", "means"]
    y_vars_names = ["RMSE", "#mu"]
    # y_labels = ["#it{RMSE} (cm)", "Mean (cm)"]
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

    date = datetime.date.today().strftime("%Y%m%d")

    # flucDistR_entries>50
    # deltaSCBinCenter>0.0121 && deltaSCBinCenter<0.0122
    # deltaSCBinCenter>0.020 && deltaSCBinCenter<0.023
    # rBinCenter > 200.0 && deltaSCBinCenter>0.04 && deltaSCBinCenter<0.057
    # deltaSCBinCenter > 0.06 && deltaSCBinCenter < 0.07
    delta_sc_str = "#int_{#it{r}, #it{#varphi}, #it{z}} " +\
                   "#frac{#it{#rho}_{SC} - #it{<#rho>}_{SC}}{#it{<#rho>}_{SC}}"
    cuts = {"deltaSC": {"r_lim": (0.05, 0.07), "fsector_lim": (0.00, 0.05),
            "desc": lambda x: "%.2f < %s < %.2f" % (x[0], delta_sc_str, x[1])},
            "z": {"r_lim":(0.0, 5.0), "fsector_lim": (0.0, 5.0),
                  "desc": lambda x: "%.2f < #it{z} < %.2f cm" % x},
            "r": {"r_lim": (0.0, 110.0), "fsector_lim": (86.0, 86.1),
                  "desc": lambda x: "%.2f cm < #it{r} < %.2f cm" % x},
            "fsector": {"r_lim": (9.00, 9.05), "fsector_lim": (0.0, 20.0),
                        "desc": lambda x: "sector %d" % int(round(x[0]))},
            "phi": {"r_lim": (3.1, 2.9), "fsector_lim": (3.1, 2.9),
                    "desc": lambda x: "%.2f < #it{#varphi} < %.2f" % x}}
    cut_fsector = "zBinCenter > %.2f && zBinCenter < %.2f" % cuts["z"]["fsector_lim"] +\
                  " && rBinCenter > %.2f && rBinCenter < %.2f" % cuts["r"]["fsector_lim"] +\
                  " && deltaSCBinCenter > %.2f && deltaSCBinCenter < %.2f" %\
                      cuts["deltaSC"]["fsector_lim"] +\
                  " && %s_rmsd > 0.0" % var_name
    cut_r = "zBinCenter > %.2f && zBinCenter < %.2f" % cuts["z"]["r_lim"] +\
            " && abs(phiBinCenter - %.1f) < %.1f" % cuts["phi"]["r_lim"] +\
            " && deltaSCBinCenter > %.2f && deltaSCBinCenter < %.2f" %\
                cuts["deltaSC"]["r_lim"] +\
            " && %s_rmsd > 0.0" % var_name
    cuts_list = [cut_r, cut_fsector]

    # for y_var, y_label in zip(y_vars, y_labels):
    y_label = "#it{RMSE} and #it{#mu} (cm)"
    canvas, leg = setup_canvas("perf_%s" % y_label)
    pdf_files = [TFile.Open(pdf_file_name, "read") for pdf_file_name in pdf_file_names]
    trees = [pdf_file.Get("pdfmaps") for pdf_file in pdf_files]
    variables = zip(x_vars, x_vars_short, x_labels, cuts_list)
    for x_var, x_var_short, x_label, cut in variables:
        hist_str = hist_strs["%s_%s_%s" % (x_var_short, y_vars[0], y_vars[1])]
        hists = []
        styles = enumerate(zip(nevs, colors, markers_list, trees, grans))
        for ind, (nev, color, markers, tree, gran) in styles:
            tree.SetMarkerColor(color)
            tree.SetMarkerSize(2)
            gran_str = "180#times 33 #times 33" if gran == 180 else "90#times 17 #times 17"

            for y_ind, (y_var, y_var_name, marker) in enumerate(zip(y_vars, y_vars_names, markers)):
                tree.SetMarkerStyle(marker)
                hist_def = "th_%d_%d(%s)" % (ind, y_ind, hist_str)
                tree.Draw("%s_%s:%s>>%s" % (var_name, y_var, x_var, hist_def), cut, "prof")
                hists.append(tree.GetHistogram())
                leg.AddEntry(hists[-1], "%s" % y_var_name, "P")
            leg.AddEntry("", "%d, %s" % (nev, gran_str), "")

        for hist in hists:
            setup_hist(x_label, y_label, hist)
            hist.SetMaximum(0.06)
            hist.SetMinimum(-0.06)
            hist.Draw("same")
        leg.Draw()
        #tex = add_alice_text()
        #tex.Draw()
        txt = add_cut_desc(cuts, x_var_short)
        txt.Draw()
        for ff in file_formats:
            canvas.SaveAs("%s_%s_%s_%s_%s.%s" % (date, filename, x_var_short,
                                                 y_vars[0], y_vars[1], ff))
    for pdf_file in pdf_files:
        pdf_file.Close()

def main():
    draw_model_perf()

if __name__ == "__main__":
    main()
