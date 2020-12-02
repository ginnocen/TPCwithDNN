from ROOT import TFile, TCanvas, TLegend # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kBlue, kGreen, kRed, kOrange # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond # pylint: disable=import-error, no-name-in-module
from ROOT import kDarkBodyRadiator # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, gPad  # pylint: disable=import-error, no-name-in-module
# pylint: disable=fixme

def setup_canvas(hist_name):
    canvas = TCanvas(hist_name, hist_name, 0, 0, 800, 800)
    canvas.SetMargin(0.13, 0.05, 0.12, 0.05)
    canvas.SetTicks(1, 1)

    leg = TLegend(0.3, 0.65, 0.65, 0.8)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)

    return canvas, leg

def setup_frame(x_label, y_label):
    htemp = gPad.GetPrimitive("th")
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

def draw_model_perf():
    pdf_dir = "trees/phi90_r17_z17_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1" \
              "_useSCFluc1_pred_doR1_dophi0_doz0"
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    gStyle.SetPalette(kDarkBodyRadiator)

    filename = "py_model_perf"
    file_formats = ["png"] # "pdf" - lasts long

    nevs = [500, 1000, 2000, 5000]
    colors = [kBlue, kOrange, kGreen, kRed]
    markers = [kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond]

    # TODO: Why fsector from [9, 10]?
    cut_r = "zBinCenter>0 && zBinCenter<5 && flucDistR_entries>50 && fsector>9.00 && fsector<9.05" \
            " && deltaSCBinCenter>0.0121 && deltaSCBinCenter<0.0122"
    cut_fsector = "zBinCenter>0 && zBinCenter<5 && flucDistR_entries>50" \
                  " && rBinCenter>86.0 && rBinCenter<86.1" \
                  " && deltaSCBinCenter>0.0121 && deltaSCBinCenter<0.0122"
    cuts = [cut_r, cut_fsector]

    var_name = "flucDistRDiff"
    y_vars = ["rmsd", "means"]
    y_labels = ["RMSE", "Mean"] # TODO: what units?
    x_vars = ["rBinCenter", "fsector"]
    x_vars_short = ["r", "fsector"]
    x_labels = ["r (cm)", "fsector"] # TODO: what units?

    hist_strs = { "r_rmsd": "33, 83.5, 254.5, 200, 0.015, 0.145",
            "fsector_rmsd": "90, -1.0, 19, 200, 0.04, 0.27",
            "r_means": "33, 83.5, 254.5, 200, -0.13, 0.07",
            "fsector_means": "90, -1.0, 19, 200, -0.16, 0.06"}

    for y_var, y_label in zip(y_vars, y_labels):
        for x_var, x_var_short, x_label, cut in zip(x_vars, x_vars_short, x_labels, cuts):
            canvas, leg = setup_canvas("perf_%s" % y_label)
            hist_str = hist_strs["%s_%s" % (x_var_short, y_var)]
            for nev, color, marker in zip(nevs, colors, markers):
                pdf_file = TFile.Open("%s/pdfmaps_nEv%d.root" % (pdf_dir, nev), "read")
                tree = pdf_file.Get("pdfmaps")
                tree.SetMarkerColor(color)
                tree.SetMarkerStyle(marker)
                tree.SetMarkerSize(2)
                same_str = "" if nev == 500 else "same"
                tree.Draw("%s_%s:%s>>th(%s)" % (var_name, y_var, x_var, hist_str), cut, same_str)
                leg.AddEntry(tree, "N_{ev}^{training} = %d" % nev, "F")
                # leg.AddEntry(htemp, "N_{ev}^{training} = %d" % nev, "F")
                pdf_file.Close()

            setup_frame(x_label, y_label)
            # leg.Draw() # FIXME: It crashes :-(
            for file_format in file_formats:
                canvas.SaveAs("plots/%s_%s_%s.%s" % (filename, x_var_short, y_label, file_format))

def main():
    draw_model_perf()

if __name__ == "__main__":
    main()
