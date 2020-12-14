# pylint: disable=too-many-locals
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

    leg = TLegend(0.5, 0.75, 0.8, 0.9)
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
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
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

    nevs_90 = [5000, 10000, 18000]
    nevs_180 = [18000]
    nevs = nevs_90 + nevs_180
    pdf_files_90 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_90, nev) for nev in nevs_90]
    pdf_files_180 = ["%s/pdfmaps_nEv%d.root" % (pdf_dir_180, nev) for nev in nevs_180]
    pdf_file_names = pdf_files_90 + pdf_files_180

    grans = [90, 90, 90, 180]

    colors = [kBlue, kOrange, kGreen, kRed]
    markers = [kFullSquare, kFullCircle, kFullTriangleUp, kFullDiamond]

    # flucDistR_entries>50
    # deltaSCBinCenter>0.0121 && deltaSCBinCenter<0.0122
    # deltaSCBinCenter>0.020 && deltaSCBinCenter<0.023
    cut_r = "zBinCenter>0 && zBinCenter<5 && fsector>9.00 && fsector<9.05" \
           " && rBinCenter > 200.0 && deltaSCBinCenter>0.04 && deltaSCBinCenter<0.057"
    cut_fsector = "zBinCenter>0 && zBinCenter<5" \
                  " && rBinCenter>86.0 && rBinCenter<86.1" \
                  " && deltaSCBinCenter>0.00 && deltaSCBinCenter<0.05"
    cuts = [cut_r, cut_fsector]

    var_name = "flucDistRDiff"
    y_vars = ["rmsd", "means"]
    y_labels = ["RMSE", "Mean"] # TODO: what units?
    x_vars = ["rBinCenter", "fsector"]
    x_vars_short = ["r", "fsector"]
    x_labels = ["r (cm)", "fsector"] # TODO: what units?

    hist_strs = { "r_rmsd": "33, 195.0, 245.5, 20, 0.000, 0.06", # 83.5 254.5, 200
            "fsector_rmsd": "90, -1.0, 19, 200, 0.00, 0.1",
            "r_means": "33, 195.0, 245.5, 20, -0.06, 0.06",
            "fsector_means": "90, -1.0, 19, 200, -0.07, 0.01"}

    for y_var, y_label in zip(y_vars, y_labels):
        for x_var, x_var_short, x_label, cut in zip(x_vars, x_vars_short, x_labels, cuts):
            canvas, leg = setup_canvas("perf_%s" % y_label)
            hist_str = hist_strs["%s_%s" % (x_var_short, y_var)]
            pdf_files = [TFile.Open(pdf_file_name, "read") for pdf_file_name in pdf_file_names]
            trees = [pdf_file.Get("pdfmaps") for pdf_file in pdf_files]
            styles = enumerate(zip(nevs, colors, markers, trees, grans))
            for ind, (nev, color, marker, tree, gran) in styles:
                tree.SetMarkerColor(color)
                tree.SetMarkerStyle(marker)
                tree.SetMarkerSize(2)
                same_str = "" if ind == 0 else "same"
                gran_str = "180x33x33" if gran == 180 else "90x17x17"
                tree.Draw("%s_%s:%s>>th(%s)" % (var_name, y_var, x_var, hist_str), cut, same_str)
                leg.AddEntry(tree, "N_{ev}^{training} = %d, %s" % (nev, gran_str), "P")

            setup_frame(x_label, y_label)
            leg.Draw()
            for ff in file_formats:
                canvas.SaveAs("plots/20201210_%s_%s_%s.%s" % (filename, x_var_short, y_label, ff))
            for pdf_file in pdf_files:
                pdf_file.Close()

def main():
    draw_model_perf()

if __name__ == "__main__":
    main()
