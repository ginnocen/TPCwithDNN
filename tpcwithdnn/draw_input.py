# pylint: disable=too-many-statements
from ROOT import TFile, TCanvas, TH1F # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT,  gPad  # pylint: disable=import-error, no-name-in-module

def setup_frame(x_label, y_label, z_label):
    htemp = gPad.GetPrimitive("htemp")
    htemp.GetXaxis().SetTitle(x_label)
    htemp.GetYaxis().SetTitle(y_label)
    htemp.GetZaxis().SetTitle(z_label)
    htemp.GetXaxis().SetTitleOffset(1.0)
    htemp.GetYaxis().SetTitleOffset(1.0)
    htemp.GetZaxis().SetTitleOffset(1.0)
    htemp.GetXaxis().CenterTitle(True)
    htemp.GetYaxis().CenterTitle(True)
    htemp.GetZaxis().CenterTitle(True)
    htemp.GetXaxis().SetTitleSize(0.035)
    htemp.GetYaxis().SetTitleSize(0.035)
    htemp.GetZaxis().SetTitleSize(0.035)
    htemp.GetXaxis().SetLabelSize(0.035)
    htemp.GetYaxis().SetLabelSize(0.035)
    htemp.GetZaxis().SetLabelSize(0.035)

def set_margins(canvas):
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.1)
    canvas.SetTopMargin(0.03)
    canvas.SetBottomMargin(0.1)

def set_histogram(hist):
    hist.SetMarkerStyle(20)
    hist.SetMinimum(0)
    hist.SetMaximum(0.6e-6)

def draw_one_idc(vec_mean_one_idc, vec_fluc_one_idc, canvas):
    h_mean = TH1F("mean", "mean 1D IDC", 201, -0.5, 200.5)
    h_fluc = TH1F("fluc", "fluc 1D IDC", 201, -0.5, 200.5)
    for i in range(0, len(vec_mean_one_idc)):
        h_mean.SetBinContent(i + 1, vec_mean_one_idc)
        h_fluc.SetBinContent(i + 1, vec_fluc_one_idc)

    set_histogram(h_mean)
    h_mean.Draw("P")
    set_margins(canvas)
    canvas.SaveAs("mean_1D_IDC_hist.png")

    set_histogram(h_fluc)
    h_fluc.Draw("P")
    set_margins(canvas)
    canvas.SaveAs("fluc_1D_IDC_hist.png")

def draw_input(is_idc):
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    f = TFile.Open("/mnt/temp/mkabus/idc-study-20210310/" +\
                   "trees/treeInput_mean1.00_phi90_r17_z17.root","READ")
    t = f.Get("validation")

    t.SetMarkerStyle(kFullSquare)

    c1 = TCanvas()

    t.Draw("z:r:meanSC", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean SC (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("z_r_meanSC_colz_phi_sector0.png")

    t.Draw("meanSC:r:phi", "z>0 && z<1", "profcolz")
    setup_frame("#varphi (rad)", "r (cm)", "mean SC (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("meanSC_r_phi_profcolz_z_0-1.png")

    t.Draw("meanSC:phi:r", "z>0 && z<1", "colz")
    setup_frame("r (cm)", "#varphi (rad)", "mean SC (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("meanSC_phi_r_colz_z_0-1.png")

    t.Draw("r:z:meanDistR", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean distortion dr (cm)")
    set_margins(c1)
    c1.SaveAs("r_z_meanDistR_colz_phi_sector0.png")

    t.Draw("r:z:meanDistRPhi", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean distortion drphi (cm)")
    set_margins(c1)
    c1.SaveAs("r_z_meanDistRPhi_colz_phi_sector0.png")

    t.Draw("r:z:meanDistZ", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean distortion dz (cm)")
    set_margins(c1)
    c1.SaveAs("r_z_meanDistZ_colz_phi_sector0.png")

    t.Draw("z:r:flucSC", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "SC distortion fluctuation (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("r_z_flucSC_colz_phi_sector0.png")

    if is_idc:
        t.Draw("r:z:meanCorrR", "phi>0 && phi<3.14/9", "colz")
        setup_frame("z (cm)", "r (cm)", "mean correction dr (cm)")
        set_margins(c1)
        c1.SaveAs("r_z_meanCorrR_colz_phi_sector0.png")

        t.Draw("r:z:meanCorrRPhi", "phi>0 && phi<3.14/9", "colz")
        setup_frame("z (cm)", "r (cm)", "mean correction drphi (cm)")
        set_margins(c1)
        c1.SaveAs("r_z_meanCorrRPhi_colz_phi_sector0.png")

        t.Draw("r:z:meanCorrZ", "phi>0 && phi<3.14/9", "colz")
        setup_frame("z (cm)", "r (cm)", "mean correction dz (cm)")
        set_margins(c1)
        c1.SaveAs("r_z_meanCorrZ_colz_phi_sector0.png")

        vec_mean_one_idc = t.GetBranch("mean1DIDC")
        vec_fluc_one_idc = t.GetBranch("fluc1DIDC")
        draw_one_idc(vec_mean_one_idc, vec_fluc_one_idc, c1)

def main():
    draw_input(is_idc=True)

if __name__ == "__main__":
    main()
