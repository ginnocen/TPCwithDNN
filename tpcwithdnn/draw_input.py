# pylint: disable=too-many-statements
from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT,  gPad  # pylint: disable=import-error, no-name-in-module

def setup_frame(x_label, y_label, z_label=None):
    htemp = gPad.GetPrimitive("htemp")

    htemp.GetXaxis().SetTitle(x_label)
    htemp.GetXaxis().SetTitleOffset(1.0)
    htemp.GetXaxis().CenterTitle(True)
    htemp.GetXaxis().SetTitleSize(0.035)
    htemp.GetXaxis().SetLabelSize(0.035)

    htemp.GetYaxis().SetTitle(y_label)
    htemp.GetYaxis().SetTitleOffset(1.0)
    htemp.GetYaxis().CenterTitle(True)
    htemp.GetYaxis().SetTitleSize(0.035)
    htemp.GetYaxis().SetLabelSize(0.035)

    if z_label is not None:
        htemp.GetZaxis().SetTitle(z_label)
        htemp.GetZaxis().SetTitleOffset(1.0)
        htemp.GetZaxis().SetTitleSize(0.035)
        htemp.GetZaxis().CenterTitle(True)
        htemp.GetZaxis().SetLabelSize(0.035)

def set_margins(canvas):
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.1)
    canvas.SetTopMargin(0.03)
    canvas.SetBottomMargin(0.1)

def draw_input(draw_idc):
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    f = TFile.Open("/mnt/temp/mkabus/idc-study-20210310/" +\
                   "trees/treeInput_mean1.00_phi180_r65_z65.root","READ")
    t = f.Get("validation")

    t.SetMarkerStyle(kFullSquare)

    c1 = TCanvas()

    t.Draw("r:z:meanSC", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean SC (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("r_z_meanSC_colz_phi_sector0.png")

    t.Draw("meanSC:r:phi>>htemp(180, 0., 6.28, 65, 83, 255, 20, 0., 0.4)", "z>0 && z<1", "profcolz")
    setup_frame("#varphi (rad)", "r (cm)", "mean SC (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("meanSC_r_phi_profcolz_z_0-1_bins_180-65-20.png")

    t.Draw("meanSC:phi:r", "z>0 && z<1", "colz")
    setup_frame("#varphi (rad)", "mean SC (fC/cm^3)", "r (cm)")
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

    t.Draw("flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255, 20, -0.02, 0.02)", "phi>0 && phi<3.14/9", "profcolz")
    setup_frame("z (cm)", "r (cm)", "SC distortion fluctuation (fC/cm^3)")
    set_margins(c1)
    c1.SaveAs("flucSC_r_z_profcolz_phi_sector0_bins_65-65-20.png")

    if draw_idc:
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

        t.Draw("mean1DIDC", "", "", 200, 0)
        htemp = gPad.GetPrimitive("htemp")
        htemp.SetMinimum(0)
        htemp.SetMaximum(0.6e-6)
        setup_frame("mean 1D IDC", "entries")
        set_margins(c1)
        c1.SaveAs("mean_1D_IDC.png")

        t.Draw("fluc1DIDC", "", "", 200, 0)
        setup_frame("fluc 1D IDC", "entries")
        set_margins(c1)
        c1.SaveAs("fluc_1D_IDC.png")


def main():
    draw_input(draw_idc=True)

if __name__ == "__main__":
    main()
