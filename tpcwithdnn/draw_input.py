from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module
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

def draw_input():
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    f = TFile.Open("trees/treeInput_mean1.0_phi180_r65_z65.root","READ")
    t = f.Get("validation")

    t.SetMarkerStyle(kFullSquare)

    c1 = TCanvas()

    t.Draw("meanSC:r:phi", "z>0 && z<1", "profcolz")
    setup_frame("#varphi (rad)", "r (cm)", "mean SC (fC/cm^3)")
    c1.SetRightMargin(0.15)
    c1.SetLeftMargin(0.1)
    c1.SetTopMargin(0.03)
    c1.SetBottomMargin(0.1)
    c1.SaveAs("meanSC_r_phi_profcolz_z_0-1_labelled_with_z.png")

    t.Draw("r:z:meanDistR", "phi>0 && phi<3.14/9", "colz")
    setup_frame("z (cm)", "r (cm)", "mean distorsion dr (cm)")
    c1.SetRightMargin(0.15)
    c1.SetLeftMargin(0.1)
    c1.SetTopMargin(0.03)
    c1.SetBottomMargin(0.1)
    c1.SaveAs("r_z_meanDistR_phi_sector0_labelled.png")

def main():
    draw_input()

if __name__ == "__main__":
    main()
