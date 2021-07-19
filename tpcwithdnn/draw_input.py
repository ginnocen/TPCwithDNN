# pylint: disable=too-many-statements
import os

from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare, kDot # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT,  gPad  # pylint: disable=import-error, no-name-in-module

def setup_frame(x_label, y_label, z_label=None, x_offset=1.0, y_offset=1.2, z_offset=1.2):
    htemp = gPad.GetPrimitive("htemp")

    htemp.GetXaxis().SetTitle(x_label)
    htemp.GetXaxis().SetTitleOffset(x_offset)
    htemp.GetXaxis().CenterTitle(True)
    htemp.GetXaxis().SetTitleSize(0.035)
    htemp.GetXaxis().SetLabelSize(0.035)

    htemp.GetYaxis().SetTitle(y_label)
    htemp.GetYaxis().SetTitleOffset(y_offset)
    htemp.GetYaxis().CenterTitle(True)
    htemp.GetYaxis().SetTitleSize(0.035)
    htemp.GetYaxis().SetLabelSize(0.035)

    if z_label is not None:
        htemp.GetZaxis().SetTitle(z_label)
        htemp.GetZaxis().SetTitleOffset(z_offset)
        htemp.GetZaxis().SetTitleSize(0.035)
        htemp.GetZaxis().CenterTitle(True)
        htemp.GetZaxis().SetLabelSize(0.035)

def set_margins(canvas, right=0.15, left=0.1, top=0.03, bottom=0.1):
    canvas.SetRightMargin(right)
    canvas.SetLeftMargin(left)
    canvas.SetTopMargin(top)
    canvas.SetBottomMargin(bottom)

def draw_input(dirplots, draw_idc):
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    if not os.path.isdir(dirplots):
        os.makedirs(dirplots)

    if draw_idc:
        dir_infix = "idc-20210508/trees"
    else:
        dir_infix = "old-input-trees"
    f = TFile.Open("/mnt/temp/mkabus/%s/" % dir_infix +\
                   "treeInput_mean1.00_phi180_r65_z65.root","READ")
    t = f.Get("validation")

    t.SetMarkerStyle(kFullSquare)

    c1 = TCanvas()

    t.Draw("r:z:meanSC", "phi>0 && phi<3.14/9", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})")
    set_margins(c1)
    c1.SaveAs("%s/r_z_meanSC_colz_phi_sector0.png" % dirplots)

    t.Draw("meanSC-flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255)", "eventId == 0", "profcolz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "#it{#rho}_{SC} (fC/cm^{3})")
    set_margins(c1)
    c1.SaveAs("%s/r_z_randomSC_profcolz_phi_sector0_event0.png" % dirplots)

    t.Draw("meanSC:r:phi>>htemp(180, 0., 6.28, 65, 83, 255)", "z>0 && z<1", "profcolz")
    setup_frame("#it{#varphi} (rad)", "#it{r} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})")
    set_margins(c1)
    c1.SaveAs("%s/meanSC_r_phi_profcolz_z_0-1.png" % dirplots)

    t.Draw("meanSC:phi:r", "z>0 && z<1", "colz")
    setup_frame("#it{#varphi} (rad)", "#it{<#rho>}_{SC} (fC/cm^{3})", "#it{r} (cm)")
    set_margins(c1)
    c1.SaveAs("%s/meanSC_phi_r_colz_z_0-1.png" % dirplots)

    t.Draw("r:z:meanDistR", "phi>0 && phi<3.14/9", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{r}> (cm)")
    set_margins(c1)
    c1.SaveAs("%s/r_z_meanDistR_colz_phi_sector0.png" % dirplots)

    t.Draw("r:z:meanDistRPhi", "phi>0 && phi<3.14/9", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{r#varphi}> (cm)")
    set_margins(c1)
    c1.SaveAs("%s/r_z_meanDistRPhi_colz_phi_sector0.png" % dirplots)

    t.Draw("r:z:meanDistZ", "phi>0 && phi<3.14/9", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{z}> (cm)")
    set_margins(c1)
    c1.SaveAs("%s/r_z_meanDistZ_colz_phi_sector0.png" % dirplots)

    t.Draw("flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255)", "phi>0 && phi<3.14/9", "profcolz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "#it{#rho}_{SC} - #it{<#rho>}_{SC} (fC/cm^{3})",
                z_offset=1.4)
    set_margins(c1)
    c1.SaveAs("%s/flucSC_r_z_profcolz_phi_sector0.png" % dirplots)

    t.Draw("flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255)", "phi>0 && phi<3.14/9 && eventId == 0",
           "profcolz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "#it{#rho}_{SC} - #it{<#rho>}_{SC} (fC/cm^{3})")
    set_margins(c1)
    c1.SaveAs("%s/r_z_flucSC_profcolz_phi_sector0_event0.png" % dirplots)

    t.Draw("r:z:flucDistR>>htemp(65, 0, 250, 65, 83, 255)", "phi>0 && phi<3.14/9 && eventId == 0",
           "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "distortion fluctuation d#it{r} - <d#it{r}> (cm)",
                z_offset=1.5)
    set_margins(c1)
    c1.SaveAs("%s/r_z_flucDistR_colz_phi_sector0_event0.png" % dirplots)

    t.Draw("r:z:flucDistRPhi>>htemp(65, 0, 250, 65, 83, 255)",
           "phi>0 && phi<3.14/9 && eventId == 0", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)",
                "distortion fluctuation d#it{r#varphi} - <d#it{r#varphi}> (cm)", z_offset=1.5)
    set_margins(c1)
    c1.SaveAs("%s/r_z_flucDistRPhi_colz_phi_sector0_event0.png" % dirplots)

    t.Draw("r:z:flucDistZ>>htemp(65, 0, 250, 65, 83, 255)",
           "phi>0 && phi<3.14/9 && eventId == 0", "colz")
    setup_frame("#it{z} (cm)", "#it{r} (cm)", "distortion fluctuation d#it{z} - <d#it{z}> (cm)",
                z_offset=1.5)
    set_margins(c1)
    c1.SaveAs("%s/r_z_flucDistZ_colz_phi_sector0_event0.png" % dirplots)

    t.SetMarkerStyle(kDot)

    t.Draw("meanSC:z>>htemp(65, 0, 250, 20, 0.1, 0.12)", "", "profcolz")
    setup_frame("#it{z} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})", y_offset=1.9)
    set_margins(c1, right=0.02, left=0.15)
    c1.SaveAs("%s/meanSC_z_profcolz.png" % dirplots)

    t.Draw("meanSC-flucSC:z>>htemp(65, 0, 250, 20, 0.1, 0.12)", "", "profcolz")
    setup_frame("#it{z} (cm)", "#it{#rho}_{SC} (fC/cm^{3})", y_offset=1.5)
    set_margins(c1, right=0.02, left=0.15)
    c1.SaveAs("%s/randomSC_z_profcolz.png" % dirplots)

    t.SetMarkerStyle(kFullSquare)

    if draw_idc:
        t.Draw("r:z:meanCorrR", "phi>0 && phi<3.14/9", "colz")
        setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{r}> (cm)")
        set_margins(c1)
        c1.SaveAs("%s/r_z_meanCorrR_colz_phi_sector0.png" % dirplots)

        t.Draw("r:z:meanCorrRPhi", "phi>0 && phi<3.14/9", "colz")
        setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{r#varphi}> (cm)")
        set_margins(c1)
        c1.SaveAs("%s/r_z_meanCorrRPhi_colz_phi_sector0.png" % dirplots)

        t.Draw("r:z:meanCorrZ", "phi>0 && phi<3.14/9", "colz")
        setup_frame("#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{z}> (cm)")
        set_margins(c1)
        c1.SaveAs("%s/r_z_meanCorrZ_colz_phi_sector0.png" % dirplots)

        t.Draw("fluc1DIDC", "", "", 200, 0)
        setup_frame("fluc 1D IDC", "entries")
        set_margins(c1)
        c1.SaveAs("%s/fluc_1D_IDC.png" % dirplots)

        t.Draw("fluc0DIDC:flucCorrR", "fluc0DIDC!=0 && r<100 && z<50", "")
        setup_frame("0D IDC fluctuations", "d#it{r} - <d#it{r}> (cm)")
        set_margins(c1, right=0.05, top=0.05)
        c1.SaveAs("%s/flucCorrR_fluc0DIDC.png" % dirplots)

        t.Draw("fluc0DIDC:flucCorrRPhi", "fluc0DIDC!=0 && r<100 && z<50", "")
        setup_frame("0D IDC fluctuations", "d#it{r#varphi} - <d#it{r#varphi}> (cm)")
        set_margins(c1, right=0.05, top=0.05)
        c1.SaveAs("%s/flucCorrRPhi_fluc0DIDC.png" % dirplots)

        t.Draw("fluc0DIDC:flucCorrZ", "fluc0DIDC!=0 && r<100 && z<50", "")
        setup_frame("0D IDC fluctuations", "d#it{z} - <d#it{z}> (cm)")
        set_margins(c1, right=0.05, top=0.05)
        c1.SaveAs("%s/flucCorrZ_fluc0DIDC.png" % dirplots)


def main():
    draw_input(dirplots="idc-new-plots", draw_idc=True)

if __name__ == "__main__":
    main()
