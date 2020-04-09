void DrawProfile(const Int_t phislice=180, const Int_t rslice=33, const Int_t zslice=33, const Bool_t useSCmean = kTRUE, const Bool_t useSCFluc=kTRUE, const Bool_t isR=kTRUE, const Bool_t isRPhi = kFALSE, const Bool_t isZ=kFALSE);
void DrawStdDev(const Int_t phislice=180, const Int_t rslice=33, const Int_t zslice=33, const Bool_t useSCmean = kTRUE, const Bool_t useSCFluc=kTRUE, const Bool_t isR=kTRUE, const Bool_t isRPhi = kFALSE, const Bool_t isZ=kFALSE);
void DrawMeanSD(const Int_t phislice=180, const Int_t rslice=33, const Int_t zslice=33, const Bool_t useSCmean = kTRUE, const Bool_t useSCFluc=kTRUE, const Bool_t isR=kTRUE, const Bool_t isRPhi = kFALSE, const Bool_t isZ=kFALSE);
void DrawNfilter(const Int_t phislice=180, const Int_t rslice=33, const Int_t zslice=33, const Bool_t useSCmean = kTRUE, const Bool_t useSCFluc=kTRUE, const Bool_t isR=kTRUE, const Bool_t isRPhi = kFALSE, const Bool_t isZ=kFALSE, const Int_t Nev=3000);
void DrawProfileAll(){

	DrawProfile(180,33,33,kTRUE,kTRUE,kTRUE,kFALSE,kFALSE);
	DrawProfile(180,33,33,kTRUE,kTRUE,kFALSE,kTRUE,kFALSE);
	DrawProfile(180,33,33,kTRUE,kTRUE,kFALSE,kFALSE,kTRUE);

	DrawProfile(180,33,33,kFALSE,kTRUE,kTRUE,kFALSE,kFALSE);
	DrawProfile(180,33,33,kFALSE,kTRUE,kFALSE,kTRUE,kFALSE);
	DrawProfile(180,33,33,kFALSE,kTRUE,kFALSE,kFALSE,kTRUE);

}
//______________________________________________________________
void DrawProfile(const Int_t phislice, const Int_t rslice, const Int_t zslice, const Bool_t useSCmean, const Bool_t useSCFluc, const Bool_t isR, const Bool_t isRPhi, const Bool_t isZ)
{
	gStyle->SetOptStat(0);
	gStyle->SetOptTitle(0);

	TCanvas *c1 = new TCanvas("c1","c1",0,0,800,800);
	c1->SetMargin(0.12,0.05,0.12,0.05);
	c1->SetTicks(1,1);

	TH1F *frame = (TH1F*)c1->DrawFrame(-5,-0.5,+5,+0.5);
	if(isR){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dr");
		frame->GetYaxis()->SetTitle("mean value of (pred. - num.) in 3000 test events (cm), dr");
	}
	else if(isRPhi){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), rd#varphi");
		frame->GetYaxis()->SetTitle("mean value of (pred. - num.) in 3000 test events (cm), rd#varphi");
	}
	else if(isZ){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dz");
		frame->GetYaxis()->SetTitle("mean value of (pred. - num.) in 3000 test events (cm), dz");
	}
	frame->GetXaxis()->SetTitleOffset(1.5);
	frame->GetYaxis()->SetTitleOffset(1.5);
	frame->GetXaxis()->CenterTitle(kTRUE);
	frame->GetYaxis()->CenterTitle(kTRUE);
	frame->GetXaxis()->SetTitleSize(0.035);
	frame->GetYaxis()->SetTitleSize(0.035);
	frame->GetXaxis()->SetLabelSize(0.035);
	frame->GetYaxis()->SetLabelSize(0.035);

	const Int_t Nev[] = {1000, 5000, 10000};
	const Int_t n = sizeof(Nev)/sizeof(Nev[0]);
	const Int_t color[n] = {kBlue+1,kGreen+2,kRed+1};

	const TString str_param = Form("phi%d_r%d_z%d_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean%d_useSCFluc%d_pred_doR%d_dophi%d_doz%d",phislice,rslice,zslice,useSCmean,useSCFluc,isR,isRPhi,isZ);

	TLegend *leg = new TLegend(0.65,0.65,0.9,0.85);
	leg->SetBorderSize(0);
	leg->SetTextSize(0.04);
	for(Int_t i=0;i<n;i++){
		TString filename = Form("output%s_Nev%d.root",str_param.Data(),Nev[i]);
		printf("reading ... %s\n",filename.Data());

		TFile *rootfile = TFile::Open(filename,"READ");
		TProfile *h1prf = (TProfile*)rootfile->Get(Form("profiledeltasvsdistallevents%s_Nev%d",str_param.Data(),Nev[i]));
		h1prf->SetDirectory(0);
		h1prf->Draw("same");
		h1prf->SetMarkerStyle(20);
		h1prf->SetMarkerColor(color[i]);
		h1prf->SetLineColor(color[i]);

		leg->AddEntry(h1prf,Form("N_{ev}^{training} = %d k",Int_t(Nev[i]/1e+3)),"LP");
		rootfile->Close();

	}//end of Nev loop
	leg->Draw();

	TPaveText *txt1 = new TPaveText(0.15,0.8,0.4,0.92,"NDC");
	txt1->SetFillColor(kWhite);
	txt1->SetFillStyle(0);
	txt1->SetBorderSize(0);
	txt1->SetTextAlign(12);//middle,left
	txt1->SetTextFont(42);//helvetica
	txt1->SetTextSize(0.04);
  txt1->AddText(Form("#varphi slice= %d, r slice= %d, z slice= %d",phislice,rslice,zslice));
	if(useSCFluc && useSCmean) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>, <#rho_{SC}>");
	else if(useSCFluc) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>");
	txt1->Draw();

  c1->SaveAs(Form("20200403_wide_Profile_%s.png",str_param.Data()));
  c1->SaveAs(Form("20200403_wide_Profile_%s.eps",str_param.Data()));
  c1->SaveAs(Form("20200403_wide_Profile_%s.pdf",str_param.Data()));

  frame->GetYaxis()->SetRangeUser(-0.05,+0.05);

  c1->SaveAs(Form("20200403_zoom_Profile_%s.png",str_param.Data()));
  c1->SaveAs(Form("20200403_zoom_Profile_%s.eps",str_param.Data()));
  c1->SaveAs(Form("20200403_zoom_Profile_%s.pdf",str_param.Data()));

}
//______________________________________________________________________________________________________________________________________________
void DrawStdDev(const Int_t phislice, const Int_t rslice, const Int_t zslice, const Bool_t useSCmean, const Bool_t useSCFluc, const Bool_t isR, const Bool_t isRPhi, const Bool_t isZ)
{
	gStyle->SetOptStat(0);
	gStyle->SetOptTitle(0);

	TCanvas *c1 = new TCanvas("c1","c1",0,0,800,800);
	c1->SetMargin(0.12,0.05,0.12,0.05);
	c1->SetTicks(1,1);

	TH1F *frame = (TH1F*)c1->DrawFrame(-5,0.,+5,+0.2);
	if(isR){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dr");
		frame->GetYaxis()->SetTitle("std. dev. of (pred. - num.) in 3000 test events (cm), dr");
	}
	else if(isRPhi){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), rd#varphi");
		frame->GetYaxis()->SetTitle("std. dev. of (pred. - num.) in 3000 test events (cm), rd#varphi");
	}
	else if(isZ){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dz");
		frame->GetYaxis()->SetTitle("std. dev. of (pred. - num.) in 3000 test events (cm), dz");
	}
	frame->GetXaxis()->SetTitleOffset(1.5);
	frame->GetYaxis()->SetTitleOffset(1.5);
	frame->GetXaxis()->CenterTitle(kTRUE);
	frame->GetYaxis()->CenterTitle(kTRUE);
	frame->GetXaxis()->SetTitleSize(0.035);
	frame->GetYaxis()->SetTitleSize(0.035);
	frame->GetXaxis()->SetLabelSize(0.035);
	frame->GetYaxis()->SetLabelSize(0.035);

	const Int_t Nev[] = {1000, 5000, 10000};
	const Int_t n = sizeof(Nev)/sizeof(Nev[0]);
	const Int_t color[n] = {kBlue+1,kGreen+2,kRed+1};

	const TString str_param = Form("phi%d_r%d_z%d_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean%d_useSCFluc%d_pred_doR%d_dophi%d_doz%d",phislice,rslice,zslice,useSCmean,useSCFluc,isR,isRPhi,isZ);

	TLegend *leg = new TLegend(0.65,0.65,0.9,0.85);
	leg->SetBorderSize(0);
	leg->SetTextSize(0.04);
	for(Int_t i=0;i<n;i++){
		TString filename = Form("output%s_Nev%d.root",str_param.Data(),Nev[i]);
		printf("reading ... %s\n",filename.Data());

		TFile *rootfile = TFile::Open(filename,"READ");
		TH1D *h1 = (TProfile*)rootfile->Get(Form("hStdDev_allevents%s_Nev%d",str_param.Data(),Nev[i]));
		h1->SetDirectory(0);
		h1->Draw("same");
		h1->SetMarkerStyle(20);
		h1->SetMarkerColor(color[i]);
		h1->SetLineColor(color[i]);

		leg->AddEntry(h1,Form("N_{ev}^{training} = %d k",Int_t(Nev[i]/1e+3)),"LP");
		rootfile->Close();

	}//end of Nev loop
	leg->Draw();

	TPaveText *txt1 = new TPaveText(0.15,0.8,0.4,0.92,"NDC");
	txt1->SetFillColor(kWhite);
	txt1->SetFillStyle(0);
	txt1->SetBorderSize(0);
	txt1->SetTextAlign(12);//middle,left
	txt1->SetTextFont(42);//helvetica
	txt1->SetTextSize(0.04);
  txt1->AddText(Form("#varphi slice= %d, r slice= %d, z slice= %d",phislice,rslice,zslice));
	if(useSCFluc && useSCmean) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>, <#rho_{SC}>");
	else if(useSCFluc) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>");
	txt1->Draw();

  c1->SaveAs(Form("20200403_wide_StdDev_%s.png",str_param.Data()));
  c1->SaveAs(Form("20200403_wide_StdDev_%s.eps",str_param.Data()));
  c1->SaveAs(Form("20200403_wide_StdDev_%s.pdf",str_param.Data()));

  //frame->GetYaxis()->SetRangeUser(-0.05,+0.05);

  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.png",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.eps",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.pdf",str_param.Data()));

}
//______________________________________________________________________________________________________________________________________________
void DrawMeanSD(const Int_t phislice, const Int_t rslice, const Int_t zslice, const Bool_t useSCmean, const Bool_t useSCFluc, const Bool_t isR, const Bool_t isRPhi, const Bool_t isZ)
{
	gStyle->SetOptStat(0);
	gStyle->SetOptTitle(0);

	TCanvas *c1 = new TCanvas("c1","c1",0,0,800,800);
	c1->SetMargin(0.12,0.05,0.12,0.05);
	c1->SetTicks(1,1);

	TH1F *frame = (TH1F*)c1->DrawFrame(-5,-0.5,+5,+0.5);
	if(isR){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dr");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), dr");
	}
	else if(isRPhi){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), rd#varphi");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), rd#varphi");
	}
	else if(isZ){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dz");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), dz");
	}
	frame->GetXaxis()->SetTitleOffset(1.5);
	frame->GetYaxis()->SetTitleOffset(1.5);
	frame->GetXaxis()->CenterTitle(kTRUE);
	frame->GetYaxis()->CenterTitle(kTRUE);
	frame->GetXaxis()->SetTitleSize(0.035);
	frame->GetYaxis()->SetTitleSize(0.035);
	frame->GetXaxis()->SetLabelSize(0.035);
	frame->GetYaxis()->SetLabelSize(0.035);

	const Int_t Nev[] = {1000, 5000, 10000};
	const Int_t n = sizeof(Nev)/sizeof(Nev[0]);
	const Int_t color[n] = {kBlue+1,kGreen+2,kRed+1};

	const TString str_param = Form("phi%d_r%d_z%d_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean%d_useSCFluc%d_pred_doR%d_dophi%d_doz%d",phislice,rslice,zslice,useSCmean,useSCFluc,isR,isRPhi,isZ);

	TLegend *leg = new TLegend(0.65,0.65,0.9,0.85);
	leg->SetBorderSize(0);
	leg->SetTextSize(0.04);
	for(Int_t i=0;i<n;i++){
		TString filename = Form("output%s_Nev%d.root",str_param.Data(),Nev[i]);
		printf("reading ... %s\n",filename.Data());

		TFile *rootfile = TFile::Open(filename,"READ");
		TH1D *h1 = (TProfile*)rootfile->Get(Form("hMeanSD_allevents%s_Nev%d",str_param.Data(),Nev[i]));
		h1->SetDirectory(0);
		h1->Draw("sameE2");
		h1->SetMarkerStyle(20);
		h1->SetMarkerColor(color[i]);
		h1->SetLineColor(color[i]);
		h1->SetFillColor(color[i]);
		h1->SetFillStyle(3001);

		leg->AddEntry(h1,Form("N_{ev}^{training} = %d k",Int_t(Nev[i]/1e+3)),"LP");
		rootfile->Close();

	}//end of Nev loop
	leg->Draw();

	TPaveText *txt1 = new TPaveText(0.15,0.8,0.4,0.92,"NDC");
	txt1->SetFillColor(kWhite);
	txt1->SetFillStyle(0);
	txt1->SetBorderSize(0);
	txt1->SetTextAlign(12);//middle,left
	txt1->SetTextFont(42);//helvetica
	txt1->SetTextSize(0.04);
  txt1->AddText(Form("#varphi slice= %d, r slice= %d, z slice= %d",phislice,rslice,zslice));
	if(useSCFluc && useSCmean) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>, <#rho_{SC}>");
	else if(useSCFluc) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>");
	txt1->Draw();

  c1->SaveAs(Form("20200405_MeanSD_%s.png",str_param.Data()));
  c1->SaveAs(Form("20200405_MeanSD_%s.eps",str_param.Data()));
  c1->SaveAs(Form("20200405_MeanSD_%s.pdf",str_param.Data()));

  //frame->GetYaxis()->SetRangeUser(-0.05,+0.05);

  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.png",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.eps",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.pdf",str_param.Data()));

}
//______________________________________________________________________________________________________________________________________________
void DrawNfilter(const Int_t phislice, const Int_t rslice, const Int_t zslice, const Bool_t useSCmean, const Bool_t useSCFluc, const Bool_t isR, const Bool_t isRPhi, const Bool_t isZ, const Int_t Nev)
{
	gStyle->SetOptStat(0);
	gStyle->SetOptTitle(0);

	TCanvas *c1 = new TCanvas("c1","c1",0,0,800,800);
	c1->SetMargin(0.12,0.05,0.12,0.05);
	c1->SetTicks(1,1);

	TH1F *frame = (TH1F*)c1->DrawFrame(-5,-0.3,+5,+0.3);
	if(isR){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dr");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), dr");
	}
	else if(isRPhi){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), rd#varphi");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), rd#varphi");
	}
	else if(isZ){
		frame->GetXaxis()->SetTitle("numerical fluctuation (cm), dz");
		frame->GetYaxis()->SetTitle("mean #pm std. dev. of (pred. - num.) in 3000 test events (cm), dz");
	}
	frame->GetXaxis()->SetTitleOffset(1.5);
	frame->GetYaxis()->SetTitleOffset(1.5);
	frame->GetXaxis()->CenterTitle(kTRUE);
	frame->GetYaxis()->CenterTitle(kTRUE);
	frame->GetXaxis()->SetTitleSize(0.035);
	frame->GetYaxis()->SetTitleSize(0.035);
	frame->GetXaxis()->SetLabelSize(0.035);
	frame->GetYaxis()->SetLabelSize(0.035);

	const Int_t Nf[] = {2,4,8};
	const Int_t n = sizeof(Nf)/sizeof(Nf[0]);
	const Int_t color[n] = {kBlue+1,kGreen+2,kRed+1};


  //TString str_param = Form("phi%d_r%d_z%d_filter%d_poo0_drop0.00_depth4_batch0_scaler0_useSCMean%d_useSCFluc%d_pred_doR%d_dophi%d_doz%d",phislice,rslice,zslice,Nf[i],useSCmean,useSCFluc,isR,isRPhi,isZ);
  TString str_param = Form("phi%d_r%d_z%d_Nfscan_poo0_drop0.00_depth4_batch0_scaler0_useSCMean%d_useSCFluc%d_pred_doR%d_dophi%d_doz%d",phislice,rslice,zslice,useSCmean,useSCFluc,isR,isRPhi,isZ);
	TLegend *leg = new TLegend(0.7,0.7,0.92,0.92);
	leg->SetBorderSize(0);
	leg->SetTextSize(0.035);
	for(Int_t i=0;i<n;i++){
    TString tmp = str_param.Copy().ReplaceAll("Nfscan",Form("filter%d",Nf[i]));

		TString filename = Form("output%s_Nev%d.root",tmp.Data(),Nev);
		printf("reading ... %s\n",filename.Data());

		TFile *rootfile = TFile::Open(filename,"READ");
		TH1D *h1 = (TProfile*)rootfile->Get(Form("hMeanSD_allevents%s_Nev%d",tmp.Data(),Nev));
		h1->SetDirectory(0);
		h1->Draw("sameE2");
		h1->SetMarkerStyle(20);
		h1->SetMarkerColor(color[i]);
		h1->SetLineColor(color[i]);
		h1->SetFillColor(color[i]);
		h1->SetFillStyle(3001);

		leg->AddEntry(h1,Form("N_{filters} = %d",Nf[i]),"LP");
		rootfile->Close();

	}//end of Nev loop
	leg->Draw();

	TPaveText *txt1 = new TPaveText(0.15,0.8,0.4,0.92,"NDC");
	txt1->SetFillColor(kWhite);
	txt1->SetFillStyle(1001);
	txt1->SetBorderSize(0);
	txt1->SetTextAlign(12);//middle,left
	txt1->SetTextFont(42);//helvetica
	txt1->SetTextSize(0.035);
  txt1->AddText(Form("#varphi slice= %d, r slice= %d, z slice= %d",phislice,rslice,zslice));
	if(useSCFluc && useSCmean) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>, <#rho_{SC}>");
	else if(useSCFluc) txt1->AddText("inputs : #rho_{SC} - <#rho_{SC}>");
  txt1->AddText(Form("N_{ev}^{training} = %d k",Int_t(Nev/1e+3)));
	txt1->Draw();

  c1->SaveAs(Form("20200405_MeanSD_%s.png",str_param.Data()));
  c1->SaveAs(Form("20200405_MeanSD_%s.eps",str_param.Data()));
  c1->SaveAs(Form("20200405_MeanSD_%s.pdf",str_param.Data()));

  //frame->GetYaxis()->SetRangeUser(-0.05,+0.05);

  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.png",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.eps",str_param.Data()));
  //c1->SaveAs(Form("20200403_zoom_StdDev_%s.pdf",str_param.Data()));

}
//______________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________
