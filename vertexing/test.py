import uproot
#from numpy import asarray
#from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_validate
from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TH2F, TFile # pylint: disable=import-error
#import xgboost # pylint: disable=import-error

Xvar = ["fXTrack1", "fAlphaTrack1", "fYTrack1", "fZTrack1", "fSnpTrack1", \
        "fTglTrack1", "fSigned1PtTrack1", \
        "fSigmaSnpTrack1", "fSigmaTglTrack1", "fSigmaSigned1PtTrack1", \
        "fRhoZYTrack1", "fRhoSnpYTrack1", "fRhoSnpZTrack1", "fRhoTglYTrack1", \
        "fRhoTglZTrack1", "fRhoTglSnpTrack1", "fRho1PtYTrack1", \
        "fRho1PtZTrack1", "fRho1PtSnpTrack1", "fRho1PtTglTrack1",
        "fXTrack2", "fAlphaTrack2", "fYTrack2", "fZTrack2", "fSnpTrack2", \
        "fTglTrack2", "fSigned1PtTrack2", \
        "fSigmaSnpTrack2", "fSigmaTglTrack2", "fSigmaSigned1PtTrack2", \
        "fRhoZYTrack2", "fRhoSnpYTrack2", "fRhoSnpZTrack2", "fRhoTglYTrack2", \
        "fRhoTglZTrack2", "fRhoTglSnpTrack2", "fRho1PtYTrack2", \
        "fRho1PtZTrack2", "fRho1PtSnpTrack2", "fRho1PtTglTrack2",
        "fPhiTrack2", "fEtaTrack2"]

yvar = ["fXSecondaryVertex"]

treeevtorig = uproot.open("/data/vertexingDNN/VertexHFML_0.root")["secondaryVertexHFML"]
dataframe = treeevtorig.pandas.df(branches=Xvar+yvar)
dataframe = dataframe[0:500000]
X = dataframe[Xvar]
y = dataframe[yvar]


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)

def get_model(n_inputs_, n_outputs_):
    model_ = Sequential()
    model_.add(Dense(400, input_dim=n_inputs_, kernel_initializer='he_uniform', activation='relu'))
    model_.add(Dense(n_outputs_, kernel_initializer='he_uniform'))
    model_.compile(loss='mae', optimizer='adam')
    return model_

n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X_train, y_train, verbose=1, epochs=30)
#xgb.fit(X_train, y_train)
y_pred = model.predict(X_test)
histo = TH2F("histo", "histo", 100, -0.5, 0.5, 100, -0.5, 0.5)
print(y_test.shape)
print(y_pred.shape)
y_test["fXSecondaryVertex_pred"] = y_pred[:,0]
#y_test["fYSecondaryVertex_pred"] = y_pred[:,1]
#y_test["fZSecondaryVertex_pred"] = y_pred[:,2]
df_rd = y_test[["fXSecondaryVertex", "fXSecondaryVertex_pred"]]
arr2 = df_rd.to_numpy()
fill_hist(histo, arr2)
fileout = TFile("fileout.root", "recreate")
fileout.cd()
histo.Write()
