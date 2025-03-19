import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

df = pd.read_csv('Lubeck5_40_Final.csv')

selected_genes = ['FNTA', 'RPL13A', 'PRKAB2', 'LMAN1', 'PLEKHH3', 'FOXG1', 'LENEP', 'PLEKHM1', 'ZNF331', 'WASL', 'PALM', 'MMRN2', 'UPF3B', 'HK1', 'CADPS', 'KLRB1', 'ARMCX4', 'IGF2BP3', 'TMEM158', 'EXOC3', 'TJP2', 'MICA', 'HNRNPAB', 'CRADD', 'CCR1', 'ABCD1', 'OLFML1', 'OTUD3', 'PRKAR1B', 'AQP6', 'FLCN', 'YBX2', 'NTSR2', 'ABCC3', 'EPYC', 'WDR43', 'TBX5', 'TIMM10', 'DLEC1', 'RIN3', 'ACTL6A', 'NOL8', 'MCFD2', 'RNASET2', 'PRPF4B', 'PTPRM', 'CASR', 'RANBP2', 'ABCD4', 'SMARCD2', 'PPP2R5C', 'RAD54L', 'GABARAPL2', 'IRF9', 'FNBP1', 'SLC1A1', 'SYNGR3', 'ZBTB43', 'AOAH', 'ATP6V0D1', 'MAL', 'CYP4F3', 'MPP1', 'SCNN1D', 'ZNF385D', 'ZXDA', 'ZNF362', 'ZNF415', 'TEX2', 'GALNT4', 'CACNA1A', 'PODXL2', 'MYC', 'WDR91', 'TNFSF12', 'CRKL', 'MAP2K7', 'TUBB3', 'NDUFB4', 'ZBTB39', 'MARCKS', 'USP48', 'KLHDC8A', 'ARTN', 'LMAN1L', 'TM6SF1', 'ANKHD1-EIF4EBP3', 'LRRC31', 'ARPC5', 'PA2G4', 'TOP3B', 'MBD4', 'FAM49A', 'STC1', 'SLC22A4', 'TOM1L1', 'KNTC1', 'POMZP3', 'MEP1A', 'PDCD6IP', 'OSGIN1', 'HIST1H2BJ', 'DIP2A', 'ZZZ3', 'PRR11', 'CSTF3', 'MAP4K3', 'STAG2', 'PTPN11', 'TIA1', 'DOCK10', 'UROD', 'MRPL41', 'CLN8', 'SEC14L5', 'PLEKHG3', 'AGGF1', 'DYRK1B', 'IVL', 'RASIP1', 'ITGB6', 'TMSB10', 'ALS2CL', 'ITIH5', 'BET1', 'GREB1', 'STAT5A', 'HSPA14', 'TIAF1', 'FCGR2B', 'CD99', 'LLGL1', 'EFNB2', 'NDUFA10', 'WT1', 'APOA4', 'RGS2', 'ARHGEF1', 'RAB3A', 'HNRNPA0', 'SLC12A2', 'TOM1', 'OR12D2', 'POU6F2', 'PGM1', 'RBM34', 'PSMC1', 'CNOT8', 'MCTP1', 'RND3', 'CNTD2', 'HSPA1A', 'CYP1A1', 'DSC2', 'SULT1C2', 'SULT1E1', 'IL1B', 'UNC93A', 'SLAMF7', 'NDUFA4L2', 'CNPY3', 'RNF125', 'NUP188', 'SPINK1', 'ATP6V1G1', 'CYP3A7', 'ELMO1', 'NAT2', 'PRAMEF10', 'TESK1', 'GGT1', 'CERKL', 'LIN37', 'TMEM106C', 'WNT10B', 'NCAPG', 'RNASE6', 'CD34', 'ATXN2L', 'COL9A2', 'RPS24', 'TSPAN32', 'GPC3', 'GATM', 'LRRC19', 'TUBB2B', 'ZNF446', 'LILRA4', 'CALD1', 'MYT1L', 'INTS8', 'HMGN4', 'HJURP', 'AP3B2', 'MMRN1', 'TANK', 'MSH3', 'ADAMTS1', 'TOPORS', 'CEND1', 'TMCC2', 'TSPAN5', 'PRSS22', 'YPEL1', 'ZNF267', 'PEG10', 'CSH2', 'PQLC2', 'CDKN2C', 'PGBD5', 'SLC17A3', 'COX6C', 'SAC3D1', 'SRP19', 'NEUROG1', 'RHCE', 'CCDC51', 'NOL4', 'PEPD', 'ZAP70', 'LACTB2', 'C9', 'HGFAC', 'GCLM', 'GINS1', 'FASTK', 'STK3', 'RPS3', 'ALOX15', 'CHMP2B', 'MYOM2', 'SGTA', 'NRIP1', 'PCBP3', 'SIM1', 'CETN3', 'GFI1', 'ZFPL1', 'FZD7', 'TCHH', 'NFATC1', 'ALDH3A1', 'NOL12', 'UFC1', 'PMVK', 'RND2', 'NEUROD2', 'RACGAP1', 'COPS3', 'SLC12A3', 'BATF', 'GNAT1', 'EDN2', 'CD47', 'RYR2', 'CWF19L1', 'NLRP2', 'ZNF551', 'MCCC1', 'FERMT2', 'RAD9A', 'PQBP1', 'NPC1L1', 'TRIP13', 'LTA4H', 'IAPP', 'SPP1', 'TRAM1', 'RABIF', 'SNX15', 'GABRB3', 'LY6H', 'MNS1', 'SERHL2', 'RALA', 'RECQL', 'PHACTR2', 'PRSS8', 'RYK', 'RAB3D', 'KLHL3', 'BNIP3L', 'CGRRF1', 'GRM6', 'BCAS4', 'SIX3', 'CXCR6', 'ARMC1', 'RCE1', 'LYVE1', 'ZNF12', 'ARL2', 'GALNT3', 'DDX49', 'PEX5', 'CYP2A13', 'AKR1B10', 'EP300', 'RSAD2', 'CTSC', 'IL1RL1', 'RHOT2', 'SCN4A', 'HOXD1', 'TOP2A', 'IFT57', 'TOB2', 'LAGE3', 'IL32', 'CCDC15', 'PPIH', 'SV2A', 'TMEM45A', 'MYL6', 'CDH15', 'GNL3', 'MOSPD1', 'VEZF1', 'EI24', 'NONO', 'GPR89B', 'DLK2', 'TYROBP', 'TRAF2', 'MCM2', 'FSTL4', 'KLRG1', 'SCN11A', 'EPB41L4A', 'MFN1', 'PNPLA4', 'TP53BP2', 'MSX1', 'HIST1H2AM', 'PLVAP', 'SLC34A2', 'RIMBP2', 'PYY2', 'AIF1', 'DCTN4', 'BCAS1', 'DAB2', 'VGLL4', 'LYRM1', 'HS3ST3A1', 'NPHS2', 'STX3', 'HEXB', 'STK16', 'CAB39', 'MUC13', 'DPY19L2P2', 'SMYD5', 'CDH13', 'POLR2K', 'CST8', 'RAB3IL1', 'TIPRL', 'NEFH', 'MT1E', 'DBR1', 'HCLS1', 'GATA3', 'SNCAIP', 'SEC22A', 'TIMM22', 'NUBPL', 'IL9R', 'ATXN2', 'RCVRN', 'ZNF614', 'SPANXA1', 'DHX8', 'EDNRA', 'PLA2G3', 'CEP57', 'PRNP', 'APOD', 'GIP', 'ZNF148', 'HTR1B', 'GPR19', 'PELI2', 'PDE6G', 'GLUL', 'SLC28A2', 'DSPP', 'IRX5', 'OPRL1', 'HGF', 'CALCA', 'TGIF1', 'COL5A2', 'MAST1', 'TSPAN8', 'DDX58', 'TERF1', 'MPZL1', 'MATN1', 'PYGO1', 'ANKRD36', 'SUZ12', 'CYP1A2', 'DGKH', 'WDR76', 'PTTG1', 'GAS8', 'CAMK2G', 'MDK', 'DYNLT3', 'ING2', 'KRT18', 'RGN', 'SLC5A1', 'PPP2R2B', 'CXCL11', 'MAPKAPK2', 'MYH11', 'P2RY1', 'RBBP5', 'ARMC4', 'SAE1', 'VSNL1', 'FGF23', 'TSKS', 'POLR3C', 'TRIM28', 'MAGEA1', 'RASGRP1', 'BMP8B', 'PAF1', 'CILP', 'IGFALS', 'ALPK3', 'DNASE1L2', 'NGFR', 'IGLL1', 'CCNT1', 'BLK', 'MLC1', 'NNMT', 'FAM53C', 'TEK', 'CLN3', 'COMT', 'ADAMTSL2', 'SOAT2', 'EPHA3', 'TTC1', 'ATP6V1B2', 'PPID', 'PTN', 'LYRM2', 'ASF1B', 'INS-IGF2', 'ERN1', 'MDFI', 'ASTN2', 'ARID3B', 'NPTXR', 'CDH19', 'CYP11A1', 'ARMC7', 'MTCH1', 'TFE3', 'KCNN3', 'RIC3', 'RPP21', 'NLRP3', 'DKK1', 'COLEC10', 'LECT2', 'ANXA1', 'APAF1', 'BAP1', 'TACSTD2', 'RASSF1', 'IPPK', 'ATP6V0B', 'GPR21', 'S100A8', 'KCNB1', 'ENO3', 'TMEM100', 'FARP2', 'SLC2A9', 'PDK3', 'GYG1', 'CCDC40', 'CHEK1', 'TSPAN7', 'LEPR', 'VIPR1', 'ABCG5', 'CTBS', 'PDGFRA', 'CHST4', 'ATP5D', 'DNAJB14', 'BMP5']
X = df[selected_genes]
y = df['Symbol']

#X = df.drop('Symbol', axis=1)
#y = df['Symbol']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def check_class_distribution(y, num_classes):
    unique_classes = np.unique(y)
    return len(unique_classes) == num_classes

num_classes = len(np.unique(y_encoded))

from sklearn.model_selection import train_test_split

X_train_temp, X_test, y_train_temp, y_test = train_test_split(X_scaled, y_encoded, test_size=0.35, stratify=y_encoded, random_state=42)

X_train, X_validate, y_train, y_validate = train_test_split(X_train_temp, y_train_temp, test_size=0.25, stratify=y_train_temp, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_validate))
print("Test set size:", len(X_test))

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = {i: class_weights[i] for i in range(num_classes)}

print("Class weight dict: ", class_weights_dict)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes)
])

import tensorflow_addons as tfa

model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

def lr_schedule(epoch, lr):
    if epoch < 40:
        return lr
    elif epoch < 80:
        return lr * 0.1
    else:
        return lr * 0.01

lr_scheduler = LearningRateScheduler(lr_schedule)

early_stopping_callback = EarlyStopping(patience=100, restore_best_weights=True)

import time
start_train_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_validate, y_validate),
    epochs=300,
    callbacks=[early_stopping_callback], #, lr_scheduler
    verbose=1,
    class_weight=class_weights_dict
)

end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"Training Time: {training_time:.2f} seconds")

start_test_time = time.time()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

end_test_time = time.time()
testing_time = end_test_time - start_test_time
print(f"Testing Time: {testing_time:.2f} seconds")

print(f"Test Accuracy: {test_acc}")

model.save('model5-del.h5')

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred_classes)

confusion_mtx = pd.crosstab(pd.Series(y_test_decoded, name='Actual'), pd.Series(y_pred_decoded, name='Predicted'))
print(confusion_mtx)

confusion_mtx.to_csv('confusion_matrix.csv')