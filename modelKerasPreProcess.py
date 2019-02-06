import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

#Chargement des donnees
data = pd.read_csv('./train_labels.csv', sep=',')
#has_ext est deprecated, on rajoute donc les extensions de fichiers à la main
data['id'] = data['id'].astype(str) + '.tif'

#On split en train/val
train, val = train_test_split(data,test_size=0.2, shuffle=True)



trainDatagen=ImageDataGenerator(rescale=1./255)
valDatagen=ImageDataGenerator(rescale=1./255)

train_generator = trainDatagen.flow_from_dataframe(dataframe=train, directory="./train/", x_col="id",
                                            y_col="label", class_mode="binary", target_size=(96,96),
                                            batch_size=32)



val_generator = valDatagen.flow_from_dataframe(dataframe=val, directory="./train/", x_col="id",
                                            y_col="label", class_mode="binary", target_size=(96,96),
                                            batch_size=32)












