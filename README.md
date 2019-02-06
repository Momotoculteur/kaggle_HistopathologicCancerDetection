#kaggle_HistopathologicCancerDetection

## DATASET
disponible sur https://www.kaggle.com/c/histopathologic-cancer-detection

## INFOS
Prototype de pre process d'images pour un CNN


**modelDirectPreProcess.py**
Utilisation d'un dataframe qui va aggreger les chemins d'accès et leurs noms et leurs labels correspondant

**modelKerasPreProcess.py**
Utilisation du dataGenerator fournit par Keras. Nécessite d'avoir une structure de data telle quelle :
-Dataset
  |-train
     |img1.tif
     |img2.tif
  |-validation
     |img1.tif
     |img2.tif
  
Non utilisable si tout est mélangé telle quelle que l'ont à les données à la base. Nécessite un script pour ré organiser les data.
