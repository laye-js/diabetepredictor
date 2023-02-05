# Importation des bibliothèques nécessaires
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz


app = FastAPI()
@app.get("/predict")
async def predict(Pregnancies:int,Glucose:int,BloodPressure:int,SkinThickness:int,Insulin:int,BMI:float,DiabetesPedigreeFunction:float,Age:int):
    # Chargement des données
    data= pd.read_csv("./data.csv")
    X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    y = data["Outcome"]

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialisation et entraînement du modèle d'arbre de décision
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
    dt.fit(X_train, y_train)
    
    # Prédiction sur des données de test
    new_data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
    predictions = dt.predict(new_data)
        # Calcul du gain
    gain = dt.score(X_test, y_test)

    if(int(predictions[0])==0):
        return {"resultat":"vous n'avez pas le diabete","taux d'exactitude":accuracy,"gain":gain}
    else:
        return {"resultat":"vous avez  le diabete","taux d'exactitude":accuracy,"gain":gain}
 


