import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

def estoy_bien():
    print("hola")

if __name__ == "__main__":

    df_words = pd.read_csv("./conteo.csv")
    plt.figure(figsize=(12,6))
    sns.barplot(x = "Palabra", y = "Cantidad_Videos", data = df_words)
    plt.xlabel("Palabras")
    plt.ylabel("Cantidad de Videos")
    plt.title("Conteo de Palabras en Videos")
    plt.xticks(rotation=45)
    plt.show()