
# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist
try:
    import statsmodels
except ImportError:
    import os
    os.system('pip install statsmodels')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
colors = ['b', 'orange', 'g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort2 biased']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=30, azim=160)

    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition in conditions:
        condition_data = pca_result[df['Condition'] == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color='k', alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 29.83%')
    ax.set_ylabel('PC2 - 7.57%')
    ax.set_zlabel('PC3 - 3.97%')
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    # Fügen Sie die Legende hinzu
    #ax.legend(loc='upper right')

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/ pca' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
colors = ['b', 'orange', 'g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort1 unbiased']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=30, azim=160)

    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition in conditions:
        condition_data = pca_result[df['Condition'] == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color='k', alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 16.65%')
    ax.set_ylabel('PC2 - 5.95%')
    ax.set_zlabel('PC3 - 5.65%')
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    # Fügen Sie die Legende hinzu
    #ax.legend(loc='upper right')

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/ pca' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
colors = ['b', 'orange', 'g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort2 unbiased']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=30, azim=160)

    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition in conditions:
        condition_data = pca_result[df['Condition'] == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color='k', alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 16.03%')
    ax.set_ylabel('PC2 - 10.30%')
    ax.set_zlabel('PC3 - 5.82%')
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    # Fügen Sie die Legende hinzu
    #ax.legend(loc='upper right')

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/ pca' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
  
    
  
    
  
# biased durch P-value
# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
colors = ['b', 'orange', 'g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort2']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=30, azim=160)

    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition in conditions:
        condition_data = pca_result[df['Condition'] == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 23.10%')
    ax.set_ylabel('PC2 - 12.85%')
    ax.set_zlabel('PC3 - 5.87%')
    ax.legend(conditions)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    # Fügen Sie die Legende hinzu
    #ax.legend(loc='upper right')

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        #ax.view_init(elev=20, azim=50)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/ pca' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
colors = ['b', 'orange', 'g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort1']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=30, azim=160)

    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition in conditions:
        condition_data = pca_result[df['Condition'] == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 22.86%')
    ax.set_ylabel('PC2 - 22.86%')
    ax.set_zlabel('PC3 - 4.84%')
    ax.legend(conditions)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    # Fügen Sie die Legende hinzu
    #ax.legend(loc='upper right')

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        #ax.view_init(elev=20, azim=50)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm/ pca' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
    
    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Cohort1', 'Cohort2']
colors = ['g', 'r']

# Liste der Excel-Seiten
sheets = ['Cohort12']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df!=0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 33.82%%')
    ax.set_ylabel('PC2 - 8.78%')
    ax.set_zlabel('PC3 - 5.30%')
    ax.legend(conditions)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_3D_PCA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


    
#Levenes Testwie unterschiedlich sind beide Gruppen festgelegte Cluster
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm\Revision/Statistic_variance_clusters4.xlsx')

# Liste der Bedingungen
conditions = ['Cohort1', 'Cohort2']
colors = ['g', 'r']

# Liste der Excel-Seiten
sheets = ['LJ']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])
    
    # Durchführung von Levene's Test für Varianz-Homogenität
    group1 = pca_result[cohorts == 'Cohort1']
    group2 = pca_result[cohorts == 'Cohort2']
    stat, p_value = stats.levene(group1[:, 0], group2[:, 0], center='mean')
    print(f"Levene's Test statistic for PC1: {stat}, p-value: {p_value}")

    # Erstellen Sie einen 3D-Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 - 33.82%%')
    ax.set_ylabel('PC2 - 8.78%')
    ax.set_zlabel('PC3 - 5.30%')
    ax.legend(conditions)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm\Revision' + sheet + '_3D_PCA_Cluster.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], color='k')
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:\Arbeit\Figure Venndiagramm\Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


#inter gruppenvarianz und boxplots festgelegte Cluster Cohort1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm\Revision/Statistic_variance_clusters6.xlsx')

# Liste der Bedingungen
conditions = ['Cohort1']
colors = ['g']

# Liste der Excel-Seiten
sheets = ['Cohort1']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])
    
    for i, item in enumerate(pca.explained_variance_ratio_):
      print(f"Varianz für Cohort1 (PC{i+1}): {item*100} %")
      
      
     # Erstellung eines 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
    ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
    ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
    ax.legend(groups)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für Combinde Group')
    
    # Beschriften Sie die Achsen
    #ax.set_xlabel('PC1 - 33.82%%')
    #ax.set_ylabel('PC2 - 8.78%')
    #ax.set_zlabel('PC3 - 5.30%')
    #ax.legend(conditions)
    #ax.grid()
    #plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_3D_PCA_Cluster.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2])
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()

    

#Cohort2 Varianz Clusteranalyse   
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm\Revision/Statistic_variance_clusters6.xlsx')

# Liste der Bedingungen
conditions = ['Cohort2']
colors = ['r']

# Liste der Excel-Seiten
sheets = ['Cohort2']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])
    
    for i, item in enumerate(pca.explained_variance_ratio_):
      print(f"Varianz für Cohort2 (PC{i+1}): {item*100} %")
      
     # Erstellung eines 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
    ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
    ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
    ax.legend(groups)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für Combinde Group')
    
    # Beschriften Sie die Achsen
    #ax.set_xlabel('PC1 - 33.82%%')
    #ax.set_ylabel('PC2 - 8.78%')
    #ax.set_zlabel('PC3 - 5.30%')
    #ax.legend(conditions)
    #ax.grid()
    #plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_3D_PCA_Cluster.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2])
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    

#riesige Gruppe und boxplots Cohorts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import pyplot

# Daten einlesen
xls = pd.ExcelFile('D:/Arbeit/Figure Venndiagramm/Revision/Statistic_variance_clusters6.xlsx')

# Liste der Gruppen
groups = ['CombinedGroup']  # Neue große Gruppe
colors = ['blue']  # Farbe für die große Gruppe

# Liste der Excel-Seiten
sheets = ['LJ']

# Blätter auslesen und analysieren
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)

    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]

    # Ersetzen Sie NaN-Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity-Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)

    # Daten für PCA extrahieren
    data = df.iloc[:, 4:].values  # Ihre Daten
    
    # Zielvariable in Kategorien umwandeln
    target = pd.cut(df.iloc[:, -1], bins=2, labels=['low', 'high']).to_numpy()

    # Verteilung der Daten visualisieren
    fig, ax = plt.subplots()
    ax.hist(data[:, 0], bins=25, color='blue', alpha=0.7, label='Data')
    ax.set_title('Datenverteilung')
    ax.set_xlabel('Werte')
    ax.set_ylabel('Häufigkeit')
    plt.legend()
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Revision_{sheet}_PCA_Datenverteilung.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Bootstrapping-Analyse mit sklearn.utils.resample und DecisionTreeClassifier
    n_iterations = 3000
    n_size = int(len(data) * 0.50)
    stats = list()

    for i in range(n_iterations):
        # prepare train and test sets
        train_indices = resample(range(len(data)), n_samples=n_size)
        train_data, train_target = data[train_indices], target[train_indices]
        test_indices = [i for i in range(len(data)) if i not in train_indices]
        test_data, test_target = data[test_indices], target[test_indices]

        # fit model
        model = DecisionTreeClassifier()
        model.fit(train_data, train_target)

        # evaluate model
        predictions = model.predict(test_data)
        score = accuracy_score(test_target, predictions)
        print(score)
        stats.append(score)

    # Plot scores
    pyplot.hist(stats)
    pyplot.title('Bootstrap Scores Distribution')
    pyplot.xlabel('Accuracy Score')
    pyplot.ylabel('Frequency')
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Revision_{sheet}_PCA_Score_Distribution.png', bbox_inches='tight', dpi=300)
    pyplot.show()

    # Calculate and print confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

    # PCA-Analyse
    pca = PCA(n_components=3)
    pca_ft = pca.fit_transform(data)

    # Varianzanteile der Hauptkomponenten
    for i, item in enumerate(pca.explained_variance_ratio_):
        print(f"Varianz für CombinedGroup (PC{i+1}): {item*100:.2f}%")

    # Erstellung eines 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Fügen Sie die Datenpunkte zum Plot hinzu
    group_data = pca_ft
    ax.scatter(group_data[:, 0], group_data[:, 1], group_data[:, 2], label='CombinedGroup', color='blue', alpha=1.0)

    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
    ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
    ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
    ax.legend(groups)
    ax.grid()
    plt.title('3D PCA Scatter Plot für Combined Group')
    ax.view_init(elev=230, azim=135)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Revision_{sheet}_3D_PCA_Cluster.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[4:])

    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for index, row in loadings.iterrows():
        ax.scatter(row['PC1'], row['PC2'], row['PC3'])
        ax.text(row['PC1'], row['PC2'], row['PC3'], index, fontsize=10)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=230, azim=135)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Revision_{sheet}_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()




#inter gruppenvarianz echte Gruppen Cohort1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm\Revision/Statistic_variance_clusters7.xlsx')

# Liste der Bedingungen
conditions = ['Cohort1']
colors = ['g']

# Liste der Excel-Seiten
sheets = ['Cohort1_alt']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])
    
    for i, item in enumerate(pca.explained_variance_ratio_):
      print(f"Varianz für Cohort1 (PC{i+1}): {item*100} %")
      
      
     # Erstellung eines 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
    ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
    ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
    ax.legend(groups)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für Combinde Group')
    
    # Beschriften Sie die Achsen
    #ax.set_xlabel('PC1 - 33.82%%')
    #ax.set_ylabel('PC2 - 8.78%')
    #ax.set_zlabel('PC3 - 5.30%')
    #ax.legend(conditions)
    #ax.grid()
    #plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_3D_PCA_Cluster.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2])
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()

    

#Cohort2 echte Gruppen   
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile('D:\Arbeit\Figure Venndiagramm\Revision/Statistic_variance_clusters7.xlsx')

# Liste der Bedingungen
conditions = ['Cohort2']
colors = ['r']

# Liste der Excel-Seiten
sheets = ['Cohort2_alt']

# Führen Sie die PCA für jedes Arbeitsblatt durch
for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Entfernen Sie Zeilen in df, in denen alle Werte Null sind
    df = df.loc[(df != 0).any(axis=1)]
    
    # Ersetzen Sie NaN Werte durch 0
    df = df.fillna(0)

    # Ersetzen Sie Infinity Werte durch die größte finite Zahl
    df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Speichern der 'Cohort' Spalte und Entfernen aus dem DataFrame
    cohorts = df['Cohort']
    df = df.drop(columns=['Cohort'])
    
    # PCA-Objekt erstellen
    pca = PCA(n_components=3)

    # Führen Sie die PCA auf den Lipidwerten durch
    pca_result = pca.fit_transform(df.iloc[:, 3:])
    
    for i, item in enumerate(pca.explained_variance_ratio_):
      print(f"Varianz für Cohort2 (PC{i+1}): {item*100} %")
      
     # Erstellung eines 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Fügen Sie die Datenpunkte zum Plot hinzu
    for condition, color in zip(conditions, colors):
        condition_data = pca_result[cohorts == condition]
        ax.scatter(condition_data[:, 0], condition_data[:, 1], condition_data[:, 2], label=condition, color=color, alpha=1.0)
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
    ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
    ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
    ax.legend(groups)
    ax.grid()
    plt.title(f'3D PCA Scatter Plot für Combinde Group')
    
    # Beschriften Sie die Achsen
    #ax.set_xlabel('PC1 - 33.82%%')
    #ax.set_ylabel('PC2 - 8.78%')
    #ax.set_zlabel('PC3 - 5.30%')
    #ax.legend(conditions)
    #ax.grid()
    #plt.title(f'3D PCA Scatter Plot für {sheet}')
    ax.view_init(elev=310, azim=320)
    
    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_3D_PCA_Cluster.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Ausgabe der Loadings für jede Hauptkomponente
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[3:])
    
    # Erstellen Sie einen 3D-Plot für die Loadings
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(loadings.shape[0]):
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2])
        ax.text(loadings.iloc[i, 0], loadings.iloc[i, 1], loadings.iloc[i, 2], loadings.index[i], fontsize=10)
        
    # Beschriften Sie die Achsen
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=310, azim=320)

    # Speichern Sie den Plot als PNG-Datei
    plt.savefig('D:/Arbeit/Figure Venndiagramm/Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    

#riesige Gruppe Cohorts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Daten einlesen
xls = pd.read_excel('D:\Arbeit\Figure Venndiagramm\p value daten neu3.xlsx')

# Liste der Gruppen
groups = ['CombinedGroup']  # Neue große Gruppe
colors = ['blue']  # Farbe für die große Gruppe

# Liste der Excel-Seiten
sheets = ['Cohort12']

#df = pd.read_excel(xls, sheet_name=sheet)
df = xls
# Entfernen Sie Zeilen in df, in denen alle Werte Null sind
df = df.loc[(df != 0).any(axis=1)]

# Ersetzen Sie NaN Werte durch 0
df = df.fillna(0)

# Ersetzen Sie Infinity Werte durch die größte finite Zahl
df = df.replace([np.inf, -np.inf], np.finfo(np.float64).max)
data = df.iloc[:, 4:]


pca = PCA(n_components=3)
pca_ft = pca.fit_transform(data)


for i, item in enumerate(pca.explained_variance_ratio_):
  print(f"Varianz für CombinedGroup (PC{i+1}): {item*100} %")
 # Erstellung eines 3D-Plots
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Fügen Sie die Datenpunkte zum Plot hinzu
group_data = pca_ft
ax.scatter(group_data[:, 0], group_data[:, 1], group_data[:, 2], label='CombinedGroup', color='blue', alpha=1.0)


# Beschriften Sie die Achsen
ax.set_xlabel('PC1 ('+'{:.2f}'.format(pca.explained_variance_ratio_[0]*100)+'%)')
ax.set_ylabel('PC2 ('+'{:.2f}'.format(pca.explained_variance_ratio_[1]*100)+'%)')
ax.set_zlabel('PC3 ('+'{:.2f}'.format(pca.explained_variance_ratio_[2]*100)+'%)')
ax.legend(groups)
ax.grid()
plt.title(f'3D PCA Scatter Plot für Combinde Group')

ax.view_init(elev=230, azim=135)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df.columns[4:])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for index, row in loadings.iterrows():
  #print([row['PC1'],row['PC2'],row['PC3']])
  ax.scatter(row['PC1'],row['PC2'],row['PC3'])
  ax.text(row['PC1'],row['PC2'],row['PC3'], index, fontsize=10)

ax.view_init(elev=230, azim=135)

# Speichern Sie den Plot als PNG-Datei
plt.savefig('D:\Arbeit\Figure Venndiagramm\Revision' + sheet + '_PCA_Loadings.png', bbox_inches='tight', dpi=300)
plt.show()