# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist
try:
    import statsmodels
except ImportError:
    import os

    os.system("pip install statsmodels")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile("D:/Arbeit/test alle Massen/Test alle Massen4.xlsx")

# Liste der Bedingungen
conditions = ["Control", "6-20%", "21-35%", "over 35%"]

# Liste der Excel-Seiten
sheets = ["Jena"]  # , 'Jena']

# Liste der Massen, für die Boxplots erstellt werden sollen
masses = [
    "255.2325",
    "279.2343",
    "281.248",
    "303.2336",
    "305.2518",
    "307.2719",
    "327.2347",
    "328.2382",
    "375.2282",
    "391.2259",
    "392.2291",
    "415.2258",
    "439.2259",
    "480.3084",
    "594.2813",
    "626.3145",
    "682.5081",
    "697.4814",
    "736.6431",
    "773.5363",
    "893.7275",
]

for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    print(df.head())  # Drucken Sie den Kopf des DataFrame

    # Konvertieren Sie die Spaltennamen in Ihrem DataFrame in Strings
    df.columns = df.columns.astype(str)

    for mass in masses:
        # Überprüfen Sie, ob die Masse in den Spalten des DataFrames vorhanden ist
        if mass in df.columns:
            plt.figure(figsize=(10, 5))
            # Boxplot für die ausgewählte Masse und alle Bedingungen erstellen
            sns.boxplot(
                x=df["Condition"], y=df[mass], order=conditions, showfliers=True
            )
            sns.swarmplot(x=df["Condition"], y=df[mass], order=conditions, color=".25")

            # Signifikanz im Titel anzeigen
            plt.title("Boxplot for Mass " + str(mass) + " in " + "Cohort 1")

            plt.ylabel("Value")
            plt.xlabel("Condition")
            plt.savefig(
                "D:/Arbeit/test alle Massen/"
                + sheet
                + "_"
                + str(mass)
                + "_boxplot.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.show()
            plt.close()

# Installieren Sie das statsmodels-Modul, falls es noch nicht installiert ist
try:
    import statsmodels
except ImportError:
    import os

    os.system("pip install statsmodels")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
xls = pd.ExcelFile("D:/Arbeit/test alle Massen/Test alle Massen4.xlsx")

# Liste der Bedingungen
conditions = ["Control", "6-20%", "21-35%", "over 35%"]

# Liste der Excel-Seiten
sheets = ["Leipzig"]

# Liste der Massen, für die Boxplots erstellt werden sollen
masses = [
    "261.22",
    "279.2343",
    "281.248",
    "303.2336",
    "307.2719",
    "308.0999",
    "463.2859",
    "480.3084",
    "568.2783",
    "590.2423",
    "611.2839",
    "626.3145",
    "639.2427",
    "640.5277",
    "643.4746",
    "671.4641",
    "697.4814",
    "736.6431",
    "749.514",
    "788.4971",
    "822.7144",
    "835.7129",
    "851.7551",
    "863.7384",
    "877.7681",
    "885.5517",
    "891.7596",
    "896.7348",
    "922.7419",
    "932.8129",
]

for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    print(df.head())  # Drucken Sie den Kopf des DataFrame

    # Konvertieren Sie die Spaltennamen in Ihrem DataFrame in Strings
    df.columns = df.columns.astype(str)

    for mass in masses:
        # Überprüfen Sie, ob die Masse in den Spalten des DataFrames vorhanden ist
        if mass in df.columns:
            plt.figure(figsize=(10, 5))
            # Boxplot für die ausgewählte Masse und alle Bedingungen erstellen
            sns.boxplot(
                x=df["Condition"], y=df[mass], order=conditions, showfliers=True
            )
            sns.swarmplot(x=df["Condition"], y=df[mass], order=conditions, color=".25")

            # Signifikanz im Titel anzeigen
            plt.title("Boxplot for Mass " + str(mass) + " in " + "Cohort 2")

            plt.ylabel("Value")
            plt.xlabel("Condition")
            plt.savefig(
                "D:/Arbeit/test alle Massen/"
                + sheet
                + "_"
                + str(mass)
                + "_boxplot.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.show()
            plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# Daten einlesen
data_xls = pd.ExcelFile("D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx")
masses_xls = pd.ExcelFile(
    "D:\Arbeit\Figure Venndiagramm\Alle Conditons vergleich/Cohort2_Venn_Results.xlsx"
)

# Liste der Bedingungen
conditions = ["Control", "6-20%", "21-35%", "over 35%"]

# Liste der Excel-Seiten
sheets = ["Cohort2"]  # Hier können Sie weitere Blätter hinzufügen

# Massen aus der Massen-Tabelle einlesen
# Einlesen der Massen aus der zweiten Zeile und der ersten Spalte
masses_df = pd.read_excel(masses_xls, sheet_name="6-20%v21-35%UniqueDown")
masses = masses_df.iloc[1:, 0].dropna().astype(str).tolist()


def add_significance(ax, pairs, p_values, y_max):
    y_offset = 0.05  # Abstand zwischen den Linien
    for (i, j), p_value in zip(pairs, p_values):
        if p_value < 0.05:
            x1, x2 = i, j
            y, h, col = y_max + y_offset, 1.00, "k"
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            ax.text(
                (x1 + x2) * 0.5,
                y + h,
                "*",
                ha="center",
                va="bottom",
                color=col,
                fontsize=12,
            )
            y_offset += 5.00  # Erhöhe den y-Wert für die nächste Linie


for sheet in sheets:
    df = pd.read_excel(data_xls, sheet_name=sheet)
    print(df.head())  # Drucken Sie den Kopf des DataFrame

    # Konvertieren Sie die Spaltennamen in Ihrem DataFrame in Strings
    df.columns = df.columns.astype(str)

    for mass in masses:
        # Überprüfen Sie, ob die Masse in den Spalten des DataFrames vorhanden ist
        if mass in df.columns:
            plt.figure(figsize=(10, 5))
            # Boxplot für die ausgewählte Masse und alle Bedingungen erstellen
            ax = sns.boxplot(
                x=df["Condition"], y=df[mass], order=conditions, showfliers=True
            )
            sns.swarmplot(x=df["Condition"], y=df[mass], order=conditions, color=".25")

            # Paarweise t-Tests durchführen
            tukey = pairwise_tukeyhsd(
                endog=df[mass], groups=df["Condition"], alpha=0.05
            )
            print(tukey.summary())

            # Signifikanz im Titel anzeigen
            plt.title("Boxplot for Mass " + str(mass) + " in " + sheet)

            plt.ylabel("Value")
            plt.xlabel("Condition")

            # Signifikanz auf dem Plot anzeigen
            pairs = list(combinations(range(len(conditions)), 2))
            add_significance(ax, pairs, tukey.pvalues, df[mass].max())

            plt.savefig(
                "D:\Arbeit\Figure Venndiagramm\Alle Conditons vergleich/"
                + sheet
                + "_"
                + str(mass)
                + "_boxplot.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.show()
            plt.close()
