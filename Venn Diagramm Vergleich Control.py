import math, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib_venn import venn3

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort2')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']

unique_masses_up = {}
unique_masses_down = {}

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]

# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zur Control
for condition in conditions[1:]:
    p_values = []
    log2_fold_changes = []
    for mass in filtered_masses:
        control_values = df[df['Condition'] == 'Control'][mass]
        condition_values = df[df['Condition'] == condition][mass]
        if control_values.mean() != 0 and condition_values.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
            t_stat, p_value = stats.ttest_ind(control_values, condition_values, equal_var=False)
            log2_fold_change = np.log2(condition_values.mean() / control_values.mean())
            p_values.append(p_value)
            log2_fold_changes.append(log2_fold_change)
        else:
            p_values.append(np.nan)
            log2_fold_changes.append(np.nan)

    # Erstellen des Volcanoplots
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

    # Hervorheben von signifikanten Massen
    significant_up = [(log2_fc, -np.log10(p_val)) for log2_fc, p_val in zip(log2_fold_changes, p_values) if log2_fc > 3 and p_val < 0.05]
    significant_down = [(log2_fc, -np.log10(p_val)) for log2_fc, p_val in zip(log2_fold_changes, p_values) if log2_fc < -2.5 and p_val < 0.05]

    if significant_up:
        plt.scatter(*zip(*significant_up), c='red', label='Upregulated')
    if significant_down:
        plt.scatter(*zip(*significant_down), c='blue', label='Downregulated')

    # Hinzufügen von Nummerierung zu den Datenpunkten und Erstellen der Legende
    legend_elements_up = []  # Liste für erhöhte Massen
    legend_elements_down = []  # Liste für erniedrigte Massen
    num = 1
    for i in range(len(log2_fold_changes)):
        if (log2_fold_changes[i] > 3 and -np.log10(p_values[i]) > 1.3) or (log2_fold_changes[i] < -2.5 and -np.log10(p_values[i]) > 1.3):
            plt.text(log2_fold_changes[i], -np.log10(p_values[i]), str(num), fontsize=8)
            if log2_fold_changes[i] > 3:
                legend_elements_up.append(plt.Line2D([0], [0], marker='o', color='w', label=str(num)+': '+str(filtered_masses[i]), markerfacecolor='red', markersize=10))
            elif log2_fold_changes[i] < -2.5:
                legend_elements_down.append(plt.Line2D([0], [0], marker='o', color='w', label=str(num)+': '+str(filtered_masses[i]), markerfacecolor='blue', markersize=10))
            num += 1

    plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
    plt.axvline(x=3, color='k', linestyle='--')
    plt.axvline(x=-2.5, color='k', linestyle='--')

    # Erstellen Sie eine Legende mit den Massen
    legend_up = plt.legend(handles=legend_elements_up, title="Increased Masses", loc="upper left", bbox_to_anchor=(1,1))
    plt.gca().add_artist(legend_up)
    legend_down = plt.legend(handles=legend_elements_down, title="Decreased Masses", loc="upper left", bbox_to_anchor=(1.2,1))
    plt.gca().add_artist(legend_down)

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title(f'Volcanoplot for {condition} compared to Control in Cohort1')
    plt.legend()
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Cohort2_{condition}_vs_Control_volcanoplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Erstellen Sie einen DataFrame mit den Massen, die außerhalb der Schwellenwerte liegen
    significant_up = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 3 and -np.log10(p_values[i]) > 1.3]
    significant_down = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -2.5 and -np.log10(p_values[i]) > 1.3]

    # Fügen Sie die Massen zum Wörterbuch hinzu
    key_up = f"Cohort2_{condition}_up"
    key_down = f"Cohort2_{condition}_down"
    if key_up not in unique_masses_up:
        unique_masses_up[key_up] = set(significant_up)
    else:
        unique_masses_up[key_up].update(significant_up)

    if key_down not in unique_masses_down:
        unique_masses_down[key_down] = set(significant_down)
    else:
        unique_masses_down[key_down].update(significant_down)

# Erstellen der Venn-Diagramme für erhöhte und erniedrigte Massen
conditions_labels = ['6-20%', '21-35%', 'over 35%']

# Erhöhte Massen
sets_up = [unique_masses_up.get(f"Cohort2_{condition}_up", set()) for condition in conditions[1:]]
venn3(sets_up, conditions_labels)
plt.title('Venn Diagram for Increased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Venn_Increased_Masses.png')
plt.show()

# Erniedrigte Massen
sets_down = [unique_masses_down.get(f"Cohort2_{condition}_down", set()) for condition in conditions[1:]]
venn3(sets_down, conditions_labels)
plt.title('Venn Diagram for Decreased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Venn_Decreased_Masses.png')
plt.show()

# generate list index for itertools combinations
def gen_index(n):
    x = -1
    while True:
        while True:
            x = x + 1
            if bin(x).count('1') == n:
                break
            yield x
            
# generate all combinations of intersections
def make_intersections(sets):
    l = [set()] * 2**len(sets)  # Initialize with empty sets
    for i in range(1, len(sets) + 1):
        ind = gen_index(i)
        for subset in itertools.combinations(sets, i):
            inter = set.intersection(*subset)
            l[next(ind)] = inter
    return l

# get weird reversed binary string id for venn
def number2venn_id(x, n_fill):
    id = bin(x)[2:].zfill(n_fill)
    id = id[::-1]
    return id

# iterate over all combinations and remove duplicates from intersections with more sets
def sets2dict(sets):
    global l 
    l = make_intersections(sets)
    d = {}
    for i in range(1, len(l)):
        d[number2venn_id(i, len(sets))] = l[i]
        for j in range(1, len(l)):
            if bin(j).count('1') < bin(i).count('1'):
                l[j] = l[j] - l[i]
                d[number2venn_id(j, len(sets))] = l[j] - l[i]
    return d

d = pd.DataFrame(sets2dict(sets_up))
d2 = pd.DataFrame(sets2dict(sets_down))

# Erstellen der Excel-Tabelle mit den Schnittpunkten und den nicht-überschneidenden Massen
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Cohort 1 Intersection_Masses.xlsx') as writer:
    for i, condition1 in enumerate(conditions[1:]):
        for j, condition2 in enumerate(conditions[1:]):
            if i < j:
                intersection_up = sets_up[i].intersection(sets_up[j])
                intersection_down = sets_down[i].intersection(sets_down[j])
                non_intersection_up = sets_up[i].symmetric_difference(sets_up[j])
                non_intersection_down = sets_down[i].symmetric_difference(sets_down[j])
                if intersection_up:
                    df_intersection_up = pd.DataFrame(list(intersection_up), columns=[f'{condition1} & {condition2} Inc'])
                    df_intersection_up.to_excel(writer, sheet_name=f'{condition1}_{condition2}_up', index=False)
                if intersection_down:
                    df_intersection_down = pd.DataFrame(list(intersection_down), columns=[f'{condition1} & {condition2} Dec'])
                    df_intersection_down.to_excel(writer, sheet_name=f'{condition1}_{condition2}_down', index=False)
                if non_intersection_up:
                    df_non_intersection_up = pd.DataFrame(list(non_intersection_up), columns=[f'{condition1} & {condition2} Non-Inc'])
                    df_non_intersection_up.to_excel(writer, sheet_name=f'{condition1}_{condition2}_non_up', index=False)
                if non_intersection_down:
                    df_non_intersection_down = pd.DataFrame(list(non_intersection_down), columns=[f'{condition1} & {condition2} Non-Dec'])
                    df_non_intersection_down.to_excel(writer, sheet_name=f'{condition1}_{condition2}_non_down', index=False) 


#Volcanoplots mit Legende und alter Nummerierung

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib_venn import venn3

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort1')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']

unique_masses_up = {}
unique_masses_down = {}

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]

# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zur Control
for condition in conditions[1:]:
    p_values = []
    log2_fold_changes = []
    for mass in filtered_masses:
        control_values = df[df['Condition'] == 'Control'][mass]
        condition_values = df[df['Condition'] == condition][mass]
        if control_values.mean() != 0 and condition_values.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
            t_stat, p_value = stats.ttest_ind(control_values, condition_values, equal_var=False)
            log2_fold_change = np.log2(condition_values.mean() / control_values.mean())
            p_values.append(p_value)
            log2_fold_changes.append(log2_fold_change)
        else:
            p_values.append(np.nan)
            log2_fold_changes.append(np.nan)

    # Erstellen des Volcanoplots
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

    # Hervorheben von signifikanten Massen
    significant_up = [(log2_fc, -np.log10(p_val)) for log2_fc, p_val in zip(log2_fold_changes, p_values) if log2_fc > 1 and p_val < 0.05]
    significant_down = [(log2_fc, -np.log10(p_val)) for log2_fc, p_val in zip(log2_fold_changes, p_values) if log2_fc < -1 and p_val < 0.05]

    if significant_up:
        plt.scatter(*zip(*significant_up), c='red', label='Upregulated')
    if significant_down:
        plt.scatter(*zip(*significant_down), c='blue', label='Downregulated')

    # Hinzufügen von Nummerierung zu den Datenpunkten und Erstellen der Legende
    legend_elements_up = []  # Liste für erhöhte Massen
    legend_elements_down = []  # Liste für erniedrigte Massen
    num = 1
    for i in range(len(log2_fold_changes)):
        if (log2_fold_changes[i] > 1 and -np.log10(p_values[i]) > 1.3) or (log2_fold_changes[i] < -1 and -np.log10(p_values[i]) > 1.3):
            plt.text(log2_fold_changes[i], -np.log10(p_values[i]), str(num), fontsize=8)
            if log2_fold_changes[i] > 1:
                legend_elements_up.append(plt.Line2D([0], [0], marker='o', color='w', label=str(num)+': '+str(filtered_masses[i]), markerfacecolor='red', markersize=10))
            elif log2_fold_changes[i] < -1:
                legend_elements_down.append(plt.Line2D([0], [0], marker='o', color='w', label=str(num)+': '+str(filtered_masses[i]), markerfacecolor='blue', markersize=10))
            num += 1

    plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.axvline(x=-1, color='k', linestyle='--')

    # Erstellen Sie eine Legende mit den Massen
    legend_up = plt.legend(handles=legend_elements_up, title="Increased Masses", loc="upper left", bbox_to_anchor=(1,1))
    plt.gca().add_artist(legend_up)
    legend_down = plt.legend(handles=legend_elements_down, title="Decreased Masses", loc="upper left", bbox_to_anchor=(1.2,1))
    plt.gca().add_artist(legend_down)

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title(f'Volcanoplot for {condition} compared to Control in Cohort2')
    plt.legend()
    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Cohort1_{condition}_vs_Control_volcanoplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Erstellen Sie einen DataFrame mit den Massen, die außerhalb der Schwellenwerte liegen
    significant_up = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 1 and -np.log10(p_values[i]) > 1.3]
    significant_down = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -1 and -np.log10(p_values[i]) > 1.3]

    # Fügen Sie die Massen zum Wörterbuch hinzu
    key_up = f"Cohort1_{condition}_up"
    key_down = f"Cohort1_{condition}_down"
    if key_up not in unique_masses_up:
        unique_masses_up[key_up] = set(significant_up)
    else:
        unique_masses_up[key_up].update(significant_up)

    if key_down not in unique_masses_down:
        unique_masses_down[key_down] = set(significant_down)
    else:
        unique_masses_down[key_down].update(significant_down)

# Erstellen der Venn-Diagramme für erhöhte und erniedrigte Massen
conditions_labels = ['6-20%', '21-35%', 'over 35%']

# Erhöhte Massen
sets_up = [unique_masses_up.get(f"Cohort1_{condition}_up", set()) for condition in conditions[1:]]
venn3(sets_up, conditions_labels)
plt.title('Venn Diagram for Increased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Cohort 2 Venn_Increased_Masses.png')
plt.show()

# Erniedrigte Massen
sets_down = [unique_masses_down.get(f"Cohort1_{condition}_down", set()) for condition in conditions[1:]]
venn3(sets_down, conditions_labels)
plt.title('Venn Diagram for Decreased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Cohort 2 Venn_Decreased_Masses.png')
plt.show()

# Erstellen der Excel-Tabelle mit den Schnittpunkten
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Intersection_Masses.xlsx') as writer:
    for i, condition1 in enumerate(conditions[1:]):
        for j, condition2 in enumerate(conditions[1:]):
            if i < j:
                intersection_up = sets_up[i].intersection(sets_up[j])
                intersection_down = sets_down[i].intersection(sets_down[j])
                if intersection_up:
                    df_intersection_up = pd.DataFrame(list(intersection_up), columns=[f'{condition1} & {condition2} Increased'])
                    df_intersection_up.to_excel(writer, sheet_name=f'{condition1}_{condition2}_up', index=False)
                if intersection_down:
                    df_intersection_down = pd.DataFrame(list(intersection_down), columns=[f'{condition1} & {condition2} Decreased'])
                    df_intersection_down.to_excel(writer, sheet_name=f'{condition1}_{condition2}_down', index=False)


# Gibt die Volcanoplots ohne Legende aus und eine Excelltabelle mit allen Massen und ihren Zahlen ohne Sortierung

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort2')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]

# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zur Control
mass_numbers = {mass: i+1 for i, mass in enumerate(filtered_masses)}

for condition in conditions[1:]:
    p_values = []
    log2_fold_changes = []
    for mass in filtered_masses:
        control_values = df[df['Condition'] == 'Control'][mass]
        condition_values = df[df['Condition'] == condition][mass]
        if control_values.mean() != 0 and condition_values.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
            t_stat, p_value = stats.ttest_ind(control_values, condition_values, equal_var=False)
            log2_fold_change = np.log2(condition_values.mean() / control_values.mean())
            p_values.append(p_value)
            log2_fold_changes.append(log2_fold_change)
        else:
            p_values.append(np.nan)
            log2_fold_changes.append(np.nan)

    # Erstellen des Volcanoplots
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

    # Hervorheben von signifikanten Massen
    significant_up = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc > 3 and p_val < 0.05]
    significant_down = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc < -2.5 and p_val < 0.05]

    if significant_up:
        plt.scatter(*zip(*[(x, y) for x, y, _ in significant_up]), c='red', label='Upregulated')
    if significant_down:
        plt.scatter(*zip(*[(x, y) for x, y, _ in significant_down]), c='blue', label='Downregulated')

    # Hinzufügen von Nummerierung zu den Datenpunkten
    for x, y, num in significant_up + significant_down:
        plt.text(x, y, str(num), fontsize=8)

    plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
    plt.axvline(x=3, color='k', linestyle='--')
    plt.axvline(x=-2.5, color='k', linestyle='--')

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title(f'Volcanoplot for {condition} compared to Control in Cohort2')

    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Cohort2_{condition}_vs_Control_volcanoplot.png', bbox_inches='tight', dpi=300)
    plt.close()

# Erstellen der Excel-Tabelle
data_up = []
data_down = []

for condition in conditions[1:]:
    significant_up = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 1 and -np.log10(p_values[i]) > 1.3]
    significant_down = [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -1 and -np.log10(p_values[i]) > 1.3]

    for mass in significant_up:
        data_up.append((mass_numbers[mass], mass))
    for mass in significant_down:
        data_down.append((mass_numbers[mass], mass))

# Erstellen der Tabelle
max_columns = 10  # Maximale Anzahl der Spalten pro Zeile
table_up = []
table_down = []

for i in range(0, len(data_up), max_columns):
    table_up.append(data_up[i:i + max_columns])

for i in range(0, len(data_down), max_columns):
    table_down.append(data_down[i:i + max_columns])

# Speichern der Tabelle in einer Excel-Datei
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Cohort2_Venn_Results.xlsx') as writer:
    df_up = pd.DataFrame(table_up)
    df_down = pd.DataFrame(table_down)
    
    df_up.to_excel(writer, sheet_name='Increased_Masses', index=False, header=False)
    df_down.to_excel(writer, sheet_name='Decreased_Masses', index=False, header=False)
  
    
  
    
#Gibt Volcanoplots aus und eine Liste der Massen und in welchen Gruppen sie im Venn Diagramm zu sehen sind
# Zeigt nicht die Nummern der Massen in der Excell an
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib_venn import venn3

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort2')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]


# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zur Control
mass_numbers = {mass: i+1 for i, mass in enumerate(filtered_masses)}

condition_results = {}

for condition in conditions[1:]:
    p_values = []
    log2_fold_changes = []
    for mass in filtered_masses:
        control_values = df[df['Condition'] == 'Control'][mass]
        condition_values = df[df['Condition'] == condition][mass]
        if control_values.mean() != 0 and condition_values.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
            t_stat, p_value = stats.ttest_ind(control_values, condition_values, equal_var=False)
            log2_fold_change = np.log2(condition_values.mean() / control_values.mean())
            p_values.append(p_value)
            log2_fold_changes.append(log2_fold_change)
        else:
            p_values.append(np.nan)
            log2_fold_changes.append(np.nan)

    # Erstellen des Volcanoplots
    plt.figure(figsize=(10, 6))
    plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

    # Hervorheben von signifikanten Massen
    significant_up = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc > 3 and p_val < 0.05]
    significant_down = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc < -2.5 and p_val < 0.05]

    if significant_up:
        plt.scatter(*zip(*[(x, y) for x, y, _ in significant_up]), c='red', label='Upregulated')
    if significant_down:
        plt.scatter(*zip(*[(x, y) for x, y, _ in significant_down]), c='blue', label='Downregulated')

    # Hinzufügen von Nummerierung zu den Datenpunkten
    for x, y, num in significant_up + significant_down:
        plt.text(x, y, str(num), fontsize=8)

    plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
    plt.axvline(x=3, color='k', linestyle='--')
    plt.axvline(x=-2.5, color='k', linestyle='--')

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title(f'Volcanoplot for {condition} compared to Control in Cohort2')

    plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Cohort2_{condition}_vs_Control_volcanoplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Speichern der signifikanten Massen für jede Bedingung
    condition_results[condition] = {
        'up': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 3 and -np.log10(p_values[i]) > 1.3],
        'down': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -2.5 and -np.log10(p_values[i]) > 1.3]
    }

# Erstellen der Excel-Tabelle
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Cohort2_Venn_Results.xlsx') as writer:
    # Erstellen des Venn-Diagramms
    venn_data = {}
    for condition in conditions[1:]:
        venn_data[condition] = set(condition_results[condition]['up'] + condition_results[condition]['down'])

    all_masses = set.union(*venn_data.values())
    venn_table = []

    for mass in all_masses:
        row = [mass]
        present_in_conditions = [condition for condition in conditions[1:] if mass in venn_data[condition]]
        row.extend(present_in_conditions)
        venn_table.append(row)

    venn_df = pd.DataFrame(venn_table, columns=['Mass'] + conditions[1:])
    venn_df.to_excel(writer, sheet_name='Venn_Diagram', index=False)

    # Erstellen der Tabelle für jede Bedingung und gemeinsame Massen
    for condition in conditions[1:]:
        unique_up = [mass for mass in condition_results[condition]['up'] if sum([mass in condition_results[cond]['up'] for cond in conditions[1:]]) == 1]
        unique_down = [mass for mass in condition_results[condition]['down'] if sum([mass in condition_results[cond]['down'] for cond in conditions[1:]]) == 1]

        df_up = pd.DataFrame(unique_up, columns=[f'{condition} Unique Upregulated'])
        df_down = pd.DataFrame(unique_down, columns=[f'{condition} Unique Downregulated'])
        
        df_up.to_excel(writer, sheet_name=f'{condition}_Unique_Up', index=False)
        df_down.to_excel(writer, sheet_name=f'{condition}_Unique_Down', index=False)

    # Gemeinsame Massen in mehreren Bedingungen
    common_masses = {}
    for mass in all_masses:
        present_in_conditions = [condition for condition in conditions[1:] if mass in venn_data[condition]]
        if len(present_in_conditions) > 1:
            key = ', '.join(present_in_conditions)
            if key not in common_masses:
                common_masses[key] = []
            common_masses[key].append(mass)

    for key, masses in common_masses.items():
        df_common = pd.DataFrame(masses, columns=[f'Common in {key}'])
        df_common.to_excel(writer, sheet_name=f'Common_{key}', index=False)

# Erstellen der Venn-Diagramme für erhöhte und erniedrigte Massen
conditions_labels = ['6-20%', '21-35%', 'over 35%']

# Erhöhte Massen
sets_up = [set(condition_results[condition]['up']) for condition in conditions[1:]]
venn3(sets_up, conditions_labels)
plt.title('Venn Diagram for Increased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Venn_Increased_Masses.png')
plt.show()

# Erniedrigte Massen
sets_down = [set(condition_results[condition]['down']) for condition in conditions[1:]]
venn3(sets_down, conditions_labels)
plt.title('Venn Diagram for Decreased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Venn_Decreased_Masses.png')
plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib_venn import venn3

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort2')

# Liste der Bedingungen und ihre Abkürzungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
abbreviations = {'Control': 'Ct', '6-20%': '620', '21-35%': '2135', 'over 35%': '35+'}

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]

# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zu allen anderen Bedingungen
mass_numbers = {mass: i+1 for i, mass in enumerate(filtered_masses)}

condition_results = {}

for i, condition1 in enumerate(conditions):
    for condition2 in conditions[i+1:]:
        p_values = []
        log2_fold_changes = []
        for mass in filtered_masses:
            values1 = df[df['Condition'] == condition1][mass]
            values2 = df[df['Condition'] == condition2][mass]
            if values1.mean() != 0 and values2.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                log2_fold_change = np.log2(values2.mean() / values1.mean())
                p_values.append(p_value)
                log2_fold_changes.append(log2_fold_change)
            else:
                p_values.append(np.nan)
                log2_fold_changes.append(np.nan)

        # Erstellen des Volcanoplots
        plt.figure(figsize=(10, 6))
        plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

        # Hervorheben von signifikanten Massen
        significant_up = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc > 3 and p_val < 0.05]
        significant_down = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc < -2.5 and p_val < 0.05]

        if significant_up:
            plt.scatter(*zip(*[(x, y) for x, y, _ in significant_up]), c='red', label='Upregulated')
        if significant_down:
            plt.scatter(*zip(*[(x, y) for x, y, _ in significant_down]), c='blue', label='Downregulated')

        # Hinzufügen von Nummerierung zu den Datenpunkten
        for x, y, num in significant_up + significant_down:
            plt.text(x, y, str(num), fontsize=8)

        plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
        plt.axvline(x=3, color='k', linestyle='--')
        plt.axvline(x=-2.5, color='k', linestyle='--')

        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 p-value')
        plt.title(f'Volcanoplot for {condition2} compared to {condition1} in Cohort2')

        plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Cohort2_{condition2}v{condition1}_volcanoplot.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Speichern der signifikanten Massen für jede Bedingung
        condition_results[f'{condition1}v{condition2}'] = {
            'up': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 3 and -np.log10(p_values[i]) > 1.3],
            'down': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -2.5 and -np.log10(p_values[i]) > 1.3]
        }

# Erstellen der Excel-Tabelle
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Cohort2_Venn_Results.xlsx') as writer:
    # Erstellen des Venn-Diagramms
    venn_data = {}
    for comparison in condition_results:
        venn_data[comparison] = set(condition_results[comparison]['up'] + condition_results[comparison]['down'])

    all_masses = set.union(*venn_data.values())
    venn_table = []

    for mass in all_masses:
        row = [mass]
        present_in_comparisons = [comparison for comparison in condition_results if mass in venn_data[comparison]]
        row.extend(present_in_comparisons)
        venn_table.append(row)

    # Adjust the number of columns to match the data
    max_columns = max(len(row) for row in venn_table)
    columns = ['Mass'] + [f'Comparison_{i}' for i in range(1, max_columns)]
    venn_df = pd.DataFrame(venn_table, columns=columns)
    venn_df.to_excel(writer, sheet_name='Venn_Diagram', index=False)

    # Erstellen der Tabelle für jede Bedingung und gemeinsame Massen
    for comparison in condition_results:
        unique_up = [mass for mass in condition_results[comparison]['up'] if sum([mass in condition_results[comp]['up'] for comp in condition_results]) == 1]
        unique_down = [mass for mass in condition_results[comparison]['down'] if sum([mass in condition_results[comp]['down'] for comp in condition_results]) == 1]

        df_up = pd.DataFrame(unique_up, columns=[f'{comparison}UniqueUp'])
        df_down = pd.DataFrame(unique_down, columns=[f'{comparison}UniqueDown'])
        
        df_up.to_excel(writer, sheet_name=f'{comparison}UniqueUp', index=False)
        df_down.to_excel(writer, sheet_name=f'{comparison}UniqueDown', index=False)

    # Gemeinsame Massen in mehreren Bedingungen
    common_masses = {}
    for mass in all_masses:
        present_in_comparisons = [comparison for comparison in condition_results if mass in venn_data[comparison]]
        if len(present_in_comparisons) > 1:
            for comp in present_in_comparisons:
                try:
                    key = ', '.join([abbreviations[comp.split('v')[0]] + 'v' + abbreviations[comp.split('v')[1]] for comp in present_in_comparisons])
                except KeyError as e:
                    print(f"KeyError: {e} in comparison {comp}")
                    continue
                if len(key) > 31:
                    key = key[:28] + '...'
                if key not in common_masses:
                    common_masses[key] = []
                common_masses[key].append(mass)

    for key, masses in common_masses.items():
        df_common = pd.DataFrame(masses, columns=[f'Common in {key}'])
        df_common.to_excel(writer, sheet_name=f'C{key}', index=False)

# Erstellen der Venn-Diagramme für erhöhte und erniedrigte Massen
conditions_labels = ['6-20%', '21-35%', 'over 35%']

# Erhöhte Massen
sets_up = [set(condition_results[comparison]['up']) for comparison in condition_results]
venn3(sets_up, conditions_labels)
plt.title('Venn Diagram for Increased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Venn_Increased_Masses.png')
plt.show()

# Erniedrigte Massen
sets_down = [set(condition_results[comparison]['down']) for comparison in condition_results]
venn3(sets_down, conditions_labels)
plt.title('Venn Diagram for Decreased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Venn_Decreased_Masses.png')
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib_venn import venn3

# Laden der Excel-Datei
df = pd.read_excel('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx', sheet_name='Cohort2')

# Liste der Bedingungen und ihre Abkürzungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
abbreviations = {'Control': 'Ct', '6-20%': '620', '21-35%': '2135', 'over 35%': '35+'}

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Spalten mit mehr als 50% 0-Werten ausschließen
columns_to_keep = [col for col in df.columns[3:] if calculate_zero_percentage(df[col]) <= 0.5]
df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

# Filter einbauen, um Massen außerhalb des Bereichs 255-1000 m/z auszuschließen
filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]

# Berechnen der p-Werte und log2-Fold-Changes für jede Bedingung im Vergleich zu allen anderen Bedingungen
mass_numbers = {mass: i+1 for i, mass in enumerate(filtered_masses)}

condition_results = {}

for i, condition1 in enumerate(conditions):
    for condition2 in conditions[i+1:]:
        p_values = []
        log2_fold_changes = []
        for mass in filtered_masses:
            values1 = df[df['Condition'] == condition1][mass]
            values2 = df[df['Condition'] == condition2][mass]
            if values1.mean() != 0 and values2.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
                t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                log2_fold_change = np.log2(values2.mean() / values1.mean())
                p_values.append(p_value)
                log2_fold_changes.append(log2_fold_change)
            else:
                p_values.append(np.nan)
                log2_fold_changes.append(np.nan)

        # Erstellen des Volcanoplots
        plt.figure(figsize=(10, 6))
        plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

        # Hervorheben von signifikanten Massen
        significant_up = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc > 3 and p_val < 0.05]
        significant_down = [(log2_fc, -np.log10(p_val), mass_numbers[filtered_masses[i]]) for i, (log2_fc, p_val) in enumerate(zip(log2_fold_changes, p_values)) if log2_fc < -2.5 and p_val < 0.05]

        if significant_up:
            plt.scatter(*zip(*[(x, y) for x, y, _ in significant_up]), c='red', label='Upregulated')
        if significant_down:
            plt.scatter(*zip(*[(x, y) for x, y, _ in significant_down]), c='blue', label='Downregulated')

        # Hinzufügen von Nummerierung zu den Datenpunkten
        for x, y, num in significant_up + significant_down:
            plt.text(x, y, str(num), fontsize=8)

        plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
        plt.axvline(x=3, color='k', linestyle='--')
        plt.axvline(x=-2.5, color='k', linestyle='--')

        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 p-value')
        plt.title(f'Volcanoplot for {condition2} compared to {condition1} in Cohort2')

        plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Cohort2_{condition2}v{condition1}_volcanoplot.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Speichern der signifikanten Massen für jede Bedingung
        condition_results[f'{condition1}v{condition2}'] = {
            'up': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] > 1 and -np.log10(p_values[i]) > 1.3],
            'down': [filtered_masses[i] for i in range(len(log2_fold_changes)) if log2_fold_changes[i] < -1 and -np.log10(p_values[i]) > 1.3]
        }

# Erstellen der Excel-Tabelle
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Cohort2_Venn_Results.xlsx') as writer:
    # Erstellen des Venn-Diagramms
    venn_data = {}
    for comparison in condition_results:
        venn_data[comparison] = set(condition_results[comparison]['up'] + condition_results[comparison]['down'])

    all_masses = set.union(*venn_data.values())
    venn_table = []

    for mass in all_masses:
        row = [mass]
        present_in_comparisons = [comparison for comparison in condition_results if mass in venn_data[comparison]]
        row.extend(present_in_comparisons)
        venn_table.append(row)

    # Adjust the number of columns to match the data
    max_columns = max(len(row) for row in venn_table)
    columns = ['Mass'] + [f'Comparison_{i}' for i in range(1, max_columns)]
    venn_df = pd.DataFrame(venn_table, columns=columns)
    venn_df.to_excel(writer, sheet_name='Venn_Diagram', index=False)

    # Erstellen der Tabelle für jede Bedingung und gemeinsame Massen
    for comparison in condition_results:
        unique_up = [mass for mass in condition_results[comparison]['up'] if sum([mass in condition_results[comp]['up'] for comp in condition_results]) == 1]
        unique_down = [mass for mass in condition_results[comparison]['down'] if sum([mass in condition_results[comp]['down'] for comp in condition_results]) == 1]

        df_up = pd.DataFrame(unique_up, columns=[f'{comparison}UniqueUp'])
        df_down = pd.DataFrame(unique_down, columns=[f'{comparison}UniqueDown'])
        
        df_up.to_excel(writer, sheet_name=f'{comparison}UniqueUp', index=False)
        df_down.to_excel(writer, sheet_name=f'{comparison}UniqueDown', index=False)

    # Gemeinsame Massen in mehreren Bedingungen
    common_masses = {}
    for mass in all_masses:
        present_in_comparisons = [comparison for comparison in condition_results if mass in venn_data[comparison]]
        if len(present_in_comparisons) > 1:
            valid_comparisons = []
            for comp in present_in_comparisons:
                try:
                    valid_comparisons.append(abbreviations[comp.split('v')[0]] + 'v' + abbreviations[comp.split('v')[1]])
                except KeyError as e:
                    print(f"KeyError: {e} in comparison {comp}")
                    continue
            key = ', '.join(valid_comparisons)
            if len(key) > 31:
                key = key[:28] + '...'
            if key not in common_masses:
                common_masses[key] = []
            common_masses[key].append(mass)

    for key, masses in common_masses.items():
        df_common = pd.DataFrame(masses, columns=[f'Common in {key}'])
        df_common.to_excel(writer, sheet_name=f'C{key}', index=False)

# Erstellen der Venn-Diagramme für erhöhte und erniedrigte Massen
conditions_labels = ['6-20%', '21-35%', 'over 35%']

# Erhöhte Massen
sets_up = [set(condition_results[comparison]['up']) for comparison in condition_results]
venn3([sets_up[0], sets_up[1], sets_up[2]], set_labels=conditions_labels)
plt.title('Venn Diagram for Increased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Venn_Increased_Masses.png')
plt.show()

# Erniedrigte Massen
sets_down = [set(condition_results[comparison]['down']) for comparison in condition_results]
venn3([sets_down[0], sets_down[1], sets_down[2]], set_labels=conditions_labels)
plt.title('Venn Diagram for Decreased Masses')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Venn_Decreased_Masses.png')
plt.show()


#alle conditions werden verglichen als venn und volcanoplot geplottet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pylab as plt
#from matplotlib_venn import venn3
from venny4py.venny4py import venny4py

# Funktion zum Berechnen des Prozentsatzes der 0-Werte in einer Spalte
def calculate_zero_percentage(column):
    return (column == 0).sum() / len(column)

# Daten einlesen
xls = pd.ExcelFile('D:/Arbeit/Figure Venndiagramm/p value daten neu2.xlsx')

# Liste der Bedingungen
conditions = ['Control', '6-20%', '21-35%', 'over 35%']
abbreviations = {'Control': 'Ct', '6-20%': '620', '21-35%': '2135', 'over 35%': '35+'}

# Liste der Excel-Seiten
sheets = ['Cohort2']

# Liste der auszuschließenden Massen oder Zahlen für jeden Volcanoplot einzeln machen
exclude_list = []  # Fügen Sie hier die Massen oder Zahlen ein, die Sie ausschließen möchten

unique_masses = {}
condition_results = {}

for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # Extrahieren der Massen aus der ersten Zeile
    masses = df.columns[3:]  # Annahme: Die Massen beginnen ab der vierten Spalte

    # Spalten mit mehr als 50% 0-Werten ausschließen
    columns_to_keep = [col for col in masses if calculate_zero_percentage(df[col]) <= 0.5]
    df = df[['Primary ID', 'Run', 'Condition'] + columns_to_keep]

    # Filter einbauen, um Massen außerhalb des Bereichs 250-1000 m/z auszuschließen
    filtered_masses = [mass for mass in df.columns[3:] if 250 <= float(mass) <= 1000]
    df = df[['Primary ID', 'Run', 'Condition'] + filtered_masses]
    
    for i, condition1 in enumerate(conditions):
        for condition2 in conditions[i+1:]:
            p_values = []
            log2_fold_changes = []
            for mass in filtered_masses:
                values1 = df[df['Condition'] == condition1][mass]
                values2 = df[df['Condition'] == condition2][mass]
                if values1.mean() != 0 and values2.mean() != 0:  # Sicherstellen, dass der Mittelwert nicht 0 ist
                    t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                    log2_fold_change = np.log2(values2.mean() / values1.mean())
                    p_values.append(p_value)
                    log2_fold_changes.append(log2_fold_change)
                else:
                    p_values.append(np.nan)
                    log2_fold_changes.append(np.nan)

            # Erstellen des Volcanoplots
            plt.figure(figsize=(10, 6))
            plt.scatter(log2_fold_changes, -np.log10(p_values), c='grey')

            # Hervorheben von signifikanten Massen
            significant_up = [(log2_fc, -np.log10(p_val), mass) for log2_fc, p_val, mass in zip(log2_fold_changes, p_values, filtered_masses) if log2_fc > 1 and p_val < 0.05]
            significant_down = [(log2_fc, -np.log10(p_val), mass) for log2_fc, p_val, mass in zip(log2_fold_changes, p_values, filtered_masses) if log2_fc < -1 and p_val < 0.05]

            if significant_up:
                plt.scatter(*zip(*[(x, y) for x, y, _ in significant_up]), c='red', label='Upregulated')
            if significant_down:
                plt.scatter(*zip(*[(x, y) for x, y, _ in significant_down]), c='blue', label='Downregulated')

            # Hinzufügen von Nummerierung zu den Datenpunkten
            for x, y, mass in significant_up + significant_down:
                plt.text(x, y, mass, fontsize=8)

            plt.axhline(y=-np.log10(0.05), color='k', linestyle='--')
            plt.axvline(x=1, color='k', linestyle='--')
            plt.axvline(x=-1, color='k', linestyle='--')

            plt.xlabel('Log2 Fold Change')
            plt.ylabel('-Log10 p-value')
            plt.title(f'Volcanoplot for {condition2} compared to {condition1} in {sheet}')

            plt.savefig(f'D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/{sheet}_{condition2}v{condition1}_volcanoplot.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Speichern der signifikanten Massen für jede Bedingung
            condition_results[f'{condition1}v{condition2}'] = {
                'up': [mass for log2_fc, p_val, mass in zip(log2_fold_changes, p_values, filtered_masses) if log2_fc > 2.5 and -np.log10(p_val) > 1.3],
                'down': [mass for log2_fc, p_val, mass in zip(log2_fold_changes, p_values, filtered_masses) if log2_fc < -2 and -np.log10(p_val) > 1.3]
            }

# Erstellen Sie eine Excel-Datei mit mehreren Tabs
with pd.ExcelWriter('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Cohort2_Venn_Results_New.xlsx') as writer:
    # Massen, die nur in einer Bedingung vorkommen
    for condition in conditions:
        unique_masses = set()
        for comparison, results in condition_results.items():
            if condition in comparison:
                unique_masses.update(results['up'])
                unique_masses.update(results['down'])
        
        # Überprüfen, ob die Masse nur in einer Bedingung vorkommt
        unique_masses = [mass for mass in unique_masses if sum([mass in results['up'] or mass in results['down'] for results in condition_results.values()]) == 1]
        
        if unique_masses:
            df_unique = pd.DataFrame(sorted(unique_masses, key=float), columns=[f'{abbreviations[condition]}_unique'])
            df_unique.to_excel(writer, sheet_name=f'{abbreviations[condition]}_unique', index=False)

    # Massen, die in mehreren Bedingungen vorkommen
    all_masses = set()
    for results in condition_results.values():
        all_masses.update(results['up'])
        all_masses.update(results['down'])

    for mass in all_masses:
        present_in_conditions = [abbreviations[condition] for condition in conditions if any(mass in results['up'] or mass in results['down'] for comparison, results in condition_results.items() if condition in comparison)]
        if len(present_in_conditions) > 1:
            df_shared = pd.DataFrame([mass], columns=[f'Shared_in_{"_".join(present_in_conditions)}'])
            df_shared.to_excel(writer, sheet_name=f'Shared_in_{"_".join(present_in_conditions)}', index=False)

# Erstellen Sie Venn-Diagramme für die Bedingungen
control_masses = set()
condition_6_20_masses = set()
condition_21_35_masses = set()
condition_over_35_masses = set()

for comparison, results in condition_results.items():
    if 'Control' in comparison:
        control_masses.update(results['up'])
        control_masses.update(results['down'])
    if '6-20%' in comparison:
        condition_6_20_masses.update(results['up'])
        condition_6_20_masses.update(results['down'])
    if '21-35%' in comparison:
        condition_21_35_masses.update(results['up'])
        condition_21_35_masses.update(results['down'])
    if 'over 35%' in comparison:
        condition_over_35_masses.update(results['up'])
        condition_over_35_masses.update(results['down'])

# Erstellen Sie das Venn-Diagramm mit venny4py
sets = {
    'Control': control_masses,
    '6-20%': condition_6_20_masses,
    '21-35%': condition_21_35_masses,
    'over 35%': condition_over_35_masses
}

venny4py(sets=sets)
plt.title('Venn Diagram of Cohort 2')
plt.savefig('D:/Arbeit/Figure Venndiagramm/Alle Conditons vergleich/Venn_Control_Cohort2.png')
plt.show()
