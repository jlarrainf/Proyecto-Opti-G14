from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# Para la creacion de graficos
import matplotlib.pyplot as plt


## TODO:    - Comentar el archivo 
#           - Explicar restricciones y variables nuevas
#           - Manejo de soluciones
#           - Conseguir valores para los nuevos parametros
#           - Justificar el importar matplotlib


# ============================== Conjuntos ==============================

n = 15    # Número de albergues
m = 30    # Número de días
pv = 4    # Número de lotes de alimentos

I_ = range(1, n + 1)                                                        # Albergues
T_ = range(1, m + 1)                                                        # Días
Recursos_basicos = ["Alimentos", "Agua"]                                    # Recursos básicos
R_ = range(1, len(Recursos_basicos) + 1)                                    # Recursos básicos
Recursos_operativos = ["Colchonetas", "Mantas", "Ropa", "Kit de higiene"]   # Recursos operativos
O_ = range(1, len(Recursos_operativos) + 1)                                 # Recursos operativos
P_ = range(1, pv + 1)                                                       # Lotes de alimentos


# =========================== Manejo de Datos ===========================

# Se extraen los datos del mismo archivo de la entrega
excel_file = "datos.xlsx"

# Creamos un DataFrame de pandas para cada pagina del excel
sheets = pd.ExcelFile(excel_file).sheet_names

Param_albergue = pd.read_excel(excel_file, sheet_name=sheets[0])
R_df = pd.read_excel(excel_file, sheet_name=sheets[1])
o_df = pd.read_excel(excel_file, sheet_name=sheets[2])
fv = pd.read_excel(excel_file, sheet_name=sheets[3])
ct = pd.read_excel(excel_file, sheet_name=sheets[4])
Escalares = pd.read_excel(excel_file, sheet_name=sheets[5])

# Arreglamos errores de posicionamiento dentro del excel y 
# separamos los datos de paginas con mas de una variable

CT = pd.Series(Param_albergue.iloc[2:,2])
CT.index -= 1

A = pd.Series(Param_albergue.iloc[2:,3])
A.index -= 1

C = pd.Series(Param_albergue.iloc[2:,4])
C.index -= 1

MP = pd.Series(Param_albergue.iloc[2:,5])
MP.index -= 1

RM = pd.Series(R_df.iloc[2:,4])
RM.index -= 1

CR = pd.Series(R_df.iloc[2:,5])
CR.index -= 1

B = pd.Series(R_df.iloc[2:,6])
B.index -= 1

RMR = pd.Series(o_df.iloc[2:,4])
RMR.index -= 1

CO = pd.Series(o_df.iloc[2:,5])
CO.index -= 1

FV = pd.DataFrame(fv.iloc[3:,3:])
FV.index -= 2
FV.columns = P_

CT2 = pd.DataFrame(ct.iloc[2:,2:])
CT2.columns = I_
CT2.index -= 1

# En el siguiente paso transformamos estos DataFrames en diccionarios para mayor facilidad de uso en el modelo


# ============================= Parámetros =============================

RMO = {(o): RMR[o] for o in O_}                     # Requerimiento mínimo de recursos operativos
RM = {(r): RM[r] for r in R_}                       # Requerimiento mínimo de recursos básicos
FV = {(r, p): FV[p][r] for r in R_ for p in P_}     # Fecha de vencimiento
C = {(i): C[i] for i in I_}                         # Costos de habilitación
CR = {(r): CR[r] for r in R_}                       # Costos de recursos básicos
CO =  {(o): CO[o] for o in O_}                      # Costos de recursos operativos
CT = {(i): CT[i] for i in I_}                       # Costos de transporte
CT2 = {(i,j): CT2[j][i] for i in I_ for j in I_}    # Costos de transporte entre albergues
B = {(r): B[r] for r in R_}                         # Presupuesto por recurso
P = Escalares.iloc[2, 2]                            # Presupuesto total
A = {(i): A[i] for i in I_}                         # Almacenamiento en bodega
MP = {(i): MP[i] for i in I_}                       # Maximo de personas en albergue

## TODO: Conseguir datos del excel
D_t = {(t): 3600 for t in T_}                         # Demanda diaria de alojamiento
tau = 3                                            # Tiempo de permanencia minima en un albergue
DR = {(r,p): 50 for r in R_ for p in P_}           # Cantidad inicial de donaciones


D_r = {(r): sum(DR[r,p] for p in P_) for r in R_}   # Cantidad inicial total de donaciones del recurso r
DG = sum(RM[r] for r in R_)                         # Cantidad de desechos generados

# Iniciacion del modelo
modelo = Model()
modelo.setParam("TimeLimit", 5 * 60)  # Seteamos un tiempo limite de 5 minutos

# ============================= Variables =============================

x = modelo.addVars(I_, T_,  vtype=GRB.SEMIINT, name="x")                # Personas en cada albergue
y = modelo.addVars(I_, T_, vtype=GRB.BINARY, name="y")                  # Habilitación de albergues
z = modelo.addVars(I_, T_, vtype=GRB.INTEGER, name="z")                 # Flujo de personas
z_plus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_plus")       # Personas que ingresan al albergue
z_minus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_minus")     # Personas que se retiran del albergue
g = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="g")         # Recursos asignados
r = modelo.addVars(O_, I_, T_, vtype=GRB.SEMIINT, name="r")             # Recursos operativos
h = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="h")         # Recursos básicos
b = modelo.addVars(R_, I_, T_, vtype=GRB.SEMIINT, name="b")             # Recursos asignados a cada albergue
T = modelo.addVars(R_, I_, I_, T_, vtype=GRB.SEMIINT, name="T")         # Recursos transferidos
I = modelo.addVars(R_, P_, T_, vtype=GRB.SEMIINT, name="I")             # Recursos asignados
I_A = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="I_A")     # Recursos asignados adicionales
a = modelo.addVars(T_, vtype=GRB.SEMIINT, name="a")                     # Personas sin albergue por dia
CN = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="CN")       # Recursos consumidos


# Actualizamos el modelo para guardar las variables
modelo.update()

# =========================== Restricciones ===========================

# R1: Catidad inicial y maxima de personas
modelo.addConstrs(x[i, 1] == z[i, 1] for i in I_)
modelo.addConstrs(x[i, t] <= MP[i] * y[i, t] for i in I_ for t in T_)

# R2: Control de flujo de personas
modelo.addConstrs(x[i,1] == x[i, t-1] + z[i, 1] for i in I_ for t in range(2, m + 1))
modelo.addConstrs(x[i,1] >= x[i, t-1] - z[i, 1] for i in I_ for t in range(2, m + 1))

# R3: Control de recursos operativos
modelo.addConstrs(x[i, t] * RMR[o] <= r[o, i, t] for o in O_ for i in I_ for t in T_)

# R4: Control de recursos básicos
modelo.addConstrs(x[i, t] * RM[r] <= quicksum(g[r, p, i, t] + h[r, p, i, t] for p in P_) for r in R_ for i in I_ for t in T_)

# R5: Actualizacion diaria de recursos y donaciones
modelo.addConstrs(I[r, p, 1] == DR[r, p] for r in R_ for p in P_)
modelo.addConstrs(I[r, p, t + 1] == I[r, p, t] - quicksum(h[r, p, i, t] for i in I_) for r in R_ for p in P_ for t in range(1, m))

# R6: Prohibicion de asignacion de recursos y donaciones vencidas
modelo.addConstrs(quicksum(h[r, p, i, t] + g[r, p, i, t] for i in I_) == 0 for r in R_ for p in P_ for t in T_ if t > FV[r,p])

# R7: Desechos generados
modelo.addConstrs(b[r, i, t] == I_A[r, p, i, t] for r in R_ for p in P_ for i in I_ for t in T_ if t > FV[r,p])
modelo.addConstrs(b[r, i, t] == 0 for r in R_ for p in P_ for i in I_ for t in T_ if t <= FV[r,p])

# R8: Inventario inicial
modelo.addConstrs(I_A[r, p, i, 1] == 0 for r in R_ for p in P_ for i in I_)

# R9: Balance de inventario
modelo.addConstrs(I_A[r, p, i, t]  == I_A[r, p, i, t-1] + g[r, p, i, t-1] + h[r, p, i, t-1] + 
                  quicksum(T[r,j,i,t-1] - T[r,i,j,t-1] for j in I_ if j != i) - CN[r, p, i, t-1] for r in R_ for p in P_ for i in I_  for t in range(2,m+1))

# R10: Consumo de recursos
modelo.addConstrs(quicksum(CN[r, p, i, t] for p in P_) == RM[r] * x[i, t] for r in R_  for i in I_ for t in T_)

# R11: Limite de inventario
modelo.addConstrs(CN[r, p, i, t] <= I_A[r, p, i, t] for r in R_ for p in P_ for i in I_ for t in T_)

# R12: Limite de donaciones asignadas
modelo.addConstrs(quicksum(h[r, p, i, t] for p in P_ for i in I_ for t in T_) <= D_r[r] for r in R_)

# R13: Capacidad maxima de almacenamiento
modelo.addConstrs(quicksum( I_A[r, p, i, t] + g[r, p, i, t] + h[r, p, i, t] + 
                            quicksum(T[r,i,j,t] for j in I_ if j != i) for r in R_ for p in P_ ) <= A[i]
                            for i in I_ for t in T_)

# R14: Los recursos operativos son los mismos todos los dias
modelo.addConstrs(r[o, i, t] == r[o, i, t-1] for o in O_ for i in I_ for t in range(2, m+1))

# R15: Capacidad maxima almacenamiento
modelo.addConstrs(quicksum(r[o, i, t] for o in O_) <= A[i] for i in I_ for t in T_)

# R16: Permanencia minima en el albergue
modelo.addConstrs(z_minus[i, t] == 0 for i in I_ for t in range(1, tau))
modelo.addConstrs(z_minus[i, t] <= quicksum(z_plus[i, k] for k in range(t - tau + 1, t + 1)) for i in I_ for t in range(tau, m + 1))

# R17: Flujo neto de personas en el albergue
modelo.addConstrs(z[i, t] == z_plus[i, t] - z_minus[i, t] for i in I_ for t in T_)

# R18: Personas sin albergue
modelo.addConstrs(a[t] == D_t[t] - quicksum(x[i, t] for i in I_) for t in T_)


# Funcion Objetivo: Minimizar la cantidad de personas sin albergue
objetivo = quicksum(a[t] for t in T_)
modelo.setObjective(objetivo, GRB.MINIMIZE)

# Optimizar el modelo
modelo.optimize()

# =========================== Manejo de soluciones ============================

### TODO: Toda esta parte la verdad

# El output en consola de este programa son los datos sobre el modelo, los datos sobre las variables
# y su respectiva interpretacion seran guardados en la carpeta Resultados


print("\n\n====================== Caracteristicas del modelo ===========================")
# Esto se imprime en consola

### Imprimir gap del modelo
gap = modelo.getAttr("MIPGap")
print(f"gap: {gap}")

### Numero de variables
n_vars = len(modelo.getVars())
print(f"Numero de Variables: {n_vars}")

### Numero de restricciones
n_res = len(modelo.getConstrs())
print(f"Numero de Restricciones: {n_res}")


### Numero de Restricciones Activas
n_res_activas = 0       # Contamos el numero de restricciones con slack 0
for constr in modelo.getConstrs():
    if constr.getAttr("slack") == 0:
        n_res_activas += 1

print(f"Numero Restricciones Activas: {n_res_activas}")


### Tiempo de ejecución
print(f"Tiempo de ejecución: {modelo.Runtime} segundos")

### Valor optimo del modelo
print(f"Valor Optimo: {modelo.objVal}")



# =================== Guardar valores de las variables ======================

# Esto se guarda en Resultados/Vars

# Listas con las variables indices y nombres para guardarlos en forma de DataFrame
Variables = [x,y,z,z_plus,z_minus,g,r,h,b,T,I,I_A,a,CN]
VarNames = ["x","y","z","z_plus","z_minus","g","r","h","b","T","I","I_A","a","CN"]
Indexes = [["i","t"], ["i","t"], ["i","t"], ["i","t"], ["i","t"], 
           ["r", "p", "i","t"], ["o", "i","t"], ["r", "p", "i","t"], 
           ["r", "i","t"], ["r", "i", "i'","t"], ["r", "p", "t"], ["r", "p", "i","t"], ["t"], ["r", "p", "i","t"]]


# Por cada variable, se crea un DataFrame entregando los valores de la variable por cada combinacion de indices
# Estos se guardan en forma de archivo csv en la carpeta Resultados/Vars/

for v, name, indx in zip(Variables, VarNames, Indexes):
    lst = []
    
    for tupla_indices in v:
        if name == "a":     # Debido a que tiene un solo indice este se trata de un int y no de una tupla como los otros
            indice = tupla_indices
            data = (tupla_indices, v[tupla_indices].x)
        else:
            data = tupla_indices + (v[tupla_indices].x,)
        lst.append(data)
    
    indx.append(name)
    columns = indx
    df = pd.DataFrame(lst, columns=columns)
    
    df.to_csv(f"Resultados/Vars/{name}.csv", columns=columns)



# ======================= Creacion de Graficos ============================

# Esto se guarda en Resultados/Graficos
# Para esto se usa el modulo matplotlib

### Cantidad de personas refugiadas en un albergue por dia
plt.style.use('ggplot')

y_axis = [sum(x[i,t].X for i in I_) for t in T_]    # Cantidad total de personas refugiadas en dia t
x_axis = T_

# Ajustamos el tamaño del grafico resultante
fig, ax = plt.subplots(figsize=(8,6))

plt.title("Cantidad de personas refugiadas por dia")
ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(1, m+1), xticks=np.arange(1, m+1),
       ylim=(0, 100), yticks=np.arange(0, 101, 10))

plt.savefig("Resultados/Graficos/Cantidad de personas refugiadas por dia")



### Numero de albergues habilitados por dia

y_axis = [sum(y[i,t].X for i in I_) for t in T_]    # Cantidad total de personas refugiadas en dia t
x_axis = T_


# Ajustamos el tamaño del grafico resultante
fig, ax = plt.subplots(figsize=(12,8))

plt.title("Numero de albergues habilitados por dia")
ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=2)

ax.set(xlim=(0, m+1), xticks=np.arange(1, m+1),
       ylim=(0, n+1), yticks=np.arange(1, n+1))

plt.savefig("Resultados/Graficos/Numero de albergues habilitados por dia")



# TODO: Interpretacion de los resultados
"""
if modelo.status == GRB.OPTIMAL:
    print(f"Valor objetivo: {modelo.objVal}")

    # Imprimir los valores de las variables de decisión
    for i,t in x:
        print(f"Se encuentran {x[i,t].X} personas en el albergue {i} el dia {t}")
"""