from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

## TODO:    - Comentar el archivo 
#           - Explicar restricciones y variables nuevas
#           - Manejo de soluciones
#           - Conseguir valores para los nuevos parametros


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

# Abrimos el archivo de datos de la entrega
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

FV = pd.DataFrame(fv.iloc[3:,3:], dtype=np.int64)
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

D_t = {(t): 1 for t in T_}              ## TODO: Conseguir datos del excel
tau = 0                                 ## TODO: Escalar
DR = {(r,p): 1 for r in R_ for p in P_} ## TODO: Conseguir datos

D_r = {(r): sum(DR[r,p] for p in P_) for r in R_}
DG = sum(RM[r] for r in R_)



# Iniciacion del modelo
modelo = Model()
modelo.setParam("TimeLimit", 5 * 60)  #Limite 5 minutos

# ============================= Variables =============================

x = modelo.addVars(I_, T_,  vtype=GRB.SEMIINT, name="x")                # Personas en cada albergue
y = modelo.addVars(I_, T_, vtype=GRB.BINARY, name="y")                  # Habilitación de albergues
z = modelo.addVars(I_, T_, vtype=GRB.INT, name="z")                     # Flujo de personas
z_plus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_plus")
z_minus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_minus")                
g = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="g")         # Recursos asignados
r = modelo.addVars(O_, I_, T_, vtype=GRB.SEMIINT, name="r")             # Recursos operativos
h = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="h")         # Recursos básicos
b = modelo.addVars(R_, I_, T_, vtype=GRB.SEMIINT, name="b")             # Recursos asignados a cada albergue
T = modelo.addVars(R_, I_, I_, T_, vtype=GRB.SEMIINT, name="T")         # Recursos transferidos
I = modelo.addVars(R_, P_, T_, vtype=GRB.SEMIINT, name="I")             # Recursos asignados
I_A = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="I_A")     # Recursos asignados adicionales
a = modelo.addVars(T_, vtype=GRB.SEMIINT, name="a")
CN = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="CN")


# Actualizamos el modelo
modelo.update()

# =========================== Restricciones ===========================

# R1:
modelo.addConstrs(x[i, 1] == z[i, 1] for i in I_)
modelo.addConstrs(x[i, t] <= MP[i] * y[i, t] for i in I_ for t in T_)

# R2: Control de flujo de personas
modelo.addConstrs(x[i,1] == x[i, t-1] + z[i, 1] for i in I_ for t in range(2, m + 1))
modelo.addConstrs(x[i,1] >= x[i, t-1] - z[i, 1] for i in I_ for t in range(2, m + 1))

# R3: Control de habilitación de albergues
modelo.addConstrs(x[i, t] * RMR[o] <= r[o, i, t] for o in O_ for i in I_ for t in T_)

# R4: Control de recursos básicos
modelo.addConstrs(x[i, t] * RM[r] <= quicksum(g[r, p, i, t] + h[r, p, i, t] for p in P_) for r in R_ for i in I_ for t in T_)

# R5: Control de recursos básicos
modelo.addConstrs(I[r, p, 1] == DR[r, p] for r in R_ for p in P_)
modelo.addConstrs(I[r, p, t + 1] == I[r, p, t] - quicksum(h[r, p, i, t] for i in I_) for r in R_ for p in P_ for t in range(1, m))

# R6: Control de recursos asignados
modelo.addConstrs(quicksum(h[r, p, i, t] + g[r, p, i, t] for i in I_) == 0 for r in R_ for p in P_ for t in range(FV[r, p] + 1, m + 1))

# R7:
modelo.addConstrs(b[r, i, t] == I_A[r, p, i, t] for i in I_ for r in R_ for p in P_ for t in range(FV[r, p] + 1, m + 1))
modelo.addConstrs(b[r, i, t] == 0 for i in I_ for r in R_ for p in P_ for t in range(1, FV[r, p] + 1))

# R8:
modelo.addConstr(I_A[r, p, i, 1] == 0 for r in R_ for p in P_ for i in I_)

# R9:
modelo.addConstrs(I_A[r, p, i, t]  == I_A[r, p, i, t-1] + g[r, p, i, t-1] + h[r, p, i, t-1] + 
                  quicksum(T[r,j,i,t-1] - T[r,i,j,t-1] for j in I_ if j != i) - CN[r, p, i, t-1] for r in R_ for t in range(2,m) for i in I_ for p in P_)

# R10
modelo.addConstrs(quicksum(CN[r, p, i, t] for p in P_) == RM[r] * x[i, t] for r in R_ for t in T_ for i in I_)

# R11
modelo.addConstrs(CN[r, p, i, t] <= I_A[r, p, i, t] for r in R_ for t in T_ for i in I_ for p in P_)

# R12
modelo.addConstrs(quicksum(h[r, p, i, t] for p in P_ for i in I_ for t in T_) <= D_r[r] for r in R_)


# R13
modelo.addConstrs(quicksum( I_A[r, p, i, t] + g[r, p, i, t] + h[r, p, i, t] + 
                            quicksum(T[r,i,j,t] for j in I_ if j != i) for p in P_ for r in R_) <= A[i]
                            for t in T_ for i in I_)

# R14
modelo.addConstrs(r[o, i, t] == r[o, i, t-1] for o in O_ for i in I_ for t in range(2, m+1))

# R15:
modelo.addConstrs(quicksum(r[o, i, t] for o in O_) <= A[i] for i in I_ for t in T_)

# R16:
modelo.addConstrs(z_minus[i, t] == 0 for i in I_ for t in range(1, tau))
modelo.addConstrs(z_minus[i, t] <= quicksum(z_plus[i, k] for k in range(t - tau + 1, t + 1)) for i in I_ for t in range(tau, m + 1))

# R17:
modelo.addConstrs(z[i, t] == z_plus[i, t] - z_minus[i, t] for i in I_ for t in T_)

# R18:
modelo.addConstrs(a[t] == D_t[t] - quicksum(x[i, t] for i in I_) for t in T_)


# Funcion Objetivo: Minimizar costos
objetivo = quicksum(a[t] for t in T_)
modelo.setObjective(objetivo, GRB.MINIMIZE)

# Optimizar el modelo
modelo.optimize()

# =========================== Manejo de soluciones ============================

### TODO: Toda esta parte la verdad

print("\n\n====================== Caracteristicas del modelo ===========================")


gap = modelo.getAttr("MIPGap")
n_res = len(modelo.getConstrs())
n_vars = len(modelo.getVars())


print(f"gap: {gap}")
print(f"Numero de Variables: {n_vars}")
print(f"Numero de Restricciones: {n_res}")

# Tiempo de ejecución
print(f"Tiempo de ejecución: {modelo.Runtime} segundos")



if modelo.status == GRB.OPTIMAL:
    print(f"Valor objetivo: {modelo.objVal}")

    # Imprimir los valores de las variables de decisión
    for i,t in x:
        print(f"Se encuentran {x[i,t].X} personas en el albergue {i} el dia {t}")


"""

    # Imprimir holguras de las restricciones
    for constr in modelo.getConstrs():
        print(constr, constr.getAttr("slack"))

    # Imprimir valores de variables duales
    for constr in modelo.getConstrs():
        print(f"Variable dual para {constr.constrName}: {0}")
"""