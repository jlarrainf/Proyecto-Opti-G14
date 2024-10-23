from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd


# Parámetros
n = 15    # Número de albergues
m = 30    # Número de días
pv = 4    # Número de lotes de alimentos

# Conjuntos
I_ = range(1, n + 1)                                                        # Albergues
T_ = range(1, m + 1)                                                        # Días
Recursos_basicos = ["Alimentos", "Agua"]                                    # Recursos básicos
R_ = range(1, len(Recursos_basicos) + 1)                                    # Recursos básicos
Recursos_operativos = ["Colchonetas", "Mantas", "Ropa", "Kit de higiene"]   # Recursos operativos
O_ = range(1, len(Recursos_operativos) + 1)                                 # Recursos operativos
P_ = range(1, pv + 1)                                                       # Lotes de alimentos


# Leemos los datos del excel
excel_file = "datos.xlsx"

sheets = pd.ExcelFile(excel_file).sheet_names

Param_albergue = pd.read_excel(excel_file, sheet_name=sheets[0])

R_df = pd.read_excel(excel_file, sheet_name=sheets[1])

o_df = pd.read_excel(excel_file, sheet_name=sheets[2])

fv = pd.read_excel(excel_file, sheet_name=sheets[3])

ct = pd.read_excel(excel_file, sheet_name=sheets[4])

Escalares = pd.read_excel(excel_file, sheet_name=sheets[5])


# Parámetros

RMR =  {(o): o_df.iloc[o+1, 4] for o in O_}                     # Requerimiento mínimo de recursos operativos
RM = {(r): R_df.iloc[r+1, 4] for r in R_}                       # Requerimiento mínimo de recursos básicos
FV = {(r, p): int(fv.iloc[r+2, p+2]) for r in R_ for p in P_}   # Fecha de vencimiento
D = Escalares.iloc[3, 2]                                        # Desechos generados
C = {(i): Param_albergue.iloc[i+1, 4] for i in I_}              # Costos de habilitación
CR = {(r): R_df.iloc[r+1, 5] for r in R_}                       # Costos de recursos básicos
CO =  {(o): o_df.iloc[o+1, 5] for o in O_}                      # Costos de recursos operativos
CT = {(i): Param_albergue.iloc[i+1,2] for i in I_}              # Costos de transporte
CT2 = {(i,j): ct.iloc[j+1,i+1] for i in I_ for j in I_}         # Costos de transporte entre albergues
B = {(r): R_df.iloc[r+1, 6] for r in R_}                        # Presupuesto por recurso
P = Escalares.iloc[2, 2]                                        # Presupuesto total
A = {(i): Param_albergue.iloc[i+1,3] for i in I_}               # Almacenamiento en bodega
MP = {(i): Param_albergue.iloc[i+1,5] for i in I_}              # Maximo de personas en albergue

modelo = Model()
modelo.setParam("TimeLimit", 5 * 60)  #Limite 5 minutos
# modelo.setParam("LogFile", "log.txt")

# Variables
x = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="x")               # Personas en cada albergue
z = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z")               # Flujo de personas
y = modelo.addVars(I_, T_, vtype=GRB.BINARY, name="y")                # Habilitación de albergues
g = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="g")       # Recursos asignados
r = modelo.addVars(O_, I_, T_, vtype=GRB.SEMIINT, name="r")           # Recursos operativos
h = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="h")       # Recursos básicos
b = modelo.addVars(R_, I_, T_, vtype=GRB.SEMIINT, name="b")           # Recursos asignados a cada albergue
T = modelo.addVars(R_, I_, I_, T_, vtype=GRB.SEMIINT, name="T")       # Recursos transferidos
I = modelo.addVars(R_, P_, T_, vtype=GRB.SEMIINT, name="I")           # Recursos asignados
I_A = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="I_A")   # Recursos asignados adicionales
DG = modelo.addVar(vtype=GRB.SEMIINT, name="DG")

a = modelo.addVars(T_, vtype=GRB.SEMIINT, name="a")
CN = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="CN")


modelo.update()

# Restricciones

# R1:
modelo.addConstrs(x[i, 1] == z[i, 1] for i in I_)
modelo.addConstrs(x[i, t] <= MP[i] * y[i, t] for i in I_ for t in T_)

# R2: Control de flujo de personas
modelo.addConstrs(x[i,1] == x[i, t-1] + z[i, 1] for i in I_ for t in range(2, m + 1))

# R3: Control de habilitación de albergues
modelo.addConstrs(x[i, t] * RMR[o] <= r[o, i, t] for o in O_ for i in I_ for t in T_)

# R4: Control de recursos básicos
modelo.addConstrs(x[i, t] * RM[r] <= g[r, p, i, t] + h[r, p, i, t] for p in P_ for r in R_ for i in I_ for t in T_)

# R5: Control de recursos básicos
modelo.addConstrs(I[r, p, t] == I[r, p, t] - quicksum(g[r, p, i, t] + h[r, p, i, t] for i in I_) for r in R_ for p in P_ for t in T_)

# R6: Control de recursos asignados
modelo.addConstrs(h[r, p, i, t] == 0 for r in R_ for i in I_ for p in P_ for t in range(FV[r, p], m + 1))
modelo.addConstrs(g[r, p, i, t] == 0 for r in R_ for i in I_ for p in P_ for t in range(FV[r, p], m + 1))

# R7: Eliminacion de inventario vencido
modelo.addConstrs(I[r, p, t] == 0 for r in R_ for p in P_ for t in range(FV[r, p], m + 1))

# R8: Eliminacion de inventario vencido
modelo.addConstr(DG == quicksum(RM[r] for r in R_))

# R9: Cantidad de desechos generados
modelo.addConstrs(b[r, i, t] == g[r, p, i, t] + h[r, p, i, t]  for i in I_ for r in R_  for p in P_ for t in range(FV[r, p], m + 1))

# R10: Alimentos se consumen en orden segun su fecha de vencimiento
modelo.addConstrs(quicksum(g[r, p, i, t] + h[r, p, i, t] for t in T_) == 
                  quicksum(g[r, p + 1, i, t] + h[r, p + 1, i, t] for t in T_) for r in R_ for i in I_ for p in range(1, pv))

# R11
modelo.addConstrs(I_A[r, p, i, 1] == 0  for i in I_ for p in P_ for r in R_)

# R12
modelo.addConstrs(CN[r, p, i, t] == RM[r] * x[i, t] for r in R_ for t in T_ for i in I_ for p in P_)

# R13
modelo.addConstrs(I_A[r, p, i, t+1]  == I_A[r, p, i, t] + g[r, p, i, t] + h[r, p, i, t] + 
                  quicksum(T[r,i,j,t] - T[r,j,i,t] for j in I_) for r in R_ for t in range(1,m) for i in I_ for p in P_)

# R14
modelo.addConstrs(CN[r,p,i,t] <= I_A[r,p,i,t] for r in R_ for t in T_ for i in I_ for p in P_)

# R15
modelo.addConstrs(quicksum(quicksum(g[r, p, i, t] + h[r, p, i, t] for p in P_) 
                           + quicksum(T[r,i,j,t] for j in I_) 
                           for r in R_) <= A[i]
                           for t in T_ for i in I_)



# R16:
modelo.addConstr(quicksum(C[i] * y[i, t] + CT[i] * 
                                    quicksum(g[r,p,i,t] + h[r,p,i,t] for p in P_ for r in R_) +
                                    quicksum(CT2[i,j] * T[r,i,j,t] for r in R_ for i in I_ for j in I_) +
                                    quicksum(CO[o] * r[o,i,t] for o in O_) +
                                    quicksum(B[r] * b[r,i,t] for r in R_)
                                    for t in T_ for i in I_) <= P)

# R17:
modelo.addConstrs(a[t] == D - quicksum(x[i, t] for i in I_) for t in T_)

# Objetivo: Minimizar costos
objetivo = quicksum(a[t] for t in T_)
modelo.setObjective(objetivo, GRB.MINIMIZE)

# Optimizar el modelo
modelo.optimize()


# Manejo de soluciones
if modelo.status == GRB.OPTIMAL:
    print(f"Valor objetivo: {modelo.objVal}")

    # Imprimir los valores de las variables de decisión
    for i in I_:
        for t in T_:
            if x[i, t].x > 0:  # Solo mostrar si la variable es mayor que cero
                print(f"Albergue {i}, Día {t}: {x[i, t].x} personas")
"""
    # Imprimir holguras de las restricciones
    for constr in modelo.getConstrs():
        print(constr, constr.getAttr("slack"))

    # Imprimir valores de variables duales
    for constr in modelo.getConstrs():
        print(f"Variable dual para {constr.constrName}: {0}")
"""
# Tiempo de ejecución
print(f"Tiempo de ejecución: {modelo.Runtime} segundos")