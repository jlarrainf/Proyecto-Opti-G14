from gurobipy import Model, GRB, quicksum
import pandas as pd
from random import randint


n = 10
m = 30
pv = 10

# Conjuntos
I_ = range(1, n+1)
T_ = range(1, m+1)

Recursos_basicos = ["Alimentos", "Agua", "Insumos medicos"]
R_ = range(1, len(Recursos_basicos) + 1)

Recursos_operativos = ["Colchonetas", "Mantas","Ropa", "Kit de higiene"]
O_ = range(1, len(Recursos_operativos) + 1)

P_ = range(1, pv+1)


# Parametros

# Estos son solo para probar si funciona por ahora
# cuando consigamos los reales nos peleamos con pandas

RMO = {(o): randint(1,10) for o in O_}
RM = {(r): randint(1,10) for r in R_}
FV = {(r,p): randint(1,10) for r in R_ for p in P_}
DG = {(i,t): randint(1,10) for i in I_ for t in T_}
C = {(i): randint(1,10) for i in I_}
CR = {(r): randint(1,10) for r in R_}
CO = {(o): randint(1,10) for o in O_}
CT = {(i): randint(1,10) for i in I_}
CT2 = {(i,j): randint(1,10) for i in I_ for j in I_}
B = {(r): randint(1,10) for r in R_}
CN = {(r,p,i,t): randint(1,10) for r in R_ for p in P_ for i in I_ for t in T_}
P = randint(1,10)
A = {(i): randint(1,10) for i in I_}
MP = {(i): randint(1,10) for i in I_}
a = {(t): randint(1,10) for t in T_}

modelo = Model()
modelo.setParam("TimeLimit", 5*60) #Limite 5 minutos
# modelo.setParam("LogFile", "log.txt")

# Variables
x = modelo.addVars(I_,T_, vtype = GRB.SEMIINT, name = "x")
z = modelo.addVars(I_,T_, vtype = GRB.SEMIINT, name = "z")
y = modelo.addVars(I_,T_, vtype = GRB.BINARY, name = "y")
g = modelo.addVars(R_, P_, I_,T_, vtype = GRB.SEMIINT, name = "g")
r = modelo.addVars(O_,I_,T_, vtype = GRB.SEMIINT, name = "r")
h = modelo.addVars(R_, P_, I_,T_, vtype = GRB.SEMIINT, name = "h")
b = modelo.addVars(R_, I_,T_, vtype = GRB.SEMIINT, name = "b")
T = modelo.addVars(R_,I_,I_, vtype = GRB.SEMIINT, name = "T")
I = modelo.addVars(R_, P_, T_, vtype = GRB.SEMIINT, name = "I")
I_A = modelo.addVars(R_, P_, I_, T_, vtype = GRB.SEMIINT, name = "I_A")

modelo.update()

# Restricciones

# R3
modelo.addConstrs(x[i,t] == x[i,t] + z[i,t] for i in I_ for t in range(2,m))

# R4
modelo.addConstrs(x[i,t] * RMO[o] <= r[o,i,t] for o in O_ for i in I_ for t in T_)

# R5
modelo.addConstrs(x[i,t] * RM[r] <= g[r,p,i,t] + h[r,p,i,t] for r in R_ for p in P_ for i in I_ for t in T_)

# R7
modelo.addConstrs(h[r,p,i,t] == 0  for r in R_ for i in I_ for p in P_ for t in range(FV[r,p], m+1))

# Objetivo: Minimizar costos
objetivo = quicksum(a[t] for t in T_)
modelo.setObjective(objetivo, GRB.MINIMIZE)

# Optimizar el modelo
modelo.optimize()

# Procesar resultados
if modelo.status == GRB.OPTIMAL:
    print("Valor objetivo:", modelo.objVal)
    resultados = []
    for i in I_:
        for t in T_:
            if x[i, t].X > 0:  # Solo mostrar resultados relevantes
                resultados.append((i, t, x[i, t].X))

    # Convertir resultados a DataFrame para una mejor visualización
    df_resultados = pd.DataFrame(resultados, columns=["Albergue", "Día", "Cantidad de Personas"])
    print(df_resultados)
else:
    print("No se encontró solución óptima.")