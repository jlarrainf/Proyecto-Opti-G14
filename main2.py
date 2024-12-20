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
#presupuesto por albergue:
pi = 1000000



I_ = range(1, n + 1)                                                        # Albergues
T_ = range(1, m + 1)                                                        # Días
Recursos_basicos = ["Alimentos", "Agua"]                                    # Recursos básicos
R_ = range(1, len(Recursos_basicos) + 1)                                    # Recursos básicos
Recursos_operativos = ["Colchonetas", "Mantas", "Ropa", "Kit de higiene"]   # Recursos operativos
O_ = range(1, len(Recursos_operativos) + 1)                                 # Recursos operativos
P_ = range(1, pv + 1)                                                       # Lotes de alimentos





d = 100000                             # Número de personas totales


D_ = {t: 25 for t in range (1, m + 1)}    # Diccionario t: D[t] para la demanda fija diaria de personas



# =========================== Manejo de Datos ===========================

# Se extraen los datos del mismo archivo de la entrega
#excel_file = "datos.xlsx"
excel_file = "datos_mod.xlsx"

# Creamos un DataFrame de pandas para cada pagina del excel
sheets = pd.ExcelFile(excel_file).sheet_names

# Leer los datos de las hojas correspondientes y eliminar la primera fila y columna
Param_albergue = pd.read_excel(excel_file, sheet_name=sheets[0]).iloc[:, 1:]    # Parametros de albergue
R_df = pd.read_excel(excel_file, sheet_name=sheets[1]).iloc[:, 1:]              # Recursos
o_df = pd.read_excel(excel_file, sheet_name=sheets[2]).iloc[:, 1:]              # Recursos operativos
fv = pd.read_excel(excel_file, sheet_name=sheets[3]).iloc[:, 1:]                # Fecha de vencimiento
ct = pd.read_excel(excel_file, sheet_name=sheets[4]).iloc[:, 1:]                # Costos de transporte
Escalares = pd.read_excel(excel_file, sheet_name=sheets[5]).iloc[:, 1:]         # Escalares



# Crear las listas
CT = pd.Series(Param_albergue.iloc[:, 0].values, index=range(1, len(Param_albergue) + 1)).astype(int)   # Costo asociado al transporte inicial al albergue "i"
A = pd.Series(Param_albergue.iloc[:, 1].values, index=range(1, len(Param_albergue) + 1)).fillna(0)      # Capacidad de almacenamiento del albergue "i"
C = pd.Series(Param_albergue.iloc[:, 2].values, index=range(1, len(Param_albergue) + 1)).fillna(0)      # Costo asociado a habilitar el albergue "i"
MP = pd.Series(Param_albergue.iloc[:, 3].values, index=range(1, len(Param_albergue) + 1)).fillna(0)     # Capacidad maxima de personas del albergue "i"

RM = pd.Series(R_df.iloc[:, 0].values, index=range(1, len(R_df) + 1)).fillna(0)                         # Cantidad de recursos requeridos "r" por persona por día 
CR = pd.Series(R_df.iloc[:, 1].values, index=range(1, len(R_df) + 1)).fillna(0)                         # Costo de adquirir el recursos "r"
B = pd.Series(R_df.iloc[:, 2].values, index=range(1, len(R_df) + 1)).fillna(0)                          # Multa asociada al desecho del recurso "r"

RMO = pd.Series(o_df.iloc[:, 0].values, index=range(1, len(o_df) + 1)).fillna(0)                        # Cantidad de recursos operativos "o" por persona por mes
CO = pd.Series(o_df.iloc[:, 1].values, index=range(1, len(o_df) + 1)).fillna(0)                         # Costo de adquirir el recurso operativo "o"

FV = pd.DataFrame(fv).fillna(0)                                                                         # Fecha de vencimiento del recurso "r" del lote "p"
FV.index = range(1, len(FV) + 1)                                                                        
FV.columns = range(1, len(FV.columns) + 1)

CT2 = pd.DataFrame(ct).fillna(0)                                                                        # Costo asociado al transporte entre el albergue "i" y el albergue "j"
CT2.columns = range(1, len(CT2.columns) + 1)
CT2.index = range(1, len(CT2) + 1)


# Convertir a diccionarios
RMO_dict = {(o): RMO[o] for o in RMO.index}                                     # Cantidad de recursos operativos "o" por persona por mes
RM_dict = {(r): RM[r] for r in RM.index}                                        # Cantidad de recursos requeridos "r" por persona por día 
FV_dict = {(r, p): FV.at[r, p] for r in FV.index for p in FV.columns}           # Fecha de vencimiento del recurso "r" del lote "p"
C_dict = {(i): C[i] for i in C.index}                                           # Costo asociado a habilitar el albergue "i"
CR_dict = {(r): CR[r] for r in CR.index}                                        # Costo de adquirir el recursos "r"
CO_dict = {(o): CO[o] for o in CO.index}                                        # Costo de adquirir el recurso operativo "o"
CT_dict = {(i): CT[i] for i in CT.index}                                        # Costo asociado al transporte inicial al albergue "i"
CT2_dict = {(i, j): CT2.at[i, j] for i in CT2.index for j in CT2.columns}       # Costo asociado al transporte entre el albergue "i" y el albergue "j"
B_dict = {(r): B[r] for r in B.index}                                           # Multa asociada al desecho del recurso "r"
P = Escalares.iloc[0, 0]                                                        # Cantidad de lotes de alimentos "1414810000"
d = Escalares.iloc[1, 0]                                                        # Número de personas totales "100000" en todos los días
A_dict = {(i): A[i] for i in A.index}                                           # Capacidad de almacenamiento del albergue "i"
MP_dict = {(i): MP[i] for i in MP.index}                                        # Capacidad maxima de personas del albergue "i"

""" D_t = {(t): 10 for t in range(1, 31)}                                           # Demanda diaria de personas que necesitan alojarse en albergues en día "t" (30 días) """

tau1 = 4                                                                        # Permanencia mínima en el albergue (15 días)
tau2 = 14                                                                       # Permanencia máxima en el albergue (7 días)
DR = {(r, p): 100 for r in RM_dict.keys() for p in range(1, 5)}                 # Cantidad inicial de donaciones del recurso "r" del lote "p" en el centro de distribución


D_r = {(r): sum(DR[r, p] for p in range(1, 5)) for r in RM_dict.keys()}         # Cantidad inicial total de donaciones del recurso "r"
DG = sum(RM_dict[r] for r in RM_dict.keys())                                    # Cantidad de desechos generados por persona por día


# Iniciacion del modelo
modelo = Model()
modelo.setParam("TimeLimit", 30 * 60)  # Seteamos un tiempo limite de 20 minutos

# ============================= Variables =============================

D_t = modelo.addVars(T_, vtype=GRB.SEMIINT, name="D_t")                 # Demanda diaria de personas que necesitan alojarse en albergues en día "t" (30 días)

Pi = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="Pi")                # Presupuesto disponible en el día "t" (30 días)



x = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="x")                 # Cantidad de personas en el albergue $i$ en día $t$.
y = modelo.addVars(I_, T_, vtype=GRB.BINARY, name="y")                  # Variable binaria que indica si el albergue $i$ esta habilitado (1) o no (0) en el día $t$.
z = modelo.addVars(I_, T_, vtype=GRB.INTEGER, name="z")                 # Flujo de personas en el albergue $i$ en día $t$.
z_plus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_plus")       # Personas que llegan al albergue $i$ en el día $t$.
z_minus = modelo.addVars(I_, T_, vtype=GRB.SEMIINT, name="z_minus")     # Personas que salen del albergue $i$ en el día $t$.
g = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="g")         # Cantidad de recurso $r$ del lote $p$ comprado y asignado en el albergue $i$ en día $t$.
r = modelo.addVars(O_, I_, T_, vtype=GRB.SEMIINT, name="r")             # Cantidad de recurso operativo $o$ en albergue $i$ en día $t$.
h = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="h")         # Cantidad de recurso $r$ del lote $p$ donado y asignado en el albergue $i$ en día $t$.
b = modelo.addVars(R_, I_, T_, vtype=GRB.SEMIINT, name="b")             # Cantidad de recurso $r$ desechado en albergue $i$ en día $t$
T = modelo.addVars(R_, I_, I_, T_, vtype=GRB.SEMIINT, name="T")         # Cantidad de recurso $r$ transferido desde albergue $i$ a albergue $i'$ en día $t$.
I = modelo.addVars(R_, P_, T_, vtype=GRB.SEMIINT, name="I")             # Inventario del centro de distribución del recurso $r$ del lote $p$ al final del día $t$.
I_A = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="I_A")     # Inventario del recurso $r$ del lote $p$ en el albergue $i$ en el día $t$.
a = modelo.addVars(T_, vtype=GRB.SEMIINT, name="a")                     # Cantidad de personas sin albergue en el día $t$.
CN = modelo.addVars(R_, P_, I_, T_, vtype=GRB.SEMIINT, name="CN")       # Cantidad de recurso $r$ consumido del lote $p$ en el albergue $i$ en el día $t$.


# Actualizamos el modelo para guardar las variables
modelo.update()


# =========================== Restricciones ===========================
# Agregar la restricción usando Big-M
M = 1000000  # Un valor suficientemente grande


#RE0 x[i, 1]: Flujo al inicio, debe ser igual al número de personas que puede entrar al albergue, teniendo que es D_ si es menor que la capacidad máxima del albergue y que es MP_dict[i] si es mayor 
""" for i in I_:
    if D_[1] <= MP_dict[i]:
        modelo.addConstr(x[i, 1] == D_[1], name=f"RE0.D")
    else:
        modelo.addConstr(x[i, 1] == MP_dict[i], name=f"RE0.MP") """

for i in I_:
    modelo.addConstr(x[i, 1] == D_[1], name="RE0")
    #modelo.addConstr(x[i, 1] > D_[1], name="RE0")


# RE1 D_t: primer día y actualización diaria de la demanda
# Demanda del primer día
modelo.addConstr(D_t[1] == x[i,1], name="RE1.1")
# Actualización diaria de la demanda
modelo.addConstrs((D_t[t] == D_t[t-1] + D_[t] - quicksum(z_minus[i, t] for i in I_) for t in range(2, m + 1)), name="RE1.2")

# RE2 z_minus: Si las personas que llegaron hace k+1 días llevan más de 4 días
for i in I_:
    for t in range(4, m + 1):  # t >= 5 para asegurar que llevan más de 4 días
        for k in range(1, 15):
            if t - k >= 1:
                # Restricción para cuando el albergue está lleno
                modelo.addConstr((y[i, t - 1] == 1) >> (z_minus[i, t] == z_plus[i, t - k]), name=f"NEW1_full_{i}_{t}_{k}")

                # Restricción para cuando no hay recursos en el día t debe salir la misma cantidad de personas que llegaron hace k días
                # R10: Consumo de recursos en albergue
                modelo.addConstrs((quicksum(CN[r, p, i, t] for p in P_) == RM_dict[r] * x[i, t] for r in R_ for i in I_ for t in T_), name="NEW_2") # Consumo de recursos en albergue = requerimiento mínimo por persona * personas en albergue

                # Restricción para cuando las personas llevan más de 14 días
                if t - k >= 15:
                    modelo.addConstr(z_minus[i, t] == z_plus[i, t - k], name=f"NEW3_{i}_{t}_{k}")
                    
# RE3: Flujo de personas en cada albergue: demanda diaria y capacidad máxima
# Asegurar que solo puedan entrar si hay capacidad y suficientes recursos
modelo.addConstrs((z_plus[i, t] <= (MP_dict[i] - x[i, t]) * y[i, t] for i in I_ for t in T_), name="RE3_capacity")
modelo.addConstrs((z_plus[i, t] * RM_dict[r] <= quicksum(g[r, p, i, t] + h[r, p, i, t] for p in P_) for r in R_ for i in I_ for t in T_), name="RE3_resources")

# RE4: Actualización del inventario
modelo.addConstrs((I_A[r, p, i, t + 1] == I_A[r, p, i, t] - CN[r, p, i, t] + quicksum(T[r, j, i, t] - T[r, i, j, t] for j in I_ if j != i) for r in R_ for p in P_ for i in I_ for t in range(1, m)), name="RE3_inventory")

# RE5: Presupuesto disponible en el día t
# Establecer el presupuesto inicial en el día 1
for i in I_:
    modelo.addConstr(Pi[i, 1] == 1000000000, name=f"RE5.1")
# Actualización del presupuesto en t+1 según las personas que hay en t
modelo.addConstrs((Pi[i, t + 1] == Pi[i, t] - quicksum(CO_dict[o] * r[o, i, t] for o in O_ for i in I_) 
                   - quicksum(CR_dict[r] * CN[r, p, i, t] for r in R_ for p in P_ for i in I_) 
                   - quicksum(CT_dict[i] * z[i, t] for i in I_) 
                   - quicksum(B_dict[r] * b[r, i, t] for r in R_ for i in I_) 
                   for t in range(1, m) for i in I_), name="RE5.2")


# RE6 y: Restricciones para y (si albergue esta habilitado) 
# Habilitar albergues el primer día
modelo.addConstrs((y[i, 1] == 1 for i in I_), name="RE4.1")
# Deshabilitar albergue si está lleno
modelo.addConstrs((y[i, t+1] <= 1 - (x[i, t] / MP_dict[i]) for i in I_ for t in range(1, m)), name="RE4.2")



# R1: Cantidad inicial y máximo de personas en cada albergue. Las personas solo se pueden alojar en albergues habilitados.
modelo.addConstrs((x[i, 1] == z[i, 1] for i in I_), name="R1.1")  # Personas el primer día = flujo de ingreso
modelo.addConstrs((x[i, t] <= MP[i] * y[i, t] for i in I_ for t in T_), name="R1.2")  # Personas <= capacidad máxima si habilitado

""" RE0 hace esto con el minimo entre D_[1] y MP_dict[i] para cada i en I_
# R1: Cantidad inicial y máximo de personas en cada albergue. Las personas solo se pueden alojar en albergues habilitados.
# Cantidad inicial de personas en cada albergue
modelo.addConstrs((x[i, 1] == z[i, 1] for i in I_), name="R1.1")

    RE3 restringe z_plus a la capacidad máxima del albergue - personas en el albergue,
    por lo que si hay 0 personas en el albergue el máximo es la capacidad máxima del albergue
# Cantidad máxima de personas en cada albergue si está habilitado
modelo.addConstrs((x[i, t] <= MP_dict[i] * y[i, t] for i in I_ for t in T_), name="R1.2")"""


# R2: Balance del flujo de personas en cada albergue.
modelo.addConstrs((x[i, t] == x[i, t-1] + z[i, t] for i in I_ for t in range(2, m + 1)), name="R2")

# R3: El número de recursos operativos debe cumplir con el requerimiento mínimo por persona.
modelo.addConstrs((x[i, t] * RMO_dict[o] <= r[o, i, t] for o in O_ for i in I_ for t in T_), name="R3")  # Personas * req. operativos <= recursos operativos


# R4: El número de recursos debe cumplir con el requerimiento mínimo por persona.
modelo.addConstrs((x[i, t] * RM_dict[r] <= quicksum(g[r, p, i, t] + h[r, p, i, t] for p in P_) for r in R_ for i in I_ for t in T_), name="R4")


# R5: Actualización diaria de recursos y donaciones en inventario
# Inventario inicial del centro de distribución
modelo.addConstrs((I[r, p, 1] == DR[r, p] for r in R_ for p in P_), name="R5.1")
# Actualización diaria del inventario del centro de distribución
modelo.addConstrs((I[r, p, t + 1] == I[r, p, t] - quicksum(h[r, p, i, t] for i in I_) for r in R_ for p in P_ for t in range(1, m)), name="R5.2")


# R6: Prohibición de asignación de recursos y donaciones vencidas
modelo.addConstrs((quicksum(h[r, p, i, t] + g[r, p, i, t] for i in I_) == 0 for r in R_ for p in P_ for t in T_ if t > FV_dict[(r, p)]), name="R6")


# R7 b[r, i, t]: La cantidad de desechos generados son las donaciones y recursos vencidos, más los desechos que se generan por personas en cada albergue.
# Desechos generados por recursos vencidos
modelo.addConstrs((b[r, i, t] == I_A[r, p, i, t] for r in R_ for p in P_ for i in I_ for t in T_ if t > FV_dict[(r, p)]), name="R7.1")
# No desechos si los recursos no están vencidos
modelo.addConstrs((b[r, i, t] == 0 for r in R_ for p in P_ for i in I_ for t in T_ if t <= FV_dict[(r, p)]), name="R7.2")


# R8: Inventario inicial adicional de cada albergue = 0.
#modelo.addConstrs((I_A[r, p, i, 1] == 0 for r in R_ for p in P_ for i in I_), name="R8")


# R9: Balance de inventario en los albergues
modelo.addConstrs((I_A[r, p, i, t] == I_A[r, p, i, t-1] + g[r, p, i, t-1] + h[r, p, i, t-1] + quicksum(T[r, j, i, t-1] - T[r, i, j, t-1] for j in I_ if j != i) - CN[r, p, i, t-1] for r in R_ for p in P_ for i in I_ for t in range(2, m + 1)), name="R9")


# R10: Consumo de recursos en albergue
modelo.addConstrs((quicksum(CN[r, p, i, t] for p in P_) == RM_dict[r] * x[i, t] for r in R_ for i in I_ for t in T_), name="R10") 

# R11: Límite de inventario
modelo.addConstrs((CN[r, p, i, t] <= I_A[r, p, i, t] for r in R_ for p in P_ for i in I_ for t in T_), name="R11")


# R12: Límite de donaciones asignadas
modelo.addConstrs((quicksum(h[r, p, i, t] for p in P_ for i in I_ for t in T_) <= D_r[r] for r in R_), name="R12")


# R13: Capacidad máxima de almacenamiento
modelo.addConstrs((quicksum(I_A[r, p, i, t] + g[r, p, i, t] + h[r, p, i, t] + quicksum(T[r, i, j, t] for j in I_ if j != i) for r in R_ for p in P_) <= A_dict[i] for i in I_ for t in T_), name="R13")


# R14: Los costos de cada albergue no pueden superar el presupuesto disponible por albergue
modelo.addConstrs(
    ((quicksum(CO_dict[o] * r[o, i, t] for o in O_ for t in T_) +
     quicksum(CR_dict[r] * CN[r, p, i, t] for r in R_ for p in P_ for t in T_) +
     quicksum(CT_dict[i] * z[i, t] for t in T_) +
     quicksum(B_dict[r] * b[r, i, t] for r in R_ for t in T_) <= Pi[i, m])
    for i in I_), name="R14"
)

# R15: Capacidad máxima de almacenamiento del albergue
#modelo.addConstrs((quicksum(r[o, i, t] for o in O_) <= A_dict[i] for i in I_ for t in T_), name="R15")


""" RE2 y RE3 hacen esto
# R16: Permanencia mínima en el albergue.
modelo.addConstrs((z_minus[i, t] == 0 for i in I_ for t in range(1, tau2)), name="R16.1")  # No retiros antes del tiempo mínimo 15 días
modelo.addConstrs((z_minus[i, t] <= quicksum(z_plus[i, k] for k in range(t - tau2 + 1, t + 1)) for i in I_ for t in range(tau2, m + 1)), name="R16.2")  # Retiros <= ingresos últimos tau días """


# R17: Flujo neto de personas en el albergue
modelo.addConstrs((z[i, t] == z_plus[i, t] - z_minus[i, t] for i in I_ for t in T_), name="R17")


# R18: Personas sin albergue por día
modelo.addConstrs((a[t] == D_t[t] for t in [1]), name="R18.1")
modelo.addConstrs((a[t] == D_t[t] - quicksum(x[i, t] for i in I_) for t in range(2, m+1)), name="R18.2")


# Asegurar que algunas variables sean mayores que 0
#modelo.addConstr(quicksum(x[i, 1] for i in I_) >= 1)
#modelo.addConstrs(z[i, t] >= 1 for i in I_ for t in T_)  # Flujo de personas
#modelo.addConstrs(g[r, p, i, t] >= 1 for r in R_ for p in P_ for i in I_ for t in T_)  # Recursos asignados
#modelo.addConstrs(h[r, p, i, t] >= 1 for r in R_ for p in P_ for i in I_ for t in T_)  # Recursos básicos
#modelo.addConstrs(r[o, i, t] >= 1 for o in O_ for i in I_ for t in T_)  # Recursos operativos


# =========================== Funcion Objetivo ===========================
# Funcion Objetivo: Minimizar la cantidad de personas sin albergue
objetivo = quicksum(D_[t] for t in T_)
modelo.setObjective(objetivo, GRB.MINIMIZE)


# Optimizar el modelo
modelo.optimize()



# Si el modelo es infeasible, calcular IIS
if modelo.status == GRB.INFEASIBLE:
    print("El modelo es infeasible. Calculando IIS...")
    modelo.computeIIS()
    modelo.write("model.ilp")
    for c in modelo.getConstrs():
        if c.IISConstr:
            print(f"Restricción conflictiva: {c.constrName}")

# Continuar con el resto del código
if modelo.status == GRB.OPTIMAL or modelo.status == GRB.SUBOPTIMAL:
    print(f"Valor objetivo: {modelo.ObjVal}")
    print("\n\nValores de las variables D_t después de la optimización:")
    for t in T_:
        print(f"D_t[{t}]: {D_t[t].X}")
else:
    print("El modelo es infeasible. No se pueden obtener los valores de las variables D_t.")


# Continuar con el resto del código

# Verificar las restricciones
""" print("\n\nRestricciones y sus holguras:")
for constr in modelo.getConstrs():
    try:
        print(f"{constr.ConstrName}: {constr.Slack}")
    except AttributeError:
        print(f"{constr.ConstrName}: No se pudo obtener la holgura")

        
# Verificar la función objetivo
if modelo.status == GRB.OPTIMAL or modelo.status == GRB.SUBOPTIMAL:
    print(f"Valor objetivo: {modelo.ObjVal}")
else:
    print("No se pudo obtener el valor objetivo porque el modelo es infactible.")

# Verificar los valores de las variables
print("\n\nValores de las variables después de la optimización:")
for v in modelo.getVars():
    print(f"{v.varName}: {v.x}") """


# =========================== Manejo de soluciones ============================

### TODO: Toda esta parte la verdad

# El output en consola de este programa son los datos sobre el modelo, los datos sobre las variables y su respectiva interpretacion seran guardados en la carpeta Resultados


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
n_res_activas = 0  # Contamos el numero de restricciones con slack 0
for constr in modelo.getConstrs():
    try:
        if constr.getAttr("slack") == 0:
            n_res_activas += 1
    except AttributeError:
        pass

print(f"Número de restricciones activas: {n_res_activas}")

print(f"Numero Restricciones Activas: {n_res_activas}")

 
### Tiempo de ejecución
print(f"Tiempo de ejecución: {modelo.Runtime} segundos")

### Valor optimo del modelo
try:
    print(f"Valor objetivo: {modelo.ObjVal}")
except AttributeError:
    pass


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
            data = (tupla_indices, v[tupla_indices].X)
        else:
            data = tupla_indices + (v[tupla_indices].X,)
        lst.append(data)
    
    indx.append(name)
    columns = indx
    df = pd.DataFrame(lst, columns=columns)
    
    df.to_csv(f"Resultados/Vars/{name}.csv", columns=columns)



# ======================= Creacion de Graficos ============================

# Esto se guarda en Resultados/Graficos

### Grafico cantidad de personas en todos los albergues por dia
plt.style.use('ggplot')

y_axis = [sum(x[i,t].X for i in I_) for t in T_]    # Cantidad total de personas refugiadas en dia t
x_axis = T_


fig, ax = plt.subplots()

ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("Resultados/Graficos/g1-Cantidad de personas refugiadas por dia")




### Grafico cantidad de recursos asignados a cada albergue por dia
plt.style.use('ggplot')

y_axis = [sum(g[r,p,i,t].X + h[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos asignados en dia t
x_axis = T_

fig, ax = plt.subplots()

ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
         ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("Resultados/Graficos/g2-Cantidad de recursos asignados a cada albergue por dia")



### Grafico cantidad de recursos consumidos por dia
plt.style.use('ggplot')

y_axis = [sum(CN[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos consumidos en dia t
x_axis = T_

fig, ax = plt.subplots()

ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
            ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("Resultados/Graficos/g3-Cantidad de recursos consumidos por dia")
# El gráfico esta vacio, para saber la causa de esto, puedo imprimir los valores de las variables y ver si estan siendo asignados correctamente



### Grafico cantidad de distintos tipos de recursos (consumidos, asignados, asignados adicionales, transferidos) por dia
plt.style.use('ggplot')

y_axis = [sum(CN[r,p,i,t].X + g[r,p,i,t].X + h[r,p,i,t].X + I_A[r,p,i,t].X + T[r,i,j,t].X for r in R_ for p in P_ for i in I_ for j in I_) for t in T_]    # Cantidad total de recursos consumidos en dia t
x_axis = T_

fig, ax = plt.subplots()

ax.bar(x_axis, y_axis, width=1, edgecolor="white", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
            ylim=(0, 8), yticks=np.arange(1, 8))

plt.savefig("Resultados/Graficos/g4-Cantidad de recursos consumidos por dia")



### Grafico cantidad de distintos tipos de recursos (consumidos, asignados, asignados adicionales, transferidos) por dia con diferentes colores para cada categoria:
plt.style.use('ggplot')

y_axis1 = [sum(CN[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos consumidos en dia t
y_axis2 = [sum(g[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos asignados en dia t
y_axis3 = [sum(h[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos asignados en dia t
y_axis4 = [sum(I_A[r,p,i,t].X for r in R_ for p in P_ for i in I_) for t in T_]    # Cantidad total de recursos asignados en dia t

x_axis = T_

fig, ax = plt.subplots()

ax.bar(x_axis, y_axis1, width=1, edgecolor="white", linewidth=0.7, color="blue", label="Recursos Consumidos")
ax.bar(x_axis, y_axis2, width=1, edgecolor="white", linewidth=0.7, color="green", label="Recursos Asignados")
ax.bar(x_axis, y_axis3, width=1, edgecolor="white", linewidth=0.7, color="red", label="Recursos Asignados Adicionales")
ax.bar(x_axis, y_axis4, width=1, edgecolor="white", linewidth=0.7, color="yellow", label="Recursos Transferidos")

ax.set(xlim=(10, 20), xticks=np.arange(10, 20),
            ylim=(10, 20), yticks=np.arange(10, 20))

# Se añade la leyenda con los colores y los nombres de los recursos
ax.legend()

plt.savefig("Resultados/Graficos/g5-Cantidad de distintos tipos de recursos por dia") 



# TODO: Interpretacion de los resultados
"""
if modelo.status == GRB.OPTIMAL:
    print(f"Valor objetivo: {modelo.objVal}")

    # Imprimir los valores de las variables de decisión
    for i,t in x:
        print(f"Se encuentran {x[i,t].X} personas en el albergue {i} el dia {t}")
 """