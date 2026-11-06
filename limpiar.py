
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURACION
# ==========================================

ruta_entrada = Path("data/MASTER_CHART_F1000Research.xlsx")
ruta_salida = Path("data/datos_limpios.csv")

# ==========================================
# 2. CARGAR BASE DE DATOS
# ==========================================

df = pd.read_excel(ruta_entrada)
df.columns = df.columns.str.strip()

print("\n========== LIMPIEZA DENGUE AI ==========")
print("Registros originales:", len(df))

# ==========================================
# 3. ELIMINAR DUPLICADOS
# ==========================================

duplicados = df.duplicated().sum()

df = df.drop_duplicates().copy()

print("\nDuplicados eliminados:", duplicados)
print("Registros actuales:", len(df))

# ==========================================
# 4. ELIMINAR VARIABLES NO UTILIZADAS
# ==========================================

columnas_excluir = [
    "FERRITIN",
    "IV FLUIDS",
    "ANTIBIOTICS",
    "ANTIPYRETICS",
    "STEROIDS"
]

df = df.drop(
    columns=columnas_excluir,
    errors="ignore"
)

# ==========================================
# 5. CONVERTIR VARIABLES CATEGORICAS
# ==========================================

# Codificacion original:
# 1 = Si / 2 = No
# Nueva codificacion:
# 1 = Si / 0 = No

columnas_si_no = [
    "HEADACHE",
    "MYALGIA",
    "ABDOMINAL PAIN",
    "RASH",
    "VOMITING",
    "BREATHLESSNESS",
    "BLEEDING",
    "BP AT ADMISSION",
    "ORGANOMEGALY"
]

for columna in columnas_si_no:
    df[columna] = df[columna].map({
        1: 1,
        2: 0
    })

# ==========================================
# 6. CODIFICAR SEXO Y EDAD
# ==========================================

# GENDER:
# 1 = Masculino
# 2 = Femenino

df["GENDER"] = df["GENDER"].map({
    1: 0,
    2: 1
})

# AGE:
# 1 = Menor de 6 años
# 2 = Mayor de 6 años

df["AGE"] = df["AGE"].map({
    1: 0,
    2: 1
})

# ==========================================
# 7. CODIFICAR VARIABLE OBJETIVO
# ==========================================

objetivo = "GROUPS: DENGUE FEVER OR COMPLICATED DENGUE"

df[objetivo] = df[objetivo].map({
    1: 0,
    2: 1
})

# ==========================================
# 8. TRATAR VALORES FALTANTES
# ==========================================

columnas_numericas = [
    "ALT",
    "CREATININE"
]

for columna in columnas_numericas:

    mediana = df[columna].median()

    df[columna] = df[columna].fillna(mediana)

    print(
        f"Valores faltantes de {columna}: "
        f"imputados con mediana {mediana}"
    )

# ==========================================
# 9. CREAR NUEVAS VARIABLES
# ==========================================

df["PCV_ALTO"] = (
    df["PCV"] >= 42
).astype(int)

df["PLAQUETAS_BAJAS"] = (
    df["PLATELET COUNT"] <= 75000
).astype(int)

df["WBC_ALTO"] = (
    df["HIGHEST WBC"] >= 7000
).astype(int)

df["ALT_ALTO"] = (
    df["ALT"] >= 70
).astype(int)

# ==========================================
# 10. VERIFICAR COLUMNAS CONSTANTES
# ==========================================

constantes = [
    columna for columna in df.columns
    if df[columna].nunique(dropna=False) <= 1
]

if constantes:
    print("\nColumnas constantes:", constantes)
    df = df.drop(columns=constantes)
else:
    print("\nNo existen columnas constantes.")

# ==========================================
# 11. VERIFICAR DATOS FALTANTES
# ==========================================

print("\nValores faltantes restantes:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ==========================================
# 12. GUARDAR DATASET LIMPIO
# ==========================================

df.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8-sig"
)

print("\n=====================================")
print("LIMPIEZA COMPLETADA")
print("=====================================")

print("Registros finales:", len(df))
print("Variables finales:", len(df.columns))
print("Archivo generado:", ruta_salida)

print("\nDistribucion de clases:")

print(df[objetivo].value_counts())

print("\nPrimeros registros:")

print(df.head())
