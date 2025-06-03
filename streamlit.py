# streamlit_app.py
#Heijunka project for streamlit

import streamlit as st
import tempfile, os
from pathlib import Path

# ——————————————————————————————
# 1) Import all the libraries you need
#    (same as your script: pandas, numpy, scipy, xgboost, openpyxl, etc.)
# ——————————————————————————————
import pandas as pd
import numpy as np
from scipy.stats import norm, gamma, kstest
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import openpyxl
import xgboost as xgb
#import pmdarima as pm
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font

# …any other imports you used (e.g. seaborn if you still want plots in the notebook)…
# Note: Streamlit can show matplotlib charts directly, but if you only need Excel output,
#       you don’t have to display the charts in the Streamlit UI.


# ——————————————————————————————
# 2) Wrap your existing script into one function
#    that reads an input Excel and writes a new Excel with forecasts.
#    You can literally paste your code inside here, replacing hardcoded filenames
#    with the in_path/out_path parameters.
# ——————————————————————————————

def run_forecast_pipeline(in_path: str, out_path: str):
    """
    1) Load 'GLOBAL_LEAN_1_copia.xlsx' from in_path
    2) Do all the sheet-by-sheet loops, compute ARIMA, XGBoost, Monte Carlo, etc.
    3) Write the final Forecast sheet and any Heijunka formulas into out_path
    """
    # Load the workbook
    workbook = openpyxl.load_workbook(in_path)
    sheet_ins = workbook["Instrucciones"]
    valor = sheet_ins["C2"].value

    # Create dataframes for each "Proceso i"
    dfs = {}
    params_dict = {}
    for i in range(1, valor + 1):
        df_i = pd.read_excel(in_path, sheet_name=f"Proceso {i}")
        sheet_i = workbook[f"Proceso {i}"]

        turno   = sheet_i["H1"].value
        horas   = sheet_i["H2"].value
        descanso= sheet_i["H3"].value
        prod    = sheet_i["H4"].value
        prod_obj= sheet_i["H6"].value
        trab    = sheet_i["H7"].value

        # Keep only the needed columns:
        df_i = df_i[["Día", "Turno", "Carga histórica", "Número de FTE"]].copy()
        # Add Date column, drop original Día
        df_i["Date"] = pd.to_datetime(df_i["Día"])
        df_i.drop(columns=["Día"], inplace=True)
        # Group stats (mean/std) if you want plots or stats
        # ...
        # Save to dict for later use:
        dfs[i] = df_i.copy()
        params_dict[i] = dict(
            turno=turno,
            horas=horas,
            descanso=descanso,
            productividad=prod,
            productividad_objetivo=prod_obj,
            trabajadores=trab
        )

    # Example: Focus on Proceso 1 for demonstration (you can loop all i)
    df1 = dfs[1]
    params1 = params_dict[1]

    # Split train/test by last month
    last_month = df1['Date'].dt.to_period("M").max()
    df1_train = df1[df1['Date'].dt.to_period("M") < last_month].copy()
    df1_test  = df1[df1['Date'].dt.to_period("M") == last_month].copy()

    # ——————————————————————————————
    # 3) MONTE CARLO SIMULATION (if you want that)
    #    (you already had a function `simulacion_montecarlo(df)` in your script)
    # ——————————————————————————————
    def create_features(df, label=None):
        df2 = df.copy()
        df2["dayofweek"] = df2["Date"].dt.dayofweek
        df2["quarter"]   = df2["Date"].dt.quarter
        df2["month"]     = df2["Date"].dt.month
        df2["year"]      = df2["Date"].dt.year
        df2["dayofyear"] = df2["Date"].dt.dayofyear
        df2["day"]       = df2["Date"].dt.day
        df2["week"]      = df2["Date"].dt.isocalendar().week

        X = df2[["dayofweek", "quarter", "month", "year", "dayofyear", "day", "week"]]
        if label:
            y = df2[label]
            return X, y
        return X

    # Train XGBoost on df1_train
    X_train, y_train = create_features(df1_train, label="Carga histórica")
    X_test,  y_test  = create_features(df1_test,  label="Carga histórica")

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8)
    xgb_model.fit(X_train, y_train)
    # Forecast next 30 days
    last_date = df1["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")
    df_future = pd.DataFrame({"Date": future_dates})
    X_future = create_features(df_future)
    preds_future = xgb_model.predict(X_future)

    # ——————————————————————————————
    # 4) WRITE THE NEW "Forecast" SHEET into the same workbook object
    # ——————————————————————————————

    if "Forecast" in workbook.sheetnames:
        del workbook["Forecast"]
    sheet_fc = workbook.create_sheet(title="Forecast")
    bold_font = openpyxl.styles.Font(bold=True)
    
    # 2) Build a DataFrame for these predictions
    df_final = pd.DataFrame({
        "Día": future_dates,
        "Demanda": preds_future
    })
    
    # 3) Write headers with bold font
    sheet_fc["A1"].value = "Día"
    sheet_fc["A1"].font  = bold_font
    sheet_fc["B1"].value = "Demanda"
    sheet_fc["B1"].font  = bold_font
    sheet_fc["C1"].value = "Edición"
    sheet_fc["C1"].font  = bold_font
    sheet_fc["D1"].value = "Demanda Final"
    sheet_fc["D1"].font  = bold_font
    
    # 4) Write data rows and pre-fill the "Demanda Final" formula
    for row_idx, row in enumerate(df_final.itertuples(index=False), start=2):
        # Column A: the Date
        sheet_fc.cell(row=row_idx, column=1).value = row.Día
        # Column B: the forecasted Demanda
        sheet_fc.cell(row=row_idx, column=2).value = float(row.Demanda)
        # Column D: set formula to pick edited value if any, or use B value otherwise
        sheet_fc.cell(row=row_idx, column=4).value = f"=IF(C{row_idx}=0, B{row_idx}, C{row_idx})"
    
    # 5) Fill editable cells in column C (Edición) with a yellow background
    fill_yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    for r in range(2, 2 + len(df_final)):
        sheet_fc.cell(row=r, column=3).fill = fill_yellow
    
    # 6) Add FTE-related columns E, F, G with formulas referencing "Proceso 1" sheet
    trabajadores = params1["trabajadores"]
    prod        = params1["productividad"]
    prod_obj    = params1["productividad_objetivo"]
    
    sheet_fc["E1"].value = "Número de FTE actuales"
    sheet_fc["E1"].font  = bold_font
    sheet_fc["F1"].value = "Productividad (u/hh)"
    sheet_fc["F1"].font  = bold_font
    sheet_fc["G1"].value = "Productividad objetivo (u/hh)"
    sheet_fc["G1"].font  = bold_font
    
    for r in range(2, 2 + len(df_final)):
        # Always the same numeric values or formulas pointing to "Proceso 1"!H7, H4, H6
        sheet_fc.cell(row=r, column=5).value = f"='Proceso 1'!$H$7"
        sheet_fc.cell(row=r, column=6).value = f"='Proceso 1'!$H$4"
        sheet_fc.cell(row=r, column=7).value = f"='Proceso 1'!$H$6"
    
    # 7) Add Heijunka grid/formula logic in columns H through Z (example placeholders)
    #    You can adjust column indices and formulas as needed.
    
    # Example: Column J ("Horas necesarias") = D / F  → column index 10
    #          Column K ("FTE necesarias")  = J / (H2 - (H3/60))  → column index 11
    #          Column L ("Horas Objetivo")   = D / G  → column index 12
    #          Column M ("FTE objetivos")    = L / (H2 - (H3/60))  → column index 13
    #          Column N ("Diferencia Horas") = J - L  → column index 14
    #          Column O ("Diferencia FTE")   = K - M  → column index 15
    #          Column P ("Ocupación (%)")   = J / (E * (H2 - (H3/60))) → column index 16
    #          Column Q ("Exceso/Falta Horas") = J - (E * (H2 - (H3/60))) → column index 17
    #          Column R ("Coste hora extra")    = 25  → column index 18
    #          Column S ("Coste hora ociosa")   = 18  → column index 19
    #          Column T ("Coste ineficiente")   = IF(Q>0, Q*R, -Q*S) → column index 20
    #          You can continue for columns V through Z similarly.
    
    for row in range(2, 2 + len(df_final)):
        # Column J (10)
        sheet_fc.cell(row=row, column=10).value = f"=D{row}/F{row}"
        # Column K (11)
        sheet_fc.cell(row=row, column=11).value = f"=J{row}/('Proceso 1'!$H$2-('Proceso 1'!$H$3/60))"
        # Column L (12)
        sheet_fc.cell(row=row, column=12).value = f"=D{row}/G{row}"
        # Column M (13)
        sheet_fc.cell(row=row, column=13).value = f"=L{row}/('Proceso 1'!$H$2-('Proceso 1'!$H$3/60))"
        # Column N (14)
        sheet_fc.cell(row=row, column=14).value = f"=J{row}-L{row}"
        # Column O (15)
        sheet_fc.cell(row=row, column=15).value = f"=K{row}-M{row}"
        # Column P (16)
        sheet_fc.cell(row=row, column=16).value = f"=J{row}/(E{row}*('Proceso 1'!$H$2-('Proceso 1'!$H$3/60)))"
        # Column Q (17)
        sheet_fc.cell(row=row, column=17).value = f"=J{row}-(E{row}*('Proceso 1'!$H$2-('Proceso 1'!$H$3/60)))"
        # Column R (18)
        sheet_fc.cell(row=row, column=18).value = 25
        # Column S (19)
        sheet_fc.cell(row=row, column=19).value = 18
        # Column T (20)
        sheet_fc.cell(row=row, column=20).value = f"=IF(Q{row}>0,Q{row}*R{row},-Q{row}*S{row})"
    
    # 8) Heijunka 2 logic in columns V through AB (for example)
    #     Column V (22): "Heijunka 2" header was already written above if needed.
    #     We assume V: index 22, W:23, X:24, Y:25, Z:26, AA:27, AB:28
    sheet_fc["V1"].value = "Heijunka 2"
    sheet_fc["V1"].font = bold_font
    sheet_fc["W1"].value = "Número de FTE propuesto"
    sheet_fc["W1"].font = bold_font
    sheet_fc["X1"].value = "Exceso/Falta de Horas (horas)"
    sheet_fc["X1"].font = bold_font
    sheet_fc["Y1"].value = "Coste hora extra (euros)"
    sheet_fc["Y1"].font = bold_font
    sheet_fc["Z1"].value = "Coste hora ociosa (euros)"
    sheet_fc["Z1"].font = bold_font
    sheet_fc["AA1"].value = "Coste ineficiente (euros)"
    sheet_fc["AA1"].font = bold_font
    sheet_fc["AB1"].value = "Coste total (euros)"
    sheet_fc["AB1"].font = bold_font
    
    # Place a default value of 1 in AD3 for initial W (assuming AD is column 30)
    sheet_fc["AD3"].value = 1
    sheet_fc["AC3"].value = "Número de FTE en el mes"
    sheet_fc["AC3"].font  = bold_font
    sheet_fc["AC4"].value = "Sobrecoste en el mes (euros)"
    sheet_fc["AC4"].font  = bold_font
    
    # Compute optimal W by brute force
    H2 = ws["H2"].value  # e.g. 8
    H3 = ws["H3"].value
    avail = H2 - (H3 / 60.0)
    R, S = 25.0, 18.0
    
    # Prepare J array from df_final and productividad
    J_values = df_final["Demanda"].values / params1["productividad"]
    
    def total_cost(W: float) -> float:
        X = J_values - W * avail
        AA = np.where(X > 0, X * R, -X * S)
        return AA.sum()
    
    Ws = np.arange(0, 5001, 1)  # 0..5000 inclusive, step=1
    costs = np.array([total_cost(w) for w in Ws])
    idx   = costs.argmin()
    W_opt    = Ws[idx]
    min_cost = costs[idx]
    
    # Write optimal W and cost into the sheet
    sheet_fc["AD3"].value = W_opt
    sheet_fc["AD4"].value = f"=SUM(AA2:AA{1 + len(df_final)})"  # Total cost formula based on column AA
    
    # 9) Apply number formatting (two decimals) to relevant columns
    two_decimals = NamedStyle(name="two_decimals")
    two_decimals.number_format = "0.00"
    workbook.add_named_style(two_decimals)
    
    # Columns to style: A (1), B (2), C (3), D (4), I (9), J (10), K (11), L (12), M (13), N (14), O (15), P (16), Q (17), T (20),
    # V (22), W (23), X (24), AA (27), AB (28), AC (29), AD (30)
    columns_to_style = [
        1, 2, 3, 4, 9, 10, 11, 12, 13, 14,
        15, 16, 17, 20, 22, 23, 24, 27, 28, 29, 30
    ]
    for col_idx in columns_to_style:
        col_letter = sheet_fc.cell(row=1, column=col_idx).column_letter
        for cell in sheet_fc[col_letter]:
            if cell.row > 1:  # Skip header row
                cell.style = two_decimals
    
    # 10) Fill ranges with pink and blue backgrounds
    from openpyxl.utils import column_index_from_string, get_column_letter
    
    pink_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
    blue_fill = PatternFill(start_color="CCFFFF", end_color="CCFFFF", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")
    
    for row in range(2, 2 + len(df_final)):
        # Column C is already yellow
        sheet_fc[f"C{row}"].fill = yellow_fill
    
        # Pink: columns I through T  (I=9 through T=20)
        for col in range(9, 21):
            sheet_fc.cell(row=row, column=col).fill = pink_fill
    
        # Blue: columns V through AD (V=22 through AD=30)
        for col in range(22, 31):
            sheet_fc.cell(row=row, column=col).fill = blue_fill
    
    # 11) Auto-size columns and adjust row heights
    for col_cells in sheet_fc.columns:
        max_length = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            if cell.value is not None:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        sheet_fc.column_dimensions[col_letter].width = max_length + 2  # add padding
    
    for row_cells in sheet_fc.iter_rows():
        max_lines = 1
        for cell in row_cells:
            if isinstance(cell.value, str):
                lines = str(cell.value).split("\n")
                if len(lines) > max_lines:
                    max_lines = len(lines)
        sheet_fc.row_dimensions[row_cells[0].row].height = max_lines * 15
    
    # 12) Save the updated workbook to out_path
    workbook.save(out_path)
    print(f"Forecast saved to: {out_path}")


# ——————————————————————————————
# 3) Streamlit UI: file uploader + call pipeline + download butt
# ——————————————————————————————
st.title("Global Lean Forecast (Streamlit Edition)")
st.markdown(
    """
    Upload your `GLOBAL_LEAN_1_copia.xlsx`, let Streamlit run the entire Python pipeline 
    (XGBoost, ARIMA, Monte Carlo, Heijunka formulas, etc.), and then download the updated Excel.
    """
)

uploaded_file = st.file_uploader("1) Upload your Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    st.write(f"Selected File: **{uploaded_file.name}**")

    # When the user clicks “Run Forecast,” save the uploaded contents to a temp file
    if st.button("2) Run Forecast"):
        with st.spinner("Running the full forecasting pipeline…"):
            # Save upload to a temporary location
            tmp_input = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp_input.write(uploaded_file.read())
            tmp_input.flush()
            tmp_input_path = tmp_input.name

            # Decide where to write the output
            tmp_output_path = os.path.join(tempfile.gettempdir(), "Forecast_Demanda.xlsx")

            try:
                run_forecast_pipeline(tmp_input_path, tmp_output_path)
            except Exception as e:
                st.error(f"❌ Forecast pipeline failed:\n```\n{e}\n```")
            else:
                st.success("✅ Forecast complete! Download below:")
                # Read the output file into memory
                with open(tmp_output_path, "rb") as f:
                    data = f.read()
                # Provide a Streamlit download button
                st.download_button(
                    label="Download Forecasted Excel",
                    data=data,
                    file_name="Forecast_Demanda.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Clean up the temp input file
            try:
                os.unlink(tmp_input_path)
            except OSError:
                pass
