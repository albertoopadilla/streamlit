# streamlit_app.py

import streamlit as st
import tempfile, os
from pathlib import Path

# ——————————————————————————————
# 1) Import all the libraries you need
#    (same as your script: pandas, numpy, scipy, xgboost, openpyxl, etc.)
# ——————————————————————————————
import pandas as pd
import numpy as np
import openpyxl
import xgboost as xgb
import pmdarima as pm
from scipy.stats import norm, gamma, kstest
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def run_forecast_pipeline(input_path: str, output_path: str) -> None:
    """
    1) Reads the uploaded Excel workbook from input_path.
    2) Runs the full forecasting pipeline (data prep, Monte Carlo, ARIMA, XGBoost, model comparison, etc.).
    3) Writes a new “Forecast” sheet into the workbook and saves it to output_path.
    """
    import pandas as pd
    import numpy as np
    import openpyxl
    from scipy.stats import norm, gamma, kstest
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pmdarima as pm
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # 1) Load the workbook and read “Instrucciones” to find how many “Proceso i” sheets exist
    workbook = openpyxl.load_workbook(input_path)
    instrucciones_sheet = workbook["Instrucciones"]
    valor = instrucciones_sheet["C2"].value  # Number of “Proceso” sheets

    # Dictionaries to hold each DataFrame and parameters
    dfs = {}        # will store df_{i} (full data)
    params = {}     # will store parameters extracted from each “Proceso i” sheet

    # 2) For each Proceso i: read into a pandas DataFrame, extract H1..H7 parameters, keep only needed columns
    for i in range(1, valor + 1):
        # Read sheet “Proceso i” into a pandas DataFrame
        df_i = pd.read_excel(input_path, sheet_name=f"Proceso {i}")
        sheet_i = workbook[f"Proceso {i}"]

        # Extract parameters (H1 through H7)
        turno = sheet_i["H1"].value
        horas = sheet_i["H2"].value
        descanso = sheet_i["H3"].value
        productividad = sheet_i["H4"].value
        productividad_objetivo = sheet_i["H6"].value
        trabajadores = sheet_i["H7"].value

        # Save these parameters in a dictionary
        params[i] = {
            "turno": turno,
            "horas": horas,
            "descanso": descanso,
            "productividad": productividad,
            "productividad_objetivo": productividad_objetivo,
            "trabajadores": trabajadores
        }

        # Keep only the columns ['Día', 'Turno', 'Carga histórica', 'Número de FTE']
        df_i = df_i[["Día", "Turno", "Carga histórica", "Número de FTE"]]

        # Convert “Día” → new “Date” column (datetime), then drop the original “Día”
        df_i["Date"] = pd.to_datetime(df_i["Día"], format="%Y-%m-%d", errors="coerce")
        df_i["Date"] = df_i["Date"].dt.strftime("%m/%d/%y")
        df_i["Date"] = pd.to_datetime(df_i["Date"], format="%m/%d/%y", errors="coerce")
        df_i = df_i.drop(columns=["Día"])

        # Add DayOfWeek column
        df_i["DayOfWeek"] = df_i["Date"].dt.day_name()

        # Store the cleaned DataFrame
        dfs[i] = df_i

    # 3) Split each df_i into train/test by last-month logic
    trains = {}
    tests = {}
    for i in range(1, valor + 1):
        df_i = dfs[i]
        last_month = df_i["Date"].dt.to_period("M").max()
        df_i_train = df_i[df_i["Date"].dt.to_period("M") < last_month].copy()
        df_i_test = df_i[df_i["Date"].dt.to_period("M") == last_month].copy()
        trains[i] = df_i_train
        tests[i] = df_i_test

    # 4) For convenience, let df_1 be the full DataFrame for Proceso 1
    df_1 = dfs[1]
    df_1_train = trains[1]
    df_1_test = tests[1]

    # 5) ----- MONTE CARLO SIMULATION on df_1_train -----
    def simulacion_montecarlo(df: pd.DataFrame, num_dias: int = 30) -> pd.DataFrame:
        valores_historicos = df["Carga histórica"].dropna().values
        media = np.mean(valores_historicos)
        desviacion = np.std(valores_historicos)
        pred_normal = np.random.normal(media, desviacion, num_dias)
        pred_bootstrap = np.random.choice(valores_historicos, num_dias, replace=True)
        params_fit = norm.fit(valores_historicos)
        pred_empirica = norm.rvs(*params_fit, size=num_dias)
        fechas_futuras = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=num_dias)
        predicciones_df = pd.DataFrame({
            "Fecha": fechas_futuras,
            "Distribución Normal": pred_normal,
            "Bootstrapping": pred_bootstrap,
            "Distribución Empírica": pred_empirica
        })
        return predicciones_df

    predicciones_mc = simulacion_montecarlo(df_1_train, num_dias=30)

    # 6) ----- SARIMA / AutoARIMA FORECAST on df_1_train -----
    # Prepare time series for SARIMA
    ts_data = df_1_train.set_index("Date")["Carga histórica"].sort_index()

    # SARIMA with weekly seasonality = 5 business days (or adjust as needed)
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 5)
    sarima_model = SARIMAX(ts_data, order=sarima_order, seasonal_order=seasonal_order, trend="t")
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=30)

    # AutoARIMA variants (six runs with different seasonal/trend combos)
    def run_auto_arima(data_series, seasonal: bool, m: int, trend: str):
        model = pm.auto_arima(
            data_series,
            seasonal=seasonal,
            m=m,
            trend=trend,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True
        )
        return model.predict(n_periods=30)

    auto1 = run_auto_arima(ts_data, seasonal=True, m=5, trend="t")
    auto2 = run_auto_arima(ts_data, seasonal=True, m=5, trend="n")
    auto3 = run_auto_arima(ts_data, seasonal=True, m=12, trend="t")
    auto4 = run_auto_arima(ts_data, seasonal=True, m=12, trend="n")
    auto5 = run_auto_arima(ts_data, seasonal=False, m=1, trend="t")
    auto6 = run_auto_arima(ts_data, seasonal=False, m=1, trend="n")

    # Build a DataFrame of all ARIMA forecasts
    df_arima = pd.concat(
        [
            pd.Series(auto1, name="Seasonal Week & Trend"),
            pd.Series(auto2, name="Seasonal Week & No Trend"),
            pd.Series(auto3, name="Seasonal Month & Trend"),
            pd.Series(auto4, name="Seasonal Month & No Trend"),
            pd.Series(auto5, name="Not seasonal & Trend"),
            pd.Series(auto6, name="Not seasonal & No Trend"),
            pd.Series(sarima_forecast.values, name="Sarima")
        ],
        axis=1
    )

    # 7) ----- XGBOOST FORECAST on df_1_train -----
    def create_xgb_features(df: pd.DataFrame, label_col: str = None):
        df_ = df.copy()
        df_["year"] = df_["Date"].dt.year
        df_["dayofmonth"] = df_["Date"].dt.day
        df_["dayofyear"] = df_["Date"].dt.dayofyear
        df_["weekofyear"] = df_["Date"].dt.isocalendar().week
        df_["quarter"] = df_["Date"].dt.quarter
        df_["dayofweek"] = df_["Date"].dt.dayofweek
        df_["month"] = df_["Date"].dt.month
        features = ["dayofweek", "month", "quarter", "year", "dayofmonth", "dayofyear", "weekofyear"]
        X = df_[features]
        if label_col:
            return X, df_[label_col]
        return X

    X_tr, y_tr = create_xgb_features(df_1_train, label_col="Carga histórica")
    X_te, y_te = create_xgb_features(df_1_test, label_col="Carga histórica")

    xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
    param_grid = {
        "n_estimators": [50, 200, 500],
        "max_depth": [1, 5, 10],
        "learning_rate": [0.005, 0.05, 0.15]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=False)
    grid.fit(X_tr, y_tr)

    best_xgb = grid.best_estimator_
    preds_xgb_test = best_xgb.predict(X_te)
    rmse_xgb = np.sqrt(mean_squared_error(y_te, preds_xgb_test))
    mae_xgb = mean_absolute_error(y_te, preds_xgb_test)

    # 8) ----- COMBINE ALL MODEL FORECASTS and PICK BEST by MAE against df_1_test -----
    # Build a DataFrame of “predicciones_mc” means if needed:
    # For Monte Carlo, use “Distribución Normal” column as its forecast
    mc_mean = predicciones_mc["Distribución Normal"].values

    # Actual df_1_test dates
    test_dates = df_1_test["Date"].reset_index(drop=True)

    # Assemble a comparison DataFrame
    df_compare = pd.DataFrame({
        "Fecha": test_dates,
        "MC": mc_mean[: len(df_1_test)],  # trim if needed
        "AutoARIMA_SW_T": auto1[: len(df_1_test)],
        "AutoARIMA_SW_N": auto2[: len(df_1_test)],
        "AutoARIMA_SM_T": auto3[: len(df_1_test)],
        "AutoARIMA_SM_N": auto4[: len(df_1_test)],
        "AutoARIMA_NS_T": auto5[: len(df_1_test)],
        "AutoARIMA_NS_N": auto6[: len(df_1_test)],
        "Sarima": sarima_forecast.values[: len(df_1_test)],
        "XGBoost": preds_xgb_test
    })

    # Compute MAE for each column
    maes = {}
    for col in df_compare.columns:
        if col == "Fecha":
            continue
        maes[col] = mean_absolute_error(df_1_test["Carga histórica"].values[: len(df_1_test)], df_compare[col].values)

    # Find the model name with smallest MAE
    best_model = min(maes, key=maes.get)

    # 9) ----- RE-TRAIN BEST MODEL ON FULL df_1 and PREDICT NEXT 30 DAYS -----
    if best_model == "XGBoost":
        # Refit XGBoost on the entire df_1 (train + test)
        X_full, y_full = create_xgb_features(df_1, label_col="Carga histórica")
        best_xgb.fit(X_full, y_full)

        # Build future features for next 30 calendar days
        last_date_full = df_1["Date"].max()
        future_dates = pd.date_range(start=last_date_full + pd.Timedelta(days=1), periods=30, freq="D")
        df_future_feat = pd.DataFrame({"Date": future_dates})
        X_future = create_xgb_features(df_future_feat, label_col=None)
        preds_future = best_xgb.predict(X_future)

    else:
        # If best model is SARIMA or AutoARIMA/SW_T, use that forecast for the next 30 days
        # We already have sarima_forecast and auto1..auto6. Pick the array that matches best_model:
        if best_model == "Sarima":
            preds_future = sarima_forecast.values
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_SW_T"):
            preds_future = auto1
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_SW_N"):
            preds_future = auto2
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_SM_T"):
            preds_future = auto3
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_SM_N"):
            preds_future = auto4
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_NS_T"):
            preds_future = auto5
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        elif best_model.startswith("AutoARIMA_NS_N"):
            preds_future = auto6
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
        else:
            # Use Monte Carlo “Distribución Normal” as fallback
            preds_future = predicciones_mc["Distribución Normal"].values
            future_dates = pd.date_range(start=df_1["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")

    # 10) ----- UPDATE/CREATE the “Forecast” sheet via openpyxl -----
    # Delete old Forecast sheet if it still exists
    if "Forecast" in workbook.sheetnames:
        del workbook["Forecast"]
    sheet_fc = workbook.create_sheet(title="Forecast")
    bold_font = openpyxl.styles.Font(bold=True)

    # Build a DataFrame for future_dates/preds_future
    df_final = pd.DataFrame({
        "Día": future_dates,
        "Demanda": preds_future
    })

    # Write headers (A1:D1)
    sheet_fc["A1"].value = "Día"
    sheet_fc["A1"].font = bold_font
    sheet_fc["B1"].value = "Demanda"
    sheet_fc["B1"].font = bold_font
    sheet_fc["C1"].value = "Edición"
    sheet_fc["C1"].font = bold_font
    sheet_fc["D1"].value = "Demanda Final"
    sheet_fc["D1"].font = bold_font

    # Write data rows (row 2..)
    for idx, row in enumerate(df_final.itertuples(index=False), start=2):
        sheet_fc.cell(row=idx, column=1).value = row.Día
        sheet_fc.cell(row=idx, column=2).value = float(row.Demanda)
        # Pre-fill column D with formula: =IF(Crow=0, Brow, Crow)
        sheet_fc.cell(row=idx, column=4).value = f"=IF(C{idx}=0, B{idx}, C{idx})"

    # Fill column C (editable) with yellow background
    fill_yellow = openpyxl.styles.PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    for r in range(2, 2 + len(df_final)):
        sheet_fc.cell(row=r, column=3).fill = fill_yellow

    # Add FTE columns E, F, G using params[1] (for Proceso 1)
    p1 = params[1]
    sheet_fc["E1"].value = "Número de FTE actuales"
    sheet_fc["E1"].font = bold_font
    sheet_fc["F1"].value = "Productividad (u/hh)"
    sheet_fc["F1"].font = bold_font
    sheet_fc["G1"].value = "Productividad objetivo (u/hh)"
    sheet_fc["G1"].font = bold_font

    for r in range(2, 2 + len(df_final)):
        # Use formulas that reference the “Proceso 1” sheet’s H7, H4, H6
        sheet_fc.cell(row=r, column=5).value = f"='Proceso 1'!$H$7"
        sheet_fc.cell(row=r, column=6).value = f"='Proceso 1'!$H$4"
        sheet_fc.cell(row=r, column=7).value = f"='Proceso 1'!$H$6"

    # Heijunka 1 formulas (J→T)
    for r in range(2, 2 + len(df_final)):
        sheet_fc.cell(row=r, column=10).value = f"=D{r}/F{r}" \
            # J
        sheet_fc.cell(row=r, column=11).value = f"=J{r}/('Proceso 1'!$H$2-('Proceso 1'!$H$3/60))" \
            # K
        sheet_fc.cell(row=r, column=12).value = f"=D{r}/G{r}" \
            # L
        sheet_fc.cell(row=r, column=13).value = f"=L{r}/('Proceso 1'!$H$2-('Proceso 1'!$H$3/60))" \
            # M
        sheet_fc.cell(row=r, column=14).value = f"=J{r}-L{r}" \
            # N
        sheet_fc.cell(row=r, column=15).value = f"=K{r}-M{r}" \
            # O
        sheet_fc.cell(row=r, column=16).value = f"=J{r}/(E{r}*('Proceso 1'!$H$2-('Proceso 1'!$H$3/60)))" \
            # P
        sheet_fc.cell(row=r, column=17).value = f"=J{r}-(E{r}*('Proceso 1'!$H$2-('Proceso 1'!$H$3/60)))" \
            # Q
        sheet_fc.cell(row=r, column=18).value = 25  # R
        sheet_fc.cell(row=r, column=19).value = 18  # S
        sheet_fc.cell(row=r, column=20).value = f"=IF(Q{r}>0,Q{r}*R{r},-Q{r}*S{r})"  # T

    # Heijunka 2 headers (V→AB) and defaults
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

    # Place a default “1” in AD3 and add labels in AC3/AC4
    sheet_fc["AD3"].value = 1
    sheet_fc["AC3"].value = "Número de FTE en el mes"
    sheet_fc["AC3"].font = bold_font
    sheet_fc["AC4"].value = "Sobrecoste en el mes (euros)"
    sheet_fc["AC4"].font = bold_font

    # Compute and write optimal W in AD3, SUM formula in AD4
    J_vals = df_final["Demanda"].values / p1["productividad"]
    H2 = workbook["Proceso 1"]["H2"].value
    H3 = workbook["Proceso 1"]["H3"].value
    avail = H2 - (H3 / 60.0)
    R, S = 25.0, 18.0

    def total_cost(W: float) -> float:
        X = J_vals - W * avail
        AA = np.where(X > 0, X * R, -X * S)
        return AA.sum()

    Ws = np.arange(0, 5001, 1)
    costs = np.array([total_cost(w) for w in Ws])
    idx_min = costs.argmin()
    W_opt = Ws[idx_min]

    sheet_fc["AD3"].value = W_opt
    sheet_fc["AD4"].value = f"=SUM(AA2:AA{1 + len(df_final)})"

    # 11) Apply number formatting (two decimals) to relevant columns
    two_decimals = NamedStyle(name="two_decimals")
    two_decimals.number_format = "0.00"
    workbook.add_named_style(two_decimals)

    cols_to_style = [
        1, 2, 3, 4,        # A,B,C,D
        9, 10, 11, 12,     # I,J,K,L
        13, 14, 15, 16,    # M,N,O,P
        17, 20,            # Q, T
        22, 23, 24,        # V, W, X
        27, 28, 29, 30     # AA, AB, AC, AD
    ]
    for col_idx in cols_to_style:
        letter = sheet_fc.cell(row=1, column=col_idx).column_letter
        for cell in sheet_fc[letter]:
            if cell.row > 1:
                cell.style = two_decimals

    # 12) Apply pink (I→T) and blue (V→AD) backgrounds; C is already yellow
    pink_fill = openpyxl.styles.PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
    blue_fill = openpyxl.styles.PatternFill(start_color="CCFFFF", end_color="CCFFFF", fill_type="solid")
    yellow_fill = openpyxl.styles.PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")

    for r in range(2, 2 + len(df_final)):
        sheet_fc[f"C{r}"].fill = yellow_fill
        for c in range(9, 21):   # I(9) through T(20)
            sheet_fc.cell(row=r, column=c).fill = pink_fill
        for c in range(22, 31):  # V(22) through AD(30)
            sheet_fc.cell(row=r, column=c).fill = blue_fill

    # 13) Auto-size columns & adjust row heights
    for col_cells in sheet_fc.columns:
        max_length = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            if cell.value is not None:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        sheet_fc.column_dimensions[col_letter].width = max_length + 2

    for row_cells in sheet_fc.iter_rows():
        max_lines = 1
        for cell in row_cells:
            if isinstance(cell.value, str):
                lines = cell.value.split("\n")
                if len(lines) > max_lines:
                    max_lines = len(lines)
        sheet_fc.row_dimensions[row_cells[0].row].height = max_lines * 15

    # 14) Save the updated workbook
    workbook.save(output_path)


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
