import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import datetime, date
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

# ===================== 1. 頁面設定 =====================
st.set_page_config(page_title="全方位績效分析", layout="wide")
st.title("📊 全方位業務績效診斷系統")

# 設定中文字型 (避免 Matplotlib 畫圖亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    import openpyxl
except ImportError:
    st.error("❌ 缺少套件：openpyxl，請執行 pip install openpyxl")
    st.stop()

# ===================== 2. 工具函式 (共用 & AI專用) =====================

# --- 2.1 原本的日期處理 ---
def parse_date_robust(v):
    if pd.isna(v): return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime)): return v.date()
    if isinstance(v, date): return v
    s = str(v).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try: return datetime.strptime(digits[:8], "%Y%m%d").date()
        except: pass
    if len(digits) >= 6:
        try: return datetime.strptime(digits[:6] + "01", "%Y%m%d").date()
        except: pass
    try: return pd.to_datetime(v).date()
    except: return pd.NaT

def int_to_yw(v):
    if pd.isna(v): return (np.nan, np.nan)
    v = int(v)
    return v // 100, v % 100

def weeks_diff(y1, w1, y2, w2):
    if pd.isna(y1) or pd.isna(w1) or pd.isna(y2) or pd.isna(w2): return np.nan
    try:
        d1 = date.fromisocalendar(int(y1), int(w1), 1)
        d2 = date.fromisocalendar(int(y2), int(w2), 1)
        return int((d2 - d1).days // 7)
    except: return np.nan

def iso_year_week(d):
    if pd.isna(d): return (np.nan, np.nan)
    iso = d.isocalendar()
    return (int(iso[0]), int(iso[1]))

def yw_to_int(y, w):
    if pd.isna(y) or pd.isna(w): return np.nan
    try: return int(f"{int(y)}{int(w):02d}")
    except: return np.nan

# --- 2.2 AI 專用函式 (新加入) ---
def parse_date_num(v):
    # 用於 AI 計算 (轉成 ordinal 數字)
    if pd.isna(v): return np.nan
    s = str(v).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try: return datetime.strptime(digits[:8], "%Y%m%d").toordinal()
        except: pass
    return np.nan

def parse_target_date_num(v):
    s = str(v).strip()
    if len(s) == 6:
        try: return datetime.strptime(s, "%Y%m").toordinal()
        except: pass
    return np.nan

def process_data_ai(df, le_sales=None, le_prod=None, is_training=True):
    # AI 專用的資料前處理
    # 嘗試找日期欄位
    date_col = "DONE_YEAR_MON" if "DONE_YEAR_MON" in df.columns else "DONE_DATE"
    if date_col not in df.columns:
        return None, None, None # 資料格式不對
        
    df['DONE_DATE_NUM'] = df[date_col].apply(parse_date_num)
    
    # 確保有目標日期 (REQUEST_DATE)
    if 'REQUEST_DATE' not in df.columns:
        # 如果沒有，嘗試從 BFC_QTY 欄位名稱或其他地方推論? 這裡假設必須要有
        return None, None, None

    df['TARGET_DATE_NUM'] = df['REQUEST_DATE'].apply(parse_target_date_num)
    df['WEEKS_BEFORE'] = (df['TARGET_DATE_NUM'] - df['DONE_DATE_NUM']) / 7
    
    req_cols = ['WEEKS_BEFORE', 'BFC_QTY']
    if is_training: req_cols.append('SHIPMENT_QTY')
    df = df.dropna(subset=req_cols)

    if is_training:
        le_sales = LabelEncoder()
        df['SALES_CODE'] = le_sales.fit_transform(df['SALES_NAME'].astype(str))
        le_prod = LabelEncoder()
        df['PRODUCT_CODE'] = le_prod.fit_transform(df['PRODUCT_NAME'].astype(str))
        return df, le_sales, le_prod
    else:
        # 處理新標籤
        if le_sales:
            df['SALES_CODE'] = df['SALES_NAME'].apply(lambda x: le_sales.transform([str(x)])[0] if str(x) in le_sales.classes_ else -1)
        if le_prod:
            df['PRODUCT_CODE'] = df['PRODUCT_NAME'].apply(lambda x: le_prod.transform([str(x)])[0] if str(x) in le_prod.classes_ else -1)
        mask = (df['SALES_CODE'] != -1) & (df['PRODUCT_CODE'] != -1)
        return df[mask].copy(), None, None

def train_ai_models(df_train):
    train_processed, le_s, le_p = process_data_ai(df_train, is_training=True)
    if train_processed is None: return None, None, None, None, None
    
    features = ['BFC_QTY', 'WEEKS_BEFORE', 'SALES_CODE', 'PRODUCT_CODE']
    X_train = train_processed[features]
    y_reg = train_processed['SHIPMENT_QTY']
    # 準確定義：誤差 <= 20%
    y_cls = train_processed.apply(lambda row: 1 if (row['SHIPMENT_QTY']>0 and abs(row['BFC_QTY']-row['SHIPMENT_QTY'])/row['SHIPMENT_QTY']<=0.2) or (row['SHIPMENT_QTY']==0 and row['BFC_QTY']==0) else 0, axis=1)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_reg)
    
    nb = GaussianNB()
    nb.fit(X_train, y_cls)
    
    return rf, nb, le_s, le_p, features

def run_ai_prediction(df_test, rf, nb, le_s, le_p, features, has_actual=False):
    test_processed, _, _ = process_data_ai(df_test, le_s, le_p, is_training=False)
    if test_processed is None or test_processed.empty: return None, None, None
    
    X_test = test_processed[features]
    
    test_processed['AI_修正預測'] = rf.predict(X_test).round(0)
    test_processed['AI_信心度(%)'] = (nb.predict_proba(X_test)[:, 1] * 100).round(1)
    
    conds = [(test_processed['AI_信心度(%)'] < 50), (test_processed['AI_信心度(%)'] >= 50)]
    choices = ['⚠️ 高風險', '✅ 可信賴']
    test_processed['AI_建議'] = np.select(conds, choices, default='觀察中')
    test_processed['預測差異'] = test_processed['AI_修正預測'] - test_processed['BFC_QTY']
    
    if has_actual and 'SHIPMENT_QTY' in test_processed.columns:
        test_processed['實際出貨'] = test_processed['SHIPMENT_QTY']
        test_processed['業務誤差'] = abs(test_processed['BFC_QTY'] - test_processed['實際出貨'])
        test_processed['AI_誤差'] = abs(test_processed['AI_修正預測'] - test_processed['實際出貨'])
        test_processed['AI_勝出'] = test_processed['AI_誤差'] < test_processed['業務誤差']
        
    return test_processed, X_test, nb


# ===================== 3. 核心運算 (原本的) =====================

@st.cache_data
def process_data_final(file, threshold_x_error):
    # 注意：這裡加上 file.seek(0) 確保如果被讀過還能再讀
    file.seek(0)
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"讀取失敗: {e}")
        st.stop()
        
    colmap = {
        "SALES_NAME": "SALES_NAME",
        "PRODUCT_NAME": "PRODUCT_NAME",
        "AREA_NAME": "AREA_NAME",
        "PP_GROUP_ID": "PP_GROUP_ID",
        "DONE_DATE": "DONE_YEAR_MON" if "DONE_YEAR_MON" in df.columns else "DONE_DATE",
        "BFC_QTY": "BFC_QTY",
        "SHIPMENT_QTY": "SHIPMENT_QTY",
    }
    
    dfn = pd.DataFrame()
    for t, s in colmap.items():
        if s in df.columns: dfn[t] = df[s]
        else: dfn[t] = np.nan
            
    dfn["BFC_QTY"] = pd.to_numeric(dfn["BFC_QTY"], errors="coerce").fillna(0)
    dfn["SHIPMENT_QTY"] = pd.to_numeric(dfn["SHIPMENT_QTY"], errors="coerce")
    dfn["DONE_DATE"] = dfn["DONE_DATE"].apply(parse_date_robust)
    
    done_yw = dfn["DONE_DATE"].apply(iso_year_week)
    dfn["DONE_YW"] = [yw_to_int(y, w) for y, w in done_yw]
    
    df_cleaned = dfn[~dfn["SHIPMENT_QTY"].isna() & ~dfn["DONE_YW"].isna()].copy()
    df_cleaned.sort_values(["DONE_YW", "DONE_DATE"], inplace=True)
    
    GROUP_KEYS = ["SALES_NAME", "PRODUCT_NAME", "AREA_NAME", "PP_GROUP_ID"]
    if df_cleaned["PP_GROUP_ID"].isna().all(): GROUP_KEYS.remove("PP_GROUP_ID")
    
    latest = df_cleaned.drop_duplicates(subset=GROUP_KEYS + ["DONE_YW"], keep="last").copy()
    
    pred = latest["BFC_QTY"].astype(float)
    act = latest["SHIPMENT_QTY"].astype(float)
    latest["ABS_PCT_ERR"] = np.where(act==0, np.nan, (pred-act).abs()/act)
    latest["ACCURACY_SCORE"] = (1 - latest["ABS_PCT_ERR"]).clip(0, 1)

    def calc_metrics(g):
        g = g.sort_values("DONE_YW")
        if g.empty: return pd.Series()

        start_yw = g.iloc[0]["DONE_YW"]
        sy, sw = int_to_yw(start_yw)
        actual_qty = g.iloc[-1]["SHIPMENT_QTY"]
        final_forecast = g.iloc[-1]["BFC_QTY"]
        version_count = len(g)
        
        # 1. 製造指標
        bx = np.nan
        if actual_qty > 0:
            x_mask = g["ABS_PCT_ERR"] <= threshold_x_error
            if x_mask.any():
                f = g.loc[x_mask].iloc[0]
                fy, fw = int_to_yw(f["DONE_YW"])
                bx = weeks_diff(sy, sw, fy, fw)
        
        final_acc_pct = 0.0
        if actual_qty > 0:
            err = abs(final_forecast - actual_qty) / actual_qty
            final_acc_pct = max(0.0, 1.0 - err)

        # 2. 採購指標 (嚴格加權)
        weighted_score = 0.0
        score_breakdown = {"p1":0, "p2":0, "p3":0}
        
        if actual_qty > 0:
            scores = g["ACCURACY_SCORE"].values
            p1 = scores[0:8]
            p2 = scores[8:16]
            p3 = scores[16:]
            
            s1 = np.mean(p1) if len(p1) > 0 else 0.0
            s2 = np.mean(p2) if len(p2) > 0 else 0.0
            s3 = np.mean(p3) if len(p3) > 0 else 0.0
            
            w1, w2, w3 = 0.7, 0.2, 0.1
            weighted_score = (s1 * w1) + (s2 * w2) + (s3 * w3)
            score_breakdown = {"p1":s1, "p2":s2, "p3":s3}

        return pd.Series({
            "實際銷量": actual_qty,
            "最終預測": final_forecast,
            "版本總數": version_count,
            "反應速度X": bx,
            "最終準確率Y": final_acc_pct,
            "加權準確率_採購": weighted_score,
            "P1得分": score_breakdown["p1"],
            "P2得分": score_breakdown["p2"],
            "P3得分": score_breakdown["p3"]
        })

    summary = latest.groupby(GROUP_KEYS, dropna=False).apply(calc_metrics).reset_index()
    return summary, df_cleaned

# ===================== 4. 介面呈現 =====================

uploaded_file = st.file_uploader("📂 請上傳 Excel 檔案", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # 側邊欄設定
    st.sidebar.header("👁️ 分析視角")
    view_mode = st.sidebar.radio("請選擇模式", [
        "1. 製造視角 (反應速度 vs 最終準確率)", 
        "2. 採購視角 (採購分數 vs 最終準確率)",
        "3. 綜合視角 (反應速度 vs 採購分數)",
        "4. 🔮 AI 預測實驗室 (測試中)"  # <--- 新增的選項
    ])
    
    st.sidebar.markdown("---")

    # ================= 模式 1~3: 原本的績效診斷 =================
    if not view_mode.startswith("4"):
        st.sidebar.header("⚙️ 參數設定")
        cutoff_x = st.sidebar.slider("反應速度門檻 (週)", 4, 20, 8)
        cutoff_weighted = st.sidebar.slider("採購加權分數門檻 (%)", 50, 100, 75) / 100.0
        cutoff_y = st.sidebar.slider("最終準確率門檻 (%)", 50, 100, 80) / 100.0
        thr_error = st.sidebar.number_input("誤差認定 (0.2=20%)", 0.05, 0.5, 0.2)

        with st.spinner('🚀 運算中...'):
            df_res, df_raw = process_data_final(uploaded_file, thr_error)
            
        # 篩選 (原本的邏輯)
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 篩選資料")
        all_areas = sorted(df_res["AREA_NAME"].astype(str).unique())
        sel_area = st.sidebar.multiselect("地區", all_areas)
        df_s1 = df_res[df_res["AREA_NAME"].isin(sel_area)] if sel_area else df_res
        
        avail_sales = sorted(df_s1["SALES_NAME"].astype(str).unique())
        sel_sales = st.sidebar.multiselect("業務", avail_sales)
        df_s2 = df_s1[df_s1["SALES_NAME"].isin(sel_sales)] if sel_sales else df_s1
        
        if "PP_GROUP_ID" in df_res.columns:
            avail_groups = sorted(df_s2["PP_GROUP_ID"].astype(str).unique())
            sel_group = st.sidebar.multiselect("產品群組", avail_groups)
            df_s3 = df_s2[df_s2["PP_GROUP_ID"].isin(sel_group)] if sel_group else df_s2
        else: df_s3 = df_s2
            
        avail_prod = sorted(df_s3["PRODUCT_NAME"].astype(str).unique())
        sel_prod = st.sidebar.multiselect("產品", avail_prod)
        df_show = df_s3[df_s3["PRODUCT_NAME"].isin(sel_prod)] if sel_prod else df_s3

        # 圖表邏輯...
        max_x = max(df_show["反應速度X"].max(), 20)
        if pd.isna(max_x): max_x=20
        df_show["Plot_X"] = df_show["反應速度X"].fillna(max_x + 2)
        chart_key = f"main_chart_{view_mode[:2]}" 

        if view_mode.startswith("1"):
            st.subheader("🏭 製造視角：反應速度 vs 最終準確率")
            st.info("目標：↖️ 左上角 (黃金區)。")
            def get_quad_mfg(row):
                x, y = row["反應速度X"], row["最終準確率Y"]
                x_pass = (pd.notna(x) and x <= cutoff_x)
                y_pass = (y >= cutoff_y)
                if x_pass and y_pass: return "1.黃金區 (快且準)"
                elif not x_pass and y_pass: return "2.保守區 (慢但準)"
                elif x_pass and not y_pass: return "3.賭徒區 (快但不準)"
                else: return "4.問題區 (慢且不準)"
            df_show["象限"] = df_show.apply(get_quad_mfg, axis=1)
            fig = px.scatter(
                df_show, x="Plot_X", y="最終準確率Y", color="象限",
                custom_data=["SALES_NAME", "PRODUCT_NAME"],
                hover_data=["SALES_NAME", "PRODUCT_NAME", "實際銷量"],
                color_discrete_map={"1.黃金區 (快且準)": "green", "2.保守區 (慢但準)": "orange", "3.賭徒區 (快但不準)": "purple", "4.問題區 (慢且不準)": "red"},
                labels={"Plot_X": "反應速度 (週)", "最終準確率Y": "最終準確率 (%)"}
            )
            fig.add_vline(x=cutoff_x, line_dash="dash", line_color="red")
            fig.add_hline(y=cutoff_y, line_dash="dash", line_color="red")
            fig.update_xaxes(range=[-1, max_x+5], title="反應速度 (週) -> 越左越好")
            fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="最終準確率 (%) -> 越上越好")

        elif view_mode.startswith("2"):
            st.subheader("🛒 採購視角：初期加權分數 vs 最終準確率")
            st.info("目標：↗️ 右上角 (模範生)。")
            def get_quad_proc(row):
                w_score = row["加權準確率_採購"]
                f_score = row["最終準確率Y"]
                x_pass = (w_score >= cutoff_weighted)
                y_pass = (f_score >= cutoff_y)
                if x_pass and y_pass: return "A. 模範生 (初期準+最後準)"
                elif not x_pass and y_pass: return "B. 補救型 (初期錯+最後準)"
                elif x_pass and not y_pass: return "C. 虎頭蛇尾 (初期準+最後錯)"
                else: return "D. 狀況外 (全錯)"
            df_show["象限"] = df_show.apply(get_quad_proc, axis=1)
            fig = px.scatter(
                df_show, x="加權準確率_採購", y="最終準確率Y", color="象限",
                custom_data=["SALES_NAME", "PRODUCT_NAME"],
                hover_data=["SALES_NAME", "PRODUCT_NAME", "實際銷量"],
                color_discrete_map={"A. 模範生 (初期準+最後準)": "green", "B. 補救型 (初期錯+最後準)": "orange", "C. 虎頭蛇尾 (初期準+最後錯)": "red", "D. 狀況外 (全錯)": "gray"},
                labels={"加權準確率_採購": "採購加權分數", "最終準確率Y": "出貨準確率"}
            )
            fig.add_vline(x=cutoff_weighted, line_dash="dash", line_color="red")
            fig.add_hline(y=cutoff_y, line_dash="dash", line_color="red")
            fig.update_xaxes(range=[0, 1.1], tickformat=".0%", title="採購加權分數 (越右越好)")
            fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="最終出貨準確率 (越上越好)")

        elif view_mode.startswith("3"):
            st.subheader("⚖️ 綜合視角：製造速度 vs 採購品質")
            st.info("目標：↖️ 左上角 (完美區) 且 🟢綠燈。")
            def get_color_status(row):
                if row["最終準確率Y"] >= cutoff_y: return "🟢 最終出貨準確 (Pass)"
                else: return "🔴 最終出貨失敗 (Fail)"
            df_show["最終狀態"] = df_show.apply(get_color_status, axis=1)
            fig = px.scatter(
                df_show, x="Plot_X", y="加權準確率_採購", color="最終狀態",
                custom_data=["SALES_NAME", "PRODUCT_NAME"],
                hover_data=["SALES_NAME", "PRODUCT_NAME", "實際銷量", "最終準確率Y"],
                color_discrete_map={"🟢 最終出貨準確 (Pass)": "green", "🔴 最終出貨失敗 (Fail)": "red"},
                labels={"Plot_X": "反應速度 (週)", "加權準確率_採購": "採購加權分數"}
            )
            fig.add_vline(x=cutoff_x, line_dash="dash", line_color="gray")
            fig.add_hline(y=cutoff_weighted, line_dash="dash", line_color="gray")
            fig.update_xaxes(range=[-1, max_x+5], title="反應速度 (週) -> 越左越好")
            fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="採購加權分數 -> 越上越好")

        selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
        
        # 鑽取分析
        if selection and len(selection["selection"]["points"]) > 0:
            point_data = selection["selection"]["points"][0]
            sel_sales = point_data["customdata"][0]
            sel_prod = point_data["customdata"][1]
            target_row = df_res[(df_res["SALES_NAME"] == sel_sales) & (df_res["PRODUCT_NAME"] == sel_prod)]
            
            if not target_row.empty:
                selected_row = target_row.iloc[0]
                sel_actual = selected_row["實際銷量"]
                st.markdown("---")
                st.subheader(f"🔍 深度診斷：{sel_sales} - {sel_prod}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    history_df = df_raw[
                        (df_raw["SALES_NAME"] == sel_sales) & 
                        (df_raw["PRODUCT_NAME"] == sel_prod)
                    ].copy()
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(x=history_df["DONE_DATE"], y=history_df["BFC_QTY"], mode='lines+markers', name='預測量', line=dict(color='orange', width=3)))
                    fig_hist.add_trace(go.Scatter(x=history_df["DONE_DATE"], y=[sel_actual]*len(history_df), mode='lines', name='實際出貨量', line=dict(color='green', dash='dash')))
                    fig_hist.update_layout(title="預測版本演變史", xaxis_title="預測日期", yaxis_title="數量")
                    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{chart_key}")
                with col2:
                    st.markdown("### 📊 綜合指標成績")
                    st.metric("反應速度", f"{selected_row['反應速度X']:.0f} 週" if pd.notna(selected_row['反應速度X']) else "未達標")
                    st.metric("採購總分", f"{selected_row['加權準確率_採購']:.1%}")
                    final_acc = selected_row["最終準確率Y"]
                    if final_acc >= cutoff_y: st.success(f"最終準確率: {final_acc:.1%} (Pass)")
                    else: st.error(f"最終準確率: {final_acc:.1%} (Fail)")
                    if view_mode.startswith("2") or view_mode.startswith("3"):
                        st.divider()
                        st.write(f"前 8 週: {selected_row['P1得分']:.1%}")
                        st.progress(selected_row['P1得分'])
                        st.write(f"9~16週: {selected_row['P2得分']:.1%}")
                        st.progress(selected_row['P2得分'])

        st.markdown("---")
        st.subheader("📋 資料明細")
        df_table = df_show.drop(columns=["Plot_X", "象限", "P1得分", "P2得分", "P3得分", "最終狀態"], errors="ignore").copy()
        for c in ["最終準確率Y", "加權準確率_採購"]:
            df_table[c] = df_table[c].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_table, use_container_width=True)

    # ================= 模式 4: AI 預測實驗室 =================
    elif view_mode.startswith("4"):
        st.header("🔮 AI 預測實驗室 (測試中)")
        st.info("此模組使用 **隨機森林 (Random Forest)** 與 **貝氏分類器**，學習歷史誤差來優化預測。")

        tab1, tab2, tab3 = st.tabs(["🧪 單檔自動回測", "🚀 未來預測 (雙檔)", "🔬 模型健檢"])

        # === 頁籤 1: 單檔回測 ===
        with tab1:
            st.markdown("#### 自動切分最後一個月進行驗證")
            st.caption("使用您目前上傳的檔案，將最後一個月當作「未知未來」，驗證 AI 是否比業務準。")
            if st.button("開始回測"):
                with st.spinner('AI 正在切分資料並訓練...'):
                    # 重新讀取資料 (raw mode)
                    uploaded_file.seek(0)
                    df_all = pd.read_excel(uploaded_file, engine='openpyxl')
                    
                    # 檢查是否有 REQUEST_DATE
                    if 'REQUEST_DATE' not in df_all.columns:
                        st.error("❌ 資料中缺少 `REQUEST_DATE` 欄位，無法進行 AI 運算。")
                    else:
                        df_all['TARGET_YM_TEMP'] = df_all['REQUEST_DATE'].astype(str).str[:6]
                        all_months = sorted(df_all['TARGET_YM_TEMP'].unique())
                        
                        if len(all_months) < 2:
                            st.warning("⚠️ 資料月份不足 (只有一個月)，無法切分訓練集與測試集。")
                        else:
                            last_month = all_months[-1]
                            train_months = all_months[:-1]
                            
                            df_train = df_all[df_all['TARGET_YM_TEMP'].isin(train_months)].copy()
                            df_test = df_all[df_all['TARGET_YM_TEMP'] == last_month].copy()
                            
                            rf, nb, le_s, le_p, feats = train_ai_models(df_train)
                            if rf:
                                result, _, _ = run_ai_prediction(df_test, rf, nb, le_s, le_p, feats, has_actual=True)
                                
                                st.success(f"✅ 訓練：{train_months[0]}~{train_months[-1]} | 驗證：{last_month}")
                                if 'AI_勝出' in result.columns:
                                    win_rate = result['AI_勝出'].mean()
                                    st.metric("🏆 AI 對決勝率", f"{win_rate:.1%}", delta="比業務準的機率")
                                
                                st.dataframe(result[['SALES_NAME','PRODUCT_NAME','BFC_QTY','AI_修正預測','實際出貨','AI_勝出','AI_信心度(%)']].sort_values('AI_信心度(%)'), use_container_width=True)
                            else:
                                st.error("訓練失敗，請檢查資料格式。")

        # === 頁籤 2: 未來預測 ===
        with tab2:
            st.markdown("#### 預測下個月 (需要上傳新檔案)")
            st.caption("目前上傳的檔案將作為「歷史訓練資料」，請再上傳一個「未來預測檔」。")
            
            f_fut = st.file_uploader("📂 上傳未來預測資料 (含 REQUEST_DATE, BFC_QTY)", type=['xlsx'])
            
            if f_fut and st.button("🚀 執行預測"):
                with st.spinner('正在訓練模型並預測...'):
                    uploaded_file.seek(0)
                    df_h = pd.read_excel(uploaded_file, engine='openpyxl')
                    df_f = pd.read_excel(f_fut, engine='openpyxl')
                    
                    if 'REQUEST_DATE' in df_h.columns and 'REQUEST_DATE' in df_f.columns:
                        rf, nb, le_s, le_p, feats = train_ai_models(df_h)
                        if rf:
                            result, _, _ = run_ai_prediction(df_f, rf, nb, le_s, le_p, feats, has_actual=False)
                            st.success("✅ 預測完成！")
                            st.dataframe(result[['SALES_NAME','PRODUCT_NAME','BFC_QTY','AI_修正預測','預測差異','AI_信心度(%)','AI_建議']].sort_values('AI_信心度(%)'), use_container_width=True)
                            
                            # 下載
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                result.to_excel(writer, index=False)
                            st.download_button("📥 下載預測報告", buffer.getvalue(), "AI_Future_Prediction.xlsx")
                        else:
                            st.error("模型訓練失敗，請檢查歷史資料。")
                    else:
                        st.error("❌ 檔案缺少 `REQUEST_DATE` 欄位。")

        # === 頁籤 3: 模型健檢 ===
        with tab3:
            st.markdown("#### 檢視模型判斷力 (ROC/AUC)")
            if st.button("生成健檢圖表"):
                with st.spinner('繪製中...'):
                    uploaded_file.seek(0)
                    df_all = pd.read_excel(uploaded_file, engine='openpyxl')
                    
                    if 'REQUEST_DATE' in df_all.columns:
                        # 簡單切分 80/20
                        split_idx = int(len(df_all) * 0.8)
                        df_train = df_all.iloc[:split_idx].copy()
                        df_test = df_all.iloc[split_idx:].copy()
                        
                        rf, nb, le_s, le_p, feats = train_ai_models(df_train)
                        if rf:
                            result, X_test_data, nb_model = run_ai_prediction(df_test, rf, nb, le_s, le_p, feats, has_actual=True)
                            
                            if result is not None and nb_model is not None:
                                y_true_cls = df_test.apply(lambda row: 1 if (row['SHIPMENT_QTY']>0 and abs(row['BFC_QTY']-row['SHIPMENT_QTY'])/row['SHIPMENT_QTY']<=0.2) or (row['SHIPMENT_QTY']==0 and row['BFC_QTY']==0) else 0, axis=1)
                                y_prob = nb_model.predict_proba(X_test_data)[:, 1]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("ROC 曲線")
                                    fpr, tpr, _ = roc_curve(y_true_cls, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                                    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                                    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
                                    ax1.legend(loc="lower right")
                                    ax1.set_title("ROC Curve (越凸向左上越好)")
                                    st.pyplot(fig1)
                                
                                with col2:
                                    st.subheader("信心分佈")
                                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                                    sorted_idx = np.argsort(y_prob)
                                    ax2.scatter(range(len(y_prob)), y_prob[sorted_idx], c=y_true_cls.iloc[sorted_idx], cmap='coolwarm', alpha=0.6, s=10)
                                    ax2.axhline(0.5, color='gray', linestyle='--')
                                    ax2.set_title("信心度 vs 實際 (紅錯藍對)")
                                    st.pyplot(fig2)
                            else:
                                st.warning("測試資料不足，無法繪圖。")
                    else:
                        st.error("缺少 REQUEST_DATE。")

else:
    st.info("請上傳 Excel 檔案")