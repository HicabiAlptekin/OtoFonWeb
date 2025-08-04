import streamlit as st
import pandas as pd
from analiz_script import calistir_analiz
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="OtoFon Analiz Aracı", layout="wide")

st.title("OtoFon - Otomatik Fon Analiz Aracı")
st.write("Belirtilen fon listesi için son 3 aylık performansı analiz eder ve sonuçları indirilebilir bir Excel dosyası olarak sunar.")

# --- YARDIMCI FONKSIYON ---
@st.cache_data
def to_excel(df):
    """DataFrame'i Excel formatında bir byte dizisine dönüştürür."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Fon Analizi', index=False)
        worksheet = writer.sheets['Fon Analizi']
        # Sütun genişliklerini ayarla
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)
    processed_data = output.getvalue()
    return processed_data

# --- ANA UYGULAMA AKIŞI ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.results_df = None
    st.session_state.end_date = None

if st.button("Fon Analizini Başlat", type="primary"):
    with st.spinner("Fon verileri çekiliyor ve analiz ediliyor... Bu işlem birkaç dakika sürebilir."):
        try:
            df_sonuclar, tarih = calistir_analiz()
            if df_sonuclar is not None:
                st.session_state.results_df = df_sonuclar
                st.session_state.end_date = tarih
                st.session_state.analysis_done = True
                st.success("Analiz başarıyla tamamlandı!")
            else:
                st.error("Analiz için yeterli veri bulunamadı. Lütfen daha sonra tekrar deneyin.")
                st.session_state.analysis_done = False
        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")
            st.session_state.analysis_done = False

if st.session_state.analysis_done:
    st.subheader("Analiz Sonuçları")
    st.dataframe(st.session_state.results_df)
    
    excel_data = to_excel(st.session_state.results_df)
    
    st.download_button(
        label="📥 Excel Olarak İndir",
        data=excel_data,
        file_name=f"Hisse_Senedi_Fon_Analizi_{st.session_state.end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
