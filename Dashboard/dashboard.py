import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import re
import os
from PIL import Image

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Tyler, The Creator: MER Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("../Dataset/Tyler_Dashboard_V12.csv")
    return df

# Memuat dataset
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error("⚠️ File 'Tyler_Dashboard_V12.csv' tidak ditemukan")
    data_loaded = False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigasi Dashboard")

# Membuat menu pilihan
menu = st.sidebar.radio(
    "Pilih Menu Analisis:",
    ("1. Overview Diskografi", "2. Song Analysis", "3. Eksplorasi Topik (BERTopic)")
)



# ==========================================
# ROUTING HALAMAN
# ==========================================
if data_loaded:
    if menu == "1. Overview Diskografi":
        st.title("Overview Diskografi & Pemetaan Emosi Global")
        st.markdown("Peta persebaran emosi lagu Tyler, The Creator. Titik yang menjauhi garis diagonal biru menunjukkan **Disonansi Emosi**.")
        st.markdown("### Panel Kontrol")
        col_filter, col_info = st.columns([2, 1]) 
        
        with col_filter:
            daftar_album = df['album'].dropna().unique().tolist()
            album_pilihan = st.multiselect(
                "Pilih Era Album:", 
                options=daftar_album, 
                default=None,
                help="Hapus atau tambahkan album untuk melihat perubahan tren emosi."
            )
        
        df_filtered = df[df['album'].isin(album_pilihan)]
        
        # ---------------------------------------------------
        # METRIK RINGKASAN
        # ---------------------------------------------------
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        # Filter data plot agar tidak error saat hitung korelasi
        df_plot = df_filtered.dropna(subset=['sentiment_score', 'valence'])
        
        col1.metric("Total Lagu", f"{len(df_filtered)} Lagu")
        col2.metric("Avg Audio Valence", f"{df_filtered['valence'].mean():.2f}")
        col3.metric("Avg Sentiment", f"{df_filtered['sentiment_score'].mean():.2f}")
        
        # Kalkulasi Korelasi Pearson secara dinamis
        if not df_plot.empty and len(df_plot) > 1:
            korelasi = df_plot['valence'].corr(df_plot['sentiment_score'])
            if korelasi >= 0.2:
                status_teks = "Harmoni (Konsonansi)"
            else:
                status_teks = "- Terjadi Disonansi" 
                
            col4.metric(
                label="Korelasi Pearson", 
                value=f"{korelasi:.3f}",
                delta=status_teks 
            )
        else:
            col4.metric("Korelasi Pearson", "nan")
        st.divider()
        
        # Memisahkan Ghost Tracks
        df_plot = df_filtered.dropna(subset=['sentiment_score', 'valence'])
        jumlah_ghost = len(df_filtered) - len(df_plot)
        
        # ---------------------------------------------------
        # SCATTER PLOT
        # ---------------------------------------------------
        with st.container():
            fig = px.scatter(
                df_plot, 
                x='valence', 
                y='sentiment_score',
                color='roberta_label',
                color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'},
                hover_name='title',
                hover_data={
                    'album': True,
                    'dissonance': ':.3f',
                    'valence': ':.3f',
                    'sentiment_score': ':.3f',
                    'roberta_label': False 
                },
                labels={
                    'valence': 'Audio Valence (Spotify)',
                    'sentiment_score': 'Sentiment Score (RoBERTa)'
                },
                height=550
            )
            
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)
            
            fig.add_shape(
                type="line", line=dict(dash='dash', color='blue', width=2),
                x0=0, y0=0, x1=1, y1=1
            )
            
            fig.update_layout(
                xaxis=dict(range=[-0.05, 1.05]),
                yaxis=dict(range=[-0.05, 1.05]),
                legend_title_text='Prediksi Sentimen Lirik',
                template='plotly_white',
                margin=dict(l=20, r=20, t=30, b=20) 
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if jumlah_ghost > 0:
                st.caption(f"*Catatan: {jumlah_ghost} track (instrumental/skit) disembunyikan dari grafik karena tidak memiliki metrik sentimen.*")

        # ---------------------------------------------------
        # TOP 5 LAGU PALING DISONAN
        # ---------------------------------------------------
        st.markdown("---")
        st.markdown("### Top 5 Lagu Paling Disonan")
        st.markdown("Berdasarkan album yang kamu pilih, ini adalah lagu-lagu dengan kesenjangan tertinggi antara nada musik dan lirik.")
        
        if not df_plot.empty:
            # Mengurutkan dari disonansi tertinggi
            top_5 = df_plot.sort_values(by='dissonance', ascending=False).head(5)
            
            # Memformat dataframe agar rapi saat ditampilkan
            top_5_display = top_5[['title', 'album', 'valence', 'sentiment_score', 'dissonance', 'dominant_topic']].copy()
            top_5_display.columns = ['Judul Lagu', 'Album', 'Audio Valence', 'Sentimen Lirik', 'Skor Disonansi', 'Topik Dominan (LDA)']
            
            # Menampilkan sebagai tabel interaktif Streamlit
            st.dataframe(
                top_5_display.style.format({
                    'Audio Valence': '{:.3f}',
                    'Sentimen Lirik': '{:.3f}',
                    'Skor Disonansi': '{:.3f}'
                }), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Pilih minimal satu album untuk melihat data.")
            
            
    elif menu == "2. Song Analysis":
        st.title("Song Analysis")
        st.markdown("Analisis mendalam hubungan antara melodi (Spotify) dan lirik (RoBERTa) untuk setiap lagu.")

        base_path = os.path.join(os.path.dirname(__file__), "..", "Cover_Album")

        # Mapping Nama File 
        cover_album_files = {
            "Bastard": "Bastard.png",
            "Goblin": "Goblin.png",
            "Wolf": "Wolf.png",
            "Cherry Bomb": "Cherry_bomb.png",
            "Flower Boy": "Flower_boy.png",
            "IGOR": "Igor.png",
            "CALL ME IF YOU GET LOST": "CMIYGL.png",
            "CHROMAKOPIA": "Chromakopia.png",
            "DON'T TAP THE GLASS": "DTTG.png"
        }

        # FLOW FILTER (ALBUM DULU, BARU LAGU)
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            daftar_album = df['album'].dropna().unique().tolist()
            selected_album = st.selectbox("💿 Pilih Album:", options=daftar_album)
            
        df_filtered = df[df['album'] == selected_album]

        with col_sel2:
            daftar_lagu = sorted(df_filtered['title'].unique())
            selected_song_title = st.selectbox("🎵 Pilih Judul Lagu:", options=daftar_lagu)

        if selected_song_title:
            song_data = df_filtered[df_filtered['title'] == selected_song_title].iloc[0]
            st.divider()
            st.markdown(f"## {selected_song_title}")
            st.markdown(f"**Album:** {selected_album}")
            st.markdown("<br>", unsafe_allow_html=True)

            # ===================================================
            # BAGIAN ATAS: COVER & SPOTIFY (Kiri) | METRIK (Kanan)
            # ===================================================
            col_cover, col_metrics = st.columns([1, 1.2], gap="large")

            with col_cover:
                # Menampilkan Cover Album
                nama_file = cover_album_files.get(selected_album, "")
                path_lengkap = os.path.join(base_path, nama_file)
                
                if nama_file != "" and os.path.isfile(path_lengkap):
                    st.image(path_lengkap, use_container_width=True)
                else:
                    st.info(f"Cover album untuk '{selected_album}' tidak tersedia.")

                # MENAMPILKAN PEMUTAR SPOTIFY 
                st.markdown("<br>", unsafe_allow_html=True) 
                raw_spotify_id = str(song_data.get('spotify_id', '')).strip()
                
                if "open.spotify.com/track/" in raw_spotify_id:
                    clean_id = raw_spotify_id.split("open.spotify.com/track/")[1].split("?")[0]
                elif "spotify:track:" in raw_spotify_id:
                    clean_id = raw_spotify_id.split("spotify:track:")[1]
                else:
                    clean_id = raw_spotify_id

                if clean_id and clean_id.lower() != 'nan':
                    embed_code = f"""
                    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/{clean_id}?utm_source=generator&theme=0" 
                    width="100%" height="80" frameBorder="0" allowfullscreen="" 
                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
                    """
                    components.html(embed_code, height=90)
                else:
                    st.warning("Audio preview tidak tersedia untuk lagu ini.")

            with col_metrics:
                st.subheader("Metrik Emosi")
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Audio Valence", f"{song_data.get('valence', 0):.3f}")
                m_col2.metric("Sentimen Lirik", f"{song_data.get('sentiment_score', 0):.3f}")

                diss_val = song_data.get('dissonance', None)
                
                # Cek apakah nilai disonansi kosong (NaN)
                if pd.isna(diss_val):
                    nilai_tampil = 0
                    warna_bar = "#bdc3c7" 
                    teks_status = "TIDAK TERSEDIA (INSTRUMENTAL)"
                    teks_angka = "" # Kosongkan teks jika instrumental
                else:
                    nilai_tampil = diss_val
                    warna_bar = "#e74c3c" if diss_val >= 0.4 else "#2ecc71"
                    teks_status = "Disonansi TINGGI" if diss_val >= 0.4 else "Disonansi RENDAH"
                    teks_angka = f"{nilai_tampil:.3f}" # Format 3 angka di belakang koma

                fig_diss = go.Figure(go.Indicator(
                    mode = "gauge", 
                    value = nilai_tampil,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {
                        'text': f"<b>Skor Disonansi</b><br><span style='font-size:0.8em;color:{warna_bar}'>{teks_status}</span>", 
                        'font': {'size': 20}
                    },
                    gauge = {
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': warna_bar, 'thickness': 0.3},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#d1d1d1",
                        'steps': [
                            {'range': [0, 0.4], 'color': "#f1f8f5" if not pd.isna(diss_val) else "#f2f2f2"},
                            {'range': [0.4, 1], 'color': "#fdf2f2" if not pd.isna(diss_val) else "#e6e6e6"}] 
                    }
                ))

                fig_diss.update_layout(
                    autosize=True,
                    height=300, 
                    margin=dict(l=10, r=10, t=70, b=0), 
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Arial"},
                    annotations=[
                        dict(
                            x=0.5, y=0.15, 
                            text=teks_angka,
                            showarrow=False,
                            font=dict(size=50, color="white")
                        )
                    ]
                )                
                st.plotly_chart(fig_diss, use_container_width=True, config={'responsive': True})

            st.divider()

            # ===================================================
            # SPLIT SCREEN (LIRIK vs HASIL AI) 
            # ===================================================
            col_kiri, col_kanan = st.columns([1.3, 1])

            with col_kiri:
                st.markdown("### Lirik Lagu")
                with st.container(height=480): 
                    lirik_mentah = song_data['lyrics'] if 'lyrics' in song_data else song_data['clean_lyrics']
                    
                    if pd.isna(lirik_mentah) or str(lirik_mentah).strip() == "":
                        st.info("Lirik tidak tersedia (Instrumental).")
                    else:
                        lirik_bersih = re.sub(
                            r'\[(.*?)\]', 
                            lambda m: '[' + re.sub(r'\s+', ' ', m.group(1).replace('\n', ' ')).strip() + ']', 
                            str(lirik_mentah), 
                            flags=re.DOTALL
                        )
                        lirik_bersih = re.sub(
                            r'\((.*?)\)', 
                            lambda m: '(' + re.sub(r'\s+', ' ', m.group(1).replace('\n', ' ')).strip() + ')', 
                            lirik_bersih, 
                            flags=re.DOTALL
                        )
                        st.text(lirik_bersih)

            with col_kanan:
                st.markdown("### Ringkasan Analisis")
                
                # Menyiapkan Kamus Topik
                kamus_topik = {
                -1: "Outliers / Eksperimental (Lirik Acak/Unik)",
                 0: "Autobiografi, Evolusi Persona, dan Braggadocio", 
                 1: "Romansa, Infatuasi, dan Kerentanan Emosional", 
                 2: "Konflik Asmara, Patah Hati, dan Obsesi"
                }

                # Mengambil nilai skor untuk pengecekan
                sentimen_score = song_data.get('sentiment_score', None)
                valence_score = song_data.get('valence', None)
                topik_raw = song_data.get('dominant_topic', None)
                
                # CEK KONDISI: Apakah lagu instrumental/interlude? (Tidak punya sentimen lirik)
                is_instrumental = pd.isna(sentimen_score) or pd.isna(topik_raw)

                if not is_instrumental:
                    # ==========================================
                    # SKENARIO NORMAL (ADA LIRIK)
                    # ==========================================
                    topik_id = int(topik_raw)
                    deskripsi_topik = kamus_topik.get(topik_id, "Tema tidak terdefinisi")
                    label_sentimen = song_data.get('roberta_label', 'Unknown')

                    st.markdown(
                        f"Secara tematik, algoritma **BERTopic** mengklasifikasikan lirik lagu ini ke dalam **Topik {topik_id}**, "
                        f"yang kental dengan nuansa *\"{deskripsi_topik}\"*."
                    )
                    
                    st.markdown(
                        f"Dari segi pemrosesan NLP (RoBERTa), lirik tersebut secara dominan menyampaikan sentimen bernada **{label_sentimen}** "
                        f"(Skor: {sentimen_score:.3f}). Di sisi lain, komposisi audionya dievaluasi oleh Spotify memiliki tingkat keceriaan sebesar **{valence_score:.2f}**."
                    )

                    # Kesimpulan Disonansi (Card UI)
                    st.markdown("**Kesimpulan Hubungan Emosi:**")
                    if diss_val >= 0.4:
                        st.error(
                            f"**Disonansi Tinggi (Skor: {diss_val:.3f})**\n\n"
                            f"Terdapat kontras yang signifikan antara nada musik dan makna liriknya. "
                            f"Lagu ini menggunakan melodi yang menipu (berlawanan arah) untuk menyampaikan pesan utamanya."
                        )
                    else:
                        st.success(
                            f"**Konsonansi / Harmonis (Skor: {diss_val:.3f})**\n\n"
                            f"Lagu ini memiliki keselarasan emosional yang baik. "
                            f"Suasana melodi (Audio Valence) berjalan beriringan dan mendukung makna lirik yang disampaikan."
                        )
                else:
                    # ==========================================
                    # SKENARIO INSTRUMENTAL / SKIT / LIRIK TERLALU PENDEK
                    # ==========================================
                    st.markdown(
                        f"Lagu ini terdeteksi sebagai trek **Instrumental**, **Skit**, atau memiliki lirik yang terlalu minim. "
                        f"Karena kurangnya data tekstual, algoritma NLP tidak dapat mengekstraksi tema lirik maupun sentimen sentimennya."
                    )
                    
                    if pd.notna(valence_score):
                        st.markdown(
                            f"Meski demikian, berdasarkan analisis fitur audio murni dari Spotify, komposisi musik ini memiliki tingkat keceriaan (Valence) sebesar **{valence_score:.2f}**."
                        )

                    # Kesimpulan Disonansi 
                    st.markdown("**Kesimpulan Hubungan Emosi:**")
                    st.info(
                        "**Disonansi Tidak Dapat Dihitung**\n\n"
                        "Karena absennya skor sentimen lirik (teks), sistem tidak dapat menghitung kontras (disonansi) antara lirik dan audio. Lagu ini dievaluasi murni berdasarkan nuansa melodinya saja."
                    )
                
    elif menu == "3. Eksplorasi Topik (BERTopic)":
        st.title("Eksplorasi Topik Tema Lirik (BERTopic)")
        st.markdown("Halaman ini memetakan kelompok tema lirik yang ditemukan oleh algoritma *Machine Learning* **BERTopic** pada diskografi Tyler, The Creator.")
        st.divider()

        # ---------------------------------------------------
        # VISUALISASI INTERAKTIF BERDAMPINGAN
        # ---------------------------------------------------
        st.markdown("### Peta Interaktif & Kata Kunci Topik")
        st.markdown(
            "Silakan arahkan kursor (hover) pada gelembung untuk melihat kemiripan antar era, "
            "dan lihat grafik batang di sebelahnya untuk mengetahui kata kunci penyusunnya."
        )
                
        # Membuat dua kolom
        col_peta, col_bar = st.columns(2)
        
        # Kolom Kiri: Peta Jarak Antar Topik
        with col_peta:
            try:
                # Membaca file JSON 
                with open("../Graph/bertopic_map_7.json", "r", encoding="utf-8") as f:
                    fig_map = pio.from_json(f.read())
                
                # Menampilkan secara native di Streamlit 
                st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")
            except FileNotFoundError:
                st.error("⚠️ File 'bertopic_map_7.json' belum ditemukan.")
                
        # Kolom Kanan: Bar Chart Interaktif
        with col_bar:
            try:
                with open("../Graph/bertopic_bar_7.json", "r", encoding="utf-8") as f:
                    fig_bar = pio.from_json(f.read())
                
                # Menampilkan secara native di Streamlit
                st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")
            except FileNotFoundError:
                st.error("⚠️ File 'bertopic_bar_7.json' belum ditemukan.")

        st.divider()

        # ---------------------------------------------------
        # KAMUS TOPIK & EKSPLORASI LAGU
        # ---------------------------------------------------
        st.markdown("### Eksplorasi Mendalam Tiap Tema")
        
        df_nlp = df.dropna(subset=['dominant_topic']).copy()
        df_nlp['dominant_topic'] = df_nlp['dominant_topic'].astype(int)
        
        col_kiri, col_kanan = st.columns([1, 1.5], gap="large")

        with col_kiri:
            st.markdown("#### 📖 Kamus Interpretasi Tema")
            
            
            kamus_topik = {
                -1: "Outliers / Eksperimental (Lirik Acak/Unik)",
                 0: "Autobiografi, Evolusi Persona, dan Braggadocio", 
                 1: "Romansa, Infatuasi, dan Kerentanan Emosional", 
                 2: "Konflik Asmara, Patah Hati, dan Obsesi"
            }
            
            topik_unik = sorted(df_nlp['dominant_topic'].unique())
            for t in topik_unik:
                deskripsi = kamus_topik.get(t, "Interpretasi belum didefinisikan")
                st.markdown(f"**Topik {t}**: {deskripsi}")

        with col_kanan:
            st.markdown("#### Jelajah Lagu per Tema")
            
            def format_nama_topik(t):
                if t == -1:
                    return "Topik -1 (Outliers)"
                return f"Topik {t}"

            selected_topic_id = st.selectbox(
                "Pilih Tema Lirik yang Ingin Dieksplorasi:", 
                options=topik_unik,
                format_func=format_nama_topik
            )
            
            df_topic = df_nlp[df_nlp['dominant_topic'] == selected_topic_id]
            st.success(f"Ditemukan **{len(df_topic)} lagu** pada {format_nama_topik(selected_topic_id)}.")
            
            df_display = df_topic[['album', 'title', 'sentiment_score','valence', 'dissonance']].copy()
            df_display.columns = ['Album', 'Judul Lagu', 'Sentimen', 'Valence', 'Disonansi']
            
            st.dataframe(
                df_display.style.format({
                    'Sentimen': '{:.3f}',
                    'Valence': '{:.3f}',
                    'Disonansi': '{:.3f}'
                }), 
                use_container_width=True,
                hide_index=True,
                height=400 
            )
        st.divider()

        # ---------------------------------------------------
        # TREN TOPIK BERDASARKAN ALBUM (Time-Series)
        # ---------------------------------------------------
        st.markdown("### Evolusi Tema Lirik Sepanjang Era Album")
        st.markdown("Grafik ini menunjukkan pergeseran fokus tema lirik Tyler, The Creator dari album debut hingga karya terbarunya.")
        
        # Mendefinisikan urutan kronologis album
        urutan_album = [
            "Goblin", 
            "Wolf", 
            "Cherry Bomb", 
            "Flower Boy", 
            "IGOR", 
            "CALL ME IF YOU GET LOST", 
            "CHROMAKOPIA",
            "DON'T TAP THE GLASS"
        ]
        
        df_trend_data = df_nlp.copy()
        
        # Mengunci urutan album agar tidak diurutkan sesuai abjad oleh grafik
        df_trend_data['album'] = pd.Categorical(df_trend_data['album'], categories=urutan_album, ordered=True)
        
        # Menghitung jumlah lagu untuk tiap topik di tiap album
        df_trend = df_trend_data.groupby(['album', 'dominant_topic']).size().reset_index(name='jumlah_lagu')
        
        # Memetakan nomor topik menjadi nama interpretasinya
        df_trend['nama_topik'] = df_trend['dominant_topic'].map(kamus_topik)
        
        # Membuat Visualisasi Line Chart Interaktif
        fig_trend = px.line(
            df_trend, 
            x="album", 
            y="jumlah_lagu", 
            color="nama_topik", 
            markers=True, 
            labels={"album": "Era Album", "jumlah_lagu": "Jumlah Lagu", "nama_topik": "Kategori Tema"}
        )
        
        # Merapikan tampilan grafik
        fig_trend.update_layout(
            hovermode="x unified", 
            legend=dict(
                orientation="h", 
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        
        # Menampilkan di Streamlit
        st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")    