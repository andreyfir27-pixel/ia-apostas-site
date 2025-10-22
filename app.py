import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# =========================
# 🎨 CONFIGURAÇÃO VISUAL
# =========================
st.set_page_config(page_title="IA Apostas Inteligente", page_icon="🎲", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 10px; font-size: 16px; }
    .stButton>button:hover { background-color: #ff6b6b; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .dica { color: #28a745; font-weight: bold; }
    .nenhuma { color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# =========================
# ⚙️ SIDEBAR CONFIGURAÇÕES
# =========================
st.sidebar.title("🎯 Configurações da IA")
st.sidebar.markdown("Insira suas chaves para ativar a coleta real de dados.")
THE_ODDS_API_KEY = st.sidebar.text_input("Chave The Odds API", type="password", help="Pegue em theoddsapi.com")
RAPIDAPI_KEY = st.sidebar.text_input("Chave RapidAPI (Lesionados)", type="password", help="Pegue em rapidapi.com")
threshold = st.sidebar.slider("Threshold de EV para Dicas", 0.01, 0.2, 0.05, help="Quanto maior, menos dicas (mais seguras)")

# Escolha de ligas
ligas_disponiveis = {
    "🏀 NBA": "basketball_nba",
    "⚽ Brasileirão Série A": "soccer_brazil_campeonato",
    "⚽ Premier League": "soccer_epl",
    "⚽ La Liga": "soccer_spain_la_liga"
}
liga_escolhida = st.sidebar.selectbox("Escolha o Campeonato", list(ligas_disponiveis.keys()))
sport_key = ligas_disponiveis[liga_escolhida]

# =========================
# 🧠 TÍTULO PRINCIPAL
# =========================
st.title("🤖 IA Inteligente de Dicas de Apostas")
st.markdown("""
    <div class="card">
        <h3>Bem-vindo à IA de Apostas!</h3>
        <p>Esta ferramenta usa Machine Learning para analisar odds de apostas e dados de lesionados. Gera dicas com EV (Expected Value) e stake sugerido via Kelly Criterion.</p>
        <p><strong>Como usar:</strong> Insira suas chaves na sidebar, treine o modelo e clique em 'Atualizar Dicas'.</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# 🤖 FUNÇÕES DE IA
# =========================
def train_ml_model():
    with st.spinner("Treinando modelo de IA..."):
        data = []
        for _ in range(200):
            odds_home = np.random.uniform(1.5, 3.0)
            odds_away = np.random.uniform(1.5, 3.0)
            injury_factor = np.random.uniform(0, 1)
            prob_home = 0.5 - (injury_factor * 0.2)
            result = 1 if np.random.rand() < prob_home else 0
            data.append([odds_home, odds_away, injury_factor, result])
        
        df = pd.DataFrame(data, columns=['odds_home', 'odds_away', 'injury_factor', 'result'])
        X = df[['odds_home', 'odds_away', 'injury_factor']]
        y = df['result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, 'betting_model.pkl')
        st.success(f"Modelo treinado! Acurácia: {acc:.2f}")
        return model

def load_model():
    try:
        return joblib.load('betting_model.pkl')
    except:
        return train_ml_model()

# =========================
# 📡 FUNÇÕES DE API
# =========================
def get_odds(sport_key):
    if not THE_ODDS_API_KEY:
        st.error("Insira a chave da The Odds API na sidebar!")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={THE_ODDS_API_KEY}&regions=br&markets=h2h"
    response = requests.get(url)
    if response.status_code != 200:
        return []

    data = response.json()
    hoje = datetime.utcnow().date()
    jogos_hoje = []
    for game in data:
        game_time = datetime.fromisoformat(game['commence_time'].replace("Z", "+00:00")).date()
        if game_time == hoje:
            jogos_hoje.append(game)
    return jogos_hoje

def get_injuries(league_name):
    if not RAPIDAPI_KEY:
        return [], 0.0

    url = "https://api-football-v1.p.rapidapi.com/v3/injuries"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY}

    ligas_map = {
        "soccer_brazil_campeonato": {"league": "71", "season": "2025"},
        "soccer_epl": {"league": "39", "season": "2025"},
        "soccer_spain_la_liga": {"league": "140", "season": "2025"}
    }

    if league_name not in ligas_map:
        return [], 0.0

    params = ligas_map[league_name]
    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        return [], 0.0

    data = response.json().get("response", [])
    lesionados = []
    for item in data:
        jogador = item["player"]["name"]
        time = item["team"]["name"]
        tipo = item["player"].get("type", "Desconhecido")
        lesionados.append({"Jogador": jogador, "Time": time, "Lesão": tipo})

    fator = min(len(lesionados) / 100, 1.0)
    return lesionados, fator

# =========================
# 💡 FUNÇÃO DE DICAS
# =========================
def generate_tip(home_team, away_team, odds_home, odds_away, injury_factor, model, threshold):
    features = [[odds_home, odds_away, injury_factor]]
    prob_home_est = model.predict_proba(features)[0][1]
    ev_home = (prob_home_est * (odds_home - 1)) - ((1 - prob_home_est) * 1)

    if ev_home > threshold:
        stake = min(0.1, ev_home * 2)
        return f"Aposte em {home_team}", ev_home, stake

    prob_away_est = 1 - prob_home_est
    ev_away = (prob_away_est * (odds_away - 1)) - ((1 - prob_away_est) * 1)

    if ev_away > threshold:
        stake = min(0.1, ev_away * 2)
        return f"Aposte em {away_team}", ev_away, stake

    return "Nenhuma dica forte", 0, 0

# =========================
# 🧩 INTERFACE PRINCIPAL
# =========================
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Treinar Modelo de IA"):
        model = train_ml_model()

with col2:
    if st.button("🔄 Atualizar Dicas de Apostas"):
        if not THE_ODDS_API_KEY:
            st.error("Configure as chaves na sidebar!")
        else:
            model = load_model()
            st.subheader(f"📊 Analisando {liga_escolhida}...")
            odds_data = get_odds(sport_key)
            lesionados, injury_factor = get_injuries(sport_key)
            st.write(f"🔍 Fator de lesionados estimado: {injury_factor:.2f}")

            if len(lesionados) > 0:
                with st.expander("📋 Jogadores Lesionados"):
                    st.dataframe(pd.DataFrame(lesionados))

            all_tips = []
            progress_bar = st.progress(0)

            for i, game in enumerate(odds_data[:10]):
                home_team = game['home_team']
                away_team = game['away_team']
                odds_home = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
                odds_away = game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']

                tip, ev, stake = generate_tip(home_team, away_team, odds_home, odds_away, injury_factor, model, threshold)
                if ev > 0:
                    all_tips.append({
                        'Campeonato': liga_escolhida,
                        'Jogo': f"{home_team} vs {away_team}",
                        'Odds Casa': odds_home,
                        'Odds Fora': odds_away,
                        'Fator Lesionados': f"{injury_factor:.2f}",
                        'Dica': tip,
                        'EV': f"{ev:.2f}",
                        'Stake Sugerido': f"{stake:.2%}",
                        'Horário': datetime.now().strftime("%d/%m/%Y %H:%M")
                    })
                progress_bar.progress((i+1)/len(odds_data))

            if all_tips:
                df = pd.DataFrame(all_tips)
                st.dataframe(df.style.apply(lambda x: ['background-color: lightgreen' if 'Aposte' in str(val) else '' for val in x], axis=1))
                csv = df.to_csv(index=False)
                st.download_button("📥 Baixar Dicas em CSV", csv, "dicas_apostas.csv", "text/csv")
                st.success("✅ Dicas geradas com sucesso!")

                st.markdown("### 📈 Estatísticas do Dia")
                media_ev = np.mean([float(t['EV']) for t in all_tips])
                melhor_ev = max([float(t['EV']) for t in all_tips])
                st.metric("Média de EV das Dicas", f"{media_ev:.2f}")
                st.metric("Melhor EV Encontrado", f"{melhor_ev:.2f}")
            else:
                st.warning("⚠️ Nenhuma dica forte encontrada com o threshold atual. Ajuste na sidebar.")

# =========================
# 📘 RODAPÉ
# =========================
st.markdown("---")
st.markdown("""
    <div class="card">
        <p><strong>Nota:</strong> Este aplicativo é para fins educacionais e de pesquisa. Os dados vêm de APIs públicas (TheOddsAPI e API-Football). Sempre verifique as fontes antes de apostar.</p>
    </div>
""", unsafe_allow_html=True)
