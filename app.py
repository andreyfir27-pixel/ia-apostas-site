     import streamlit as st
     import requests
     import pandas as pd
     from datetime import datetime
     import joblib
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import accuracy_score
     import numpy as np

     # Configurações visuais do site
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

     # Sidebar para configurações
     st.sidebar.title("🎯 Configurações da IA")
     st.sidebar.markdown("Insira suas chaves para ativar a coleta real de dados.")
     THE_ODDS_API_KEY = st.sidebar.text_input("Chave The Odds API", type="password", help="Pegue em theoddsapi.com")
     RAPIDAPI_KEY = st.sidebar.text_input("Chave RapidAPI (Lesionados)", type="password", help="Pegue em rapidapi.com")
     threshold = st.sidebar.slider("Threshold de EV para Dicas", 0.01, 0.2, 0.05, help="Quanto maior, menos dicas (mais seguras)")

     # Título principal
     st.title("🤖 IA Inteligente de Dicas de Apostas")
     st.markdown("""
         <div class="card">
             <h3>Bem-vindo à IA de Apostas!</h3>
             <p>Esta ferramenta usa Machine Learning para analisar odds de apostas e dados de lesionados em NBA e Brasileirão. Gera dicas com EV (Expected Value) e stake sugerido via Kelly Criterion. Use para análise educacional — não aposte dinheiro real sem pesquisa.</p>
             <p><strong>Como usar:</strong> Insira chaves na sidebar, treine o modelo e clique em 'Atualizar Dicas'.</p>
         </div>
     """, unsafe_allow_html=True)

     # Função para treinar modelo
     def train_ml_model():
         with st.spinner("Treinando modelo de IA..."):
             data = []
             for _ in range(100):
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
             
             model = RandomForestClassifier(n_estimators=100)
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

     # Funções para APIs
     def get_odds(sport_key):
         if not THE_ODDS_API_KEY:
             st.error("Insira a chave da The Odds API na sidebar!")
             return []
         url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={THE_ODDS_API_KEY}&regions=us&markets=h2h'
         response = requests.get(url)
         return response.json() if response.status_code == 200 else []

     def get_injuries(sport):
         if not RAPIDAPI_KEY:
             return 0.0
         if sport == 'basketball_nba':
             url = 'https://api.sportsdata.io/v2/json/Players'
             headers = {'Ocp-Apim-Subscription-Key': 'SUA_CHAVE_SPORTSDATA'}
         else:
             url = 'https://api-football-v1.p.rapidapi.com/v3/injuries'
             headers = {'X-RapidAPI-Key': RAPIDAPI_KEY}
             params = {'league': '71', 'season': '2023'}
         response = requests.get(url, headers=headers, params=params if 'params' in locals() else {})
         if response.status_code == 200:
             data = response.json()
             injured_count = len([p for p in data if p.get('injured')])
             return injured_count / 10
         return 0.0

     # Função para gerar dicas
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

     # Botões e lógica
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
                 sports = ['basketball_nba', 'soccer_brazil_campeonato']
                 all_tips = []
                 
                 progress_bar = st.progress(0)
                 for i, sport in enumerate(sports):
                     st.subheader(f"📊 Analisando {sport.replace('_', ' ').title()}...")
                     odds_data = get_odds(sport)
                     injury_factor = get_injuries(sport)
                     st.write(f"🔍 Fator de lesionados estimado: {injury_factor:.2f}")
                     
                     for game in odds_data[:5]:
                         home_team = game['home_team']
                         away_team = game['away_team']
                         odds_home = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
                         odds_away = game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
                         
                         tip, ev, stake = generate_tip(home_team, away_team, odds_home, odds_away, injury_factor, model, threshold)
                         if ev > 0:
                             all_tips.append({
                                 'Esporte': sport.replace('_', ' ').title(),
                                 'Jogo': f"{home_team} vs {away_team}",
                                 'Odds Casa': odds_home,
                                 'Odds Fora': odds_away,
                                 'Fator Lesionados': f"{injury_factor:.2f}",
                                 'Dica': tip,
                                 'EV': f"{ev:.2f}",
                                 'Stake Sugerido': f"{stake:.2%}",
                                 'Horário': datetime.now().strftime("%d/%m/%Y %H:%M")
                             })
                     progress_bar.progress((i+1)/len(sports))
                 
                 if all_tips:
                     df = pd.DataFrame(all_tips)
                     st.dataframe(df.style.apply(lambda x: ['background-color: lightgreen' if 'Aposte' in str(val) else '' for val in x], axis=1))
                     csv = df.to_csv(index=False)
                     st.download_button("📥 Baixar Dicas em CSV", csv, "dicas_apostas.csv", "text/csv")
                     st.success("✅ Dicas geradas com sucesso!")
                 else:
                     st.warning("⚠️ Nenhuma dica forte encontrada com o threshold atual. Ajuste na sidebar.")

     # Rodapé
     st.markdown("---")
     st.markdown("""
         <div class="card">
             <p><strong>Nota:</strong> Dados são aproximados e para fins educacionais. A IA usa ML treinado com dados simulados — melhore com históricos reais para mais precisão. Não promovo apostas; consulte profissionais.</p>
         </div>
     """, unsafe_allow_html=True)
     
