# streamlit_app.py - VERSION SIMPLIFIÉE AVEC PLUS DE DÉTAILS
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

st.title("📊 Bank Churn Prediction Dashboard")
st.markdown("Interface de prédiction et monitoring")

# URL de votre API FastAPI
API_URL = "https://bank-churn.salmonforest-247a5473.italynorth.azurecontainerapps.io"

# ============================================
# SECTION 1 : PRÉDICTION UNIQUE AVEC DÉTAILS
# ============================================
with st.expander("🔮 Prédiction Client", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 42)
        tenure = st.slider("Tenure", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 500000.0, 12500.5)
        
    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)
        estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 45000.0)
        geography = st.radio("Geography", ["France", "Spain", "Germany"], index=1)
    
    if st.button("Prédire le Churn", type="primary"):
        features = {
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card else 0,
            "IsActiveMember": 1 if is_active else 0,
            "EstimatedSalary": estimated_salary,
            "Geography_Germany": 1 if geography == "Germany" else 0,
            "Geography_Spain": 1 if geography == "Spain" else 0
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
            if response.status_code == 200:
                result = response.json()
                
                # ===== NOUVEAU : AFFICHAGE DÉTAILLÉ =====
                st.success(f"✅ **Probabilité de churn : {result['churn_probability']:.1%}**")
                
                # Métriques en colonnes
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    if result['prediction'] == 1:
                        st.error("**Prédiction : CHURN**")
                    else:
                        st.success("**Prédiction : NO CHURN**")
                
                with col_res2:
                    risk_color = {
                        "Low": "🟢",
                        "Medium": "🟡", 
                        "High": "🔴"
                    }.get(result['risk_level'], "⚪")
                    st.info(f"**Niveau de risque :** {risk_color} {result['risk_level']}")
                
                with col_res3:
                    if 'cache_info' in result:
                        cache_status = "🟢 HIT" if result['cache_info']['hit'] else "🟡 MISS"
                        st.metric("Cache", cache_status, f"{result['cache_info']['response_time_ms']:.0f} ms")
                
                # ===== NOUVEAU : INTERPRÉTATION =====
                st.markdown("---")
                st.markdown("### 📝 Interprétation")
                
                prob = result['churn_probability']
                
                if prob < 0.3:
                    st.info("""
                    **🟢 CLIENT FAIBLE RISQUE**  
                    *Probabilité < 30%*  
                    
                    **Recommandations :**
                    - Client stable, fidélisation standard
                    - Surveillance trimestrielle suffisante
                    - Offres de produits complémentaires
                    """)
                elif prob < 0.7:
                    st.warning("""
                    **🟡 CLIENT RISQUE MODÉRÉ**  
                    *Probabilité 30-70%*  
                    
                    **Recommandations :**
                    - Contacter pour feedback
                    - Offrir avantages fidélité
                    - Surveillance mensuelle
                    """)
                else:
                    st.error("""
                    **🔴 CLIENT HAUT RISQUE**  
                    *Probabilité > 70%*  
                    
                    **Recommandations URGENTES :**
                    - Contacter sous 48h
                    - Offre de rétention personnalisée
                    - Entretien avec conseiller
                    """)
                
                # ===== NOUVEAU : DÉTAILS TECHNIQUES =====
                with st.expander("🔍 Détails techniques"):
                    col_tech1, col_tech2 = st.columns(2)
                    
                    with col_tech1:
                        st.markdown("**Caractéristiques analysées :**")
                        for key, value in features.items():
                            st.text(f"• {key}: {value}")
                    
                    with col_tech2:
                        st.markdown("**Informations système :**")
                        st.text(f"• Seuil de décision: 50%")
                        st.text(f"• Timestamp: {datetime.now().strftime('%H:%M:%S')}")
                        if 'cache_info' in result:
                            st.text(f"• Cache hash: {result.get('cache_hash', 'N/A')}")
                            
            else:
                st.error(f"Erreur API: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Connexion impossible: {e}")

# ============================================
# SECTION 2 : MONITORING (SIMPLIFIÉ)
# ============================================
with st.expander("📈 Monitoring & Cache"):
    
    if st.button("📊 Voir les statistiques du cache"):
        try:
            stats = requests.get(f"{API_URL}/cache/stats").json()
            
            if "stats" in stats:
                s = stats["stats"]
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("Taux de succès", f"{s.get('hit_ratio', 0):.1%}")
                
                with col_stat2:
                    st.metric("Hits / Misses", f"{s.get('hits', 0)} / {s.get('misses', 0)}")
                
                with col_stat3:
                    st.metric("Taille cache", f"{s.get('cache_size', 0)} / {s.get('cache_maxsize', 2000)}")
                
                # Résumé
                hit_ratio = s.get('hit_ratio', 0)
                if hit_ratio > 0.7:
                    st.success(f"✅ Cache très efficace ({hit_ratio:.1%} de hits)")
                elif hit_ratio > 0.3:
                    st.warning(f"⚠️ Cache moyennement efficace ({hit_ratio:.1%} de hits)")
                else:
                    st.info(f"📊 Cache peu utilisé ({hit_ratio:.1%} de hits)")
                    
            else:
                st.json(stats)
                
        except Exception as e:
            st.error(f"Impossible de récupérer les stats: {e}")
    
    if st.button("🔄 Vider le Cache"):
        try:
            result = requests.post(f"{API_URL}/cache/clear").json()
            if result.get("status") == "success":
                st.success("✅ Cache vidé avec succès")
            else:
                st.warning("Cache déjà vide")
        except Exception as e:
            st.error(f"Erreur: {e}")

# ============================================
# SECTION 3 : TRAITEMENT MULTIPLE (GARDÉ)
# ============================================
with st.expander("📁 Traitement par lot"):
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.markdown(f"**Fichier chargé :** {len(df)} lignes")
        st.dataframe(df.head())
        
        if st.button("Prédire le lot", type="secondary"):
            # Conversion au format API
            records = df.to_dict('records')
            
            try:
                with st.spinner(f"Traitement de {len(records)} clients..."):
                    response = requests.post(f"{API_URL}/predict/batch", json=records)
                    results = response.json()
                    
                    if "predictions" in results:
                        # Résumé
                        st.success(f"✅ {results['count']} prédictions effectuées")
                        
                        # Statistiques
                        col_batch1, col_batch2, col_batch3 = st.columns(3)
                        
                        with col_batch1:
                            hit_ratio = results['cache_stats']['hit_ratio']
                            st.metric("Efficacité cache", hit_ratio)
                        
                        with col_batch2:
                            # Compter les churns
                            churn_count = sum(1 for p in results['predictions'] if p.get('prediction') == 1)
                            st.metric("Clients à risque", f"{churn_count}/{results['count']}")
                        
                        with col_batch3:
                            total_time = results['cache_stats'].get('total_time_ms', 0)
                            st.metric("Temps total", f"{total_time:.0f} ms")
                        
                        # Télécharger les résultats
                        results_df = pd.DataFrame(results['predictions'])
                        csv = results_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Télécharger les résultats",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                        
                        # Aperçu des résultats
                        with st.expander("Aperçu des prédictions"):
                            st.dataframe(results_df.head(10))
                            
                    else:
                        st.error("Format de réponse inattendu")
                        st.json(results)
                        
            except Exception as e:
                st.error(f"Erreur: {e}")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## 📋 Informations")
st.sidebar.info(f"**API URL:**\n`{API_URL}`")

# Vérification santé
if st.sidebar.button("🏥 Vérifier santé API"):
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.sidebar.success(f"✅ API en ligne - {health.get('status', 'unknown')}")
    except:
        st.sidebar.error("🔌 API hors ligne")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Seuil de décision :** 50%

**Niveaux de risque :**
- 🟢 **Low** : < 30%
- 🟡 **Medium** : 30-70%  
- 🔴 **High** : > 70%

**Cache :**
- 🟢 **HIT** : Réponse depuis cache
- 🟡 **MISS** : Nouveau calcul
""")

st.sidebar.caption(f"Streamlit Dashboard v1.1 • {datetime.now().strftime('%d/%m/%Y')}")