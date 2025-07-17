import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Calcular Information Value
@st.cache_data
def calculate_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(df[feature]):
        df[feature] = pd.qcut(df[feature], q=bins, duplicates='drop')

    grouped = df.groupby(feature)
    total_good = (df[target] == 0).sum()
    total_bad = (df[target] == 1).sum()

    iv = 0
    woe_dict = {}
    for val, group in grouped:
        good = (group[target] == 0).sum()
        bad = (group[target] == 1).sum()
        dist_good = good / total_good if total_good > 0 else 0.001
        dist_bad = bad / total_bad if total_bad > 0 else 0.001
        woe = np.log(dist_good / dist_bad) if dist_bad > 0 else 0
        iv += (dist_good - dist_bad) * woe
        woe_dict[val] = woe
    return iv, woe_dict

# Tabela de ganho
@st.cache_data
def tabela_ganho(df, score_col, target_col, bins=10):
    df = df[[score_col, target_col]].copy()
    df = df.dropna()
    df['bucket'] = pd.qcut(df[score_col], q=bins, duplicates='drop')
    grouped = df.groupby('bucket')[target_col].agg(['count', 'sum'])
    grouped['bad_rate'] = grouped['sum'] / grouped['count']
    grouped = grouped.sort_index(ascending=False).reset_index()
    grouped['acum_bad'] = grouped['sum'].cumsum()
    grouped['acum_total'] = grouped['count'].cumsum()
    grouped['acum_perc_bad'] = grouped['acum_bad'] / grouped['sum'].sum()
    return grouped

# Separar base treino e OOT
@st.cache_data
def split_oot(df):
    ultimos_meses = df['data_ref'].sort_values().unique()[-3:]
    df_oot = df[df['data_ref'].isin(ultimos_meses)].copy()
    df_dev = df[~df['data_ref'].isin(ultimos_meses)].copy()
    return df_dev, df_oot

# --- Início do app ---

st.title("Exercício 2 - Credit Scoring Analysis")

# Upload do arquivo .ftr
uploaded_file = st.file_uploader("Faça upload do arquivo credit_scoring.ftr", type=["ftr"])

if uploaded_file is not None:
    # Carregar dataframe a partir do arquivo enviado
    df = pd.read_feather(uploaded_file)
    df['data_ref'] = pd.to_datetime(df['data_ref'])
    df = df.drop(columns=['index'], errors='ignore')

    # Filtro global de data
    st.sidebar.title("Navegação")
    secao = st.sidebar.radio("Ir para seção:", ["Análise Univariada", "Análise Bivariada", "Modelagem e Avaliação", "Relatório Final"])

    st.sidebar.subheader("Filtros Globais")
    data_min = st.sidebar.date_input("Data inicial", value=df['data_ref'].min().date())
    data_max = st.sidebar.date_input("Data final", value=df['data_ref'].max().date())

    # Filtrar df pelo intervalo selecionado
    df = df[(df['data_ref'] >= pd.to_datetime(data_min)) & (df['data_ref'] <= pd.to_datetime(data_max))]
    df_dev, df_oot = split_oot(df)
    target = "mau"

    st.write(f"Amostra total: {len(df)} registros | Desenvolvimento: {len(df_dev)} | OOT: {len(df_oot)}")

    # Resto do app igual ao seu código, usando as variáveis df, df_dev, df_oot, target
    if secao == "Análise Univariada":
        st.header("1. Análise Univariada")
        variavel = st.sidebar.selectbox("Escolha uma variável para análise univariada:", df.columns.drop([target, 'data_ref']).tolist())

        st.subheader("Resumo Estatístico Simples")
        st.write(df[variavel].describe())

        st.subheader("Histograma com KDE")
        fig, ax = plt.subplots()
        sns.histplot(df[variavel].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Evolução Temporal (Longitudinal)")
        if pd.api.types.is_numeric_dtype(df[variavel]):
            agg_df = df.groupby('data_ref')[variavel].agg(['mean', 'median', 'std'])
            st.line_chart(agg_df)
        else:
            st.warning("A variável selecionada não é numérica. Selecione outra para visualizar a evolução temporal.")

    elif secao == "Análise Bivariada":
        st.header("2. Análise Bivariada")
        variavel_bi = st.sidebar.selectbox("Escolha a variável para análise bivariada:", df.columns.drop([target, 'data_ref']).tolist())

        st.subheader("Boxplot por Target")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=target, y=variavel_bi, data=df, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Perfil Longitudinal por Target")
        if pd.api.types.is_numeric_dtype(df[variavel_bi]):
            perfil = df.groupby(['data_ref', target])[variavel_bi].mean().unstack()
            st.line_chart(perfil)
        else:
            st.warning("A variável selecionada não é numérica. Selecione outra para visualizar o perfil longitudinal.")

        st.subheader("Information Value (IV)")
        iv, woe_dict = calculate_iv(df, variavel_bi, target)
        st.write(f"IV de {variavel_bi}: {iv:.4f}")

    elif secao == "Modelagem e Avaliação":
        st.header("3. Modelagem e Avaliação")
        features_possiveis = df_dev.columns.drop([target, 'data_ref']).tolist()
        default_features = features_possiveis[:5]

        features_modelo = st.sidebar.multiselect(
            "Selecione variáveis para o modelo:",
            options=features_possiveis,
            default=default_features
        )

        if len(features_modelo) > 0:
            X_dev = df_dev[features_modelo].fillna(0)
            y_dev = df_dev[target]
            X_oot = df_oot[features_modelo].fillna(0)
            y_oot = df_oot[target]

            scaler = StandardScaler()
            X_dev_scaled = scaler.fit_transform(X_dev)
            X_oot_scaled = scaler.transform(X_oot)

            # Modelo com class_weight balanceado
            model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
            model.fit(X_dev_scaled, y_dev)

            prob_dev = model.predict_proba(X_dev_scaled)[:, 1]
            prob_oot = model.predict_proba(X_oot_scaled)[:, 1]

            preds_dev = model.predict(X_dev_scaled)
            preds_oot = model.predict(X_oot_scaled)

            st.subheader("Balanceamento das Classes")
            st.write("Distribuição target DEV:", y_dev.value_counts(normalize=True))
            st.write("Distribuição target OOT:", y_oot.value_counts(normalize=True))

            st.subheader("Distribuição das Predições")
            st.write("Predições DEV:", np.unique(preds_dev, return_counts=True))
            st.write("Predições OOT:", np.unique(preds_oot, return_counts=True))

            st.subheader("Probabilidades previstas")
            st.write(f"DEV - min: {prob_dev.min():.4f}, max: {prob_dev.max():.4f}")
            st.write(f"OOT - min: {prob_oot.min():.4f}, max: {prob_oot.max():.4f}")

            # Métricas padrão (threshold 0.5)
            acc_dev = accuracy_score(y_dev, preds_dev)
            acc_oot = accuracy_score(y_oot, preds_oot)
            gini_dev = 2 * roc_auc_score(y_dev, prob_dev) - 1
            gini_oot = 2 * roc_auc_score(y_oot, prob_oot) - 1

            st.write(f"Acurácia DEV: {acc_dev:.3f} | Acurácia OOT: {acc_oot:.3f}")
            st.write(f"Gini DEV: {gini_dev:.3f} | Gini OOT: {gini_oot:.3f}")

            # Curva ROC
            fpr, tpr, _ = roc_curve(y_oot, prob_oot)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label='ROC OOT')
            ax_roc.plot([0, 1], [0, 1], '--', color='gray')
            ax_roc.set_title('Curva ROC - OOT')
            ax_roc.set_xlabel('FPR')
            ax_roc.set_ylabel('TPR')
            st.pyplot(fig_roc)

            # Matriz de confusão padrão (threshold 0.5)
            st.subheader("Matriz de Confusão - OOT (threshold=0.5)")
            cm = confusion_matrix(y_oot, preds_oot)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predito')
            ax_cm.set_ylabel('Real')
            ax_cm.set_title('Matriz de Confusão - OOT')
            st.pyplot(fig_cm)

            # Threshold customizado para sensibilidade maior
            threshold = st.slider("Ajuste do threshold para predição da classe 1", 0.0, 1.0, 0.3, 0.01)
            preds_oot_thresh = (prob_oot >= threshold).astype(int)

            st.subheader(f"Matriz de Confusão - OOT (threshold={threshold:.2f})")
            cm_thresh = confusion_matrix(y_oot, preds_oot_thresh)
            fig_cm_thresh, ax_cm_thresh = plt.subplots()
            sns.heatmap(cm_thresh, annot=True, fmt='d', cmap='Blues', ax=ax_cm_thresh)
            ax_cm_thresh.set_xlabel('Predito (threshold customizado)')
            ax_cm_thresh.set_ylabel('Real')
            ax_cm_thresh.set_title(f'Matriz de Confusão - OOT (threshold={threshold:.2f})')
            st.pyplot(fig_cm_thresh)

            # Atualiza df com score e salva no session_state para relatório final
            st.session_state['score_df'] = df_oot.copy()
            st.session_state['score_df']['score'] = prob_oot
        else:
            st.warning("Selecione ao menos uma variável para rodar o modelo.")


    elif secao == "Relatório Final":
        st.header("4. Relatório Final e Tabela de Ganhos")
        if 'score_df' in st.session_state:
            score_df = st.session_state['score_df']
            tabela = tabela_ganho(score_df, 'score', target)
            st.write(tabela)
            st.line_chart(tabela[['acum_perc_bad']])

            buffer = BytesIO()
            tabela.to_excel(buffer, index=False)
            st.download_button("📥 Baixar Tabela de Ganhos em Excel", data=buffer.getvalue(), file_name="tabela_ganhos.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Execute o modelo na aba anterior para gerar o score e a tabela de ganhos.")
else:
    st.info("Por favor, faça o upload do arquivo credit_scoring.ftr para começar.")


