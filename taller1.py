import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de Streamlit
st.set_page_config(page_title="Asesoramiento en acciones", layout="wide")

# Título principal de la aplicación
st.title("Asesoramiento en Acciones")

# Menú de navegación con pestañas
menu = st.sidebar.radio(
    "Menú",
    ["Inicio", "Acciones Sector Financiero"]
)

# Pestaña de inicio
if menu == "Inicio":
    st.write("Bienvenido al asesoramiento en acciones. Usa el menú lateral para navegar entre las opciones.")

# Pestaña de acciones del sector financiero
if menu == "Acciones Sector Financiero":
    st.header("Análisis de Acciones del Sector Financiero")

    # Descargar los datos de las acciones JPM y BAC
    tickers = ['JPM', 'BAC']
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

    # Renombrar las columnas para asegurarnos que tienen el formato correcto
    data.columns = ['JP MORGAN CHASE', 'BANK OF AMERICA']

    # Mostrar las primeras filas del DataFrame
    st.subheader("Datos de Precios Ajustados")
    st.write(data.head())

    # Gráfico de precios históricos
    st.subheader("Precios Históricos de JPM y BAC")
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(ax=ax, title='Precios históricos de JPM y BAC', grid=True, color=['lime', 'teal'])
    ax.set_ylabel('Precio (USD)')
    st.pyplot(fig)

    # Calcular rendimientos diarios
    returns = data.pct_change().dropna()

    # Gráfico de rendimientos diarios
    st.subheader("Rendimientos Diarios de JPM y BAC")
    fig, ax = plt.subplots(figsize=(10, 6))
    returns.plot(ax=ax, title='Rendimientos diarios de JPM y BAC', grid=True, color=['lime', 'teal'])
    ax.set_ylabel('Rendimientos Diarios')
    st.pyplot(fig)

    # Calcular la matriz de correlación
    st.subheader("Matriz de Correlación de Rendimientos")
    correlation_matrix = returns.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='PuBuGn', fmt=".2f", square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Matriz de Correlación de Rendimientos Diarios (BAC y JPM)', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)

    # Simulaciones de Movimiento Geométrico Browniano para predicciones
    st.subheader("Simulaciones de Precios Futuros (Próximos 90 días)")

    T = 90  # Número de días a predecir
    num_simulations = 10  # Número de simulaciones
    np.random.seed(42)

    for ticker, name in zip(tickers, data.columns):
        # Calcular los rendimientos diarios
        returns = data[name].pct_change().dropna()

        # Parámetros del Movimiento Geométrico Browniano
        mu = returns.mean()  # Retorno promedio
        sigma = returns.std()  # Volatilidad
        S0 = data[name].iloc[-1]  # Precio actual (último precio disponible)

        all_predictions = []

        for _ in range(num_simulations):
            prices = [S0]
            for t in range(T):
                epsilon = np.random.normal(0, 1)
                St_next = prices[-1] * np.exp((mu * 1) + (sigma * epsilon * np.sqrt(1)))
                prices.append(St_next)
            all_predictions.append(prices)

        # Convertir las simulaciones a DataFrame
        pred_df = pd.DataFrame(all_predictions).T

        # Gráfico de las simulaciones de precios
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("tab10", num_simulations)

        for i in range(num_simulations):
            ax.plot(pred_df[i], color=colors[i % len(colors)], alpha=0.8)

        ax.set_title(f'Simulaciones de precios para {name} (próximos 90 días)', fontsize=16)
        ax.set_xlabel('Días')
        ax.set_ylabel('Precio (USD)')
        ax.grid(True)
        st.pyplot(fig)
