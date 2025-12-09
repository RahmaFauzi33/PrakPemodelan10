import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================
# LOAD DATASET
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("used_cars_timeseries.csv")
    df = df.sort_values(by="age").reset_index(drop=True)
    return df

df = load_data()

st.title("📉 TA-10 | Simulasi RK4 Model Logistik pada Depresiasi Harga Mobil")
st.write("Dataset: **used_cars_timeseries.csv** (age vs price)")

# ============================================
# EXTRACT TIME-SERIES
# ============================================

t_data = df["age"].astype(float).values
P_data = df["price"].astype(float).values

t0 = t_data[0]
P0 = P_data[0]
n_steps = len(t_data) - 1
h = 1.0  # step size per year

# ============================================
# MODEL LOGISTIK + RK4
# ============================================

def logistic_rhs(t, P, r, K):
    return r * P * (1 - P / K)

def rk4_logistic(t0, P0, h, n_steps, r, K):
    t = t0
    P = P0
    t_values = [t]
    P_values = [P]

    for _ in range(n_steps):
        k1 = logistic_rhs(t, P, r, K)
        k2 = logistic_rhs(t + 0.5*h, P + 0.5*h*k1, r, K)
        k3 = logistic_rhs(t + 0.5*h, P + 0.5*h*k2, r, K)
        k4 = logistic_rhs(t + h,     P + h*k3,     r, K)

        P = P + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_values.append(t)
        P_values.append(P)

    return np.array(t_values), np.array(P_values)

def rmse(a, b):
    return math.sqrt(np.mean((a - b) ** 2))


# ============================================
# SIDEBAR INPUT
# ============================================

st.sidebar.header("Parameter Model Logistik")

# r < 0 karena harga mengalami depresiasi
r = st.sidebar.slider("Parameter r (negatif → depresiasi)", -1.0, 0.0, -0.3, 0.01)

# K kapasitas (harga awal atau batas atas)
K_guess = float(P_data.max())
K = st.sidebar.slider("Parameter K (kapasitas harga)", 0.5*K_guess, 2.0*K_guess, K_guess)

# ============================================
# SIMULASI RK4
# ============================================

error, = [None]

t_sim, P_sim = rk4_logistic(t0, P0, h, n_steps, r, K)

m = min(len(P_sim), len(P_data))
P_sim = P_sim[:m]
P_ref = P_data[:m]

error = rmse(P_sim, P_ref)

# ============================================
# PLOT
# ============================================

st.write("### Grafik Data vs Simulasi RK4 (Model Logistik)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(t_data, P_data, label="Data Harga Mobil", s=25)
ax.plot(t_sim, P_sim, label=f"Simulasi RK4 (r={r:.3f}, K={K:.1f})", linewidth=2)

ax.set_xlabel("Usia Mobil (tahun)")
ax.set_ylabel("Harga (USD atau satuan dataset)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ============================================
# ERROR (RMSE)
# ============================================

st.write("### RMSE (Root Mean Squared Error)")
st.metric(label="RMSE", value=f"{error:,.2f}")

st.caption("Metode Numerik: Runge-Kutta Orde 4 | Model ODE: Logistic Growth | TA-10 Streamlit App")
