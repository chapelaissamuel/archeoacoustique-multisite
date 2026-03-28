import streamlit as st
import numpy as np
from scipy import stats
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="BARABAR Diag", page_icon="🔬")

st.title("🔬 Diagnostic BARABAR")
st.success("✅ Streamlit OK")
st.success("✅ NumPy OK")
st.success("✅ SciPy OK")
st.success("✅ Matplotlib OK")
st.success("✅ Requests OK")

st.markdown("---")
st.markdown("### Test Kuramoto minimal")

if st.button("Test N=500 · T=0.5s"):
    import time
    N = 500
    dt = 0.0005
    T = 0.5
    n_steps = int(T / dt)
    
    t0 = time.time()
    omega = 2 * np.pi * np.random.normal(40, 2.0, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    noise = np.sqrt(2 * (1e-3)**2 * dt) * np.random.randn(n_steps, N)
    history = np.zeros((n_steps, N))
    for i in range(n_steps):
        history[i] = theta
        mean_p = np.angle(np.mean(np.exp(1j * theta)))
        dtheta = omega + 2.0 * np.sin(mean_p - theta) + noise[i] / dt
        theta = np.mod(theta + dtheta * dt, 2*np.pi)
    elapsed = time.time() - t0
    
    plv = float(np.abs(np.mean(np.exp(1j * history[-100:].mean(axis=1)))))
    st.success(f"✅ Kuramoto OK — PLV={plv:.3f} en {elapsed:.1f}s")

st.markdown("---")
st.info("Si tu vois ce message, le serveur fonctionne.")
st.write(f"NumPy version: {np.__version__}")
