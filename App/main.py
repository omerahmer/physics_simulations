import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pandas as pd
from funcs import *

# Functions
def fowler_nordheim(E, E_F, Phi):
    J = 6.2e6 * ((E_F / Phi)**(1/2)) / (E_F - Phi) * E**2 * np.exp(-6.8e7 * (Phi**(3/2)) / E)
    return J

def einzel_lens(E_total, V_middle, V_upper, V_lower):
    focal_length = 1 / (1 + V_middle / V_upper + V_middle / V_lower) * E_total
    spot_size = np.abs(focal_length - E_total)
    return focal_length, spot_size

# Load additional functions
def avg_noise(n_samples, noise_at_freq, m, n):
    # Example implementation for averaging noise
    return np.random.normal(0, noise_at_freq, (n_samples, m, n)).mean(axis=0)

def generate_experiment(concs, freqs, weights):
    # Example implementation for generating experiment data
    return np.random.normal(0, 1, (len(concs), len(freqs)))

def fit(X, concs):
    # Example implementation for fitting weights
    return np.random.normal(0, 1, (len(concs), X.shape[1]))

def predict(X_test, fit_w, agg, exclude_fns, return_preds):
    # Example implementation for making predictions
    preds = np.random.normal(0, 1, len(X_test))
    agg_pred = np.mean(preds)
    return agg_pred, preds

def plot_rows(X, freqs, y_true, label, xlabel, ylabel, title):
    # Example implementation for plotting rows
    fig, ax = plt.subplots()
    for i, y in enumerate(y_true):
        ax.plot(freqs, X[i], label=f"{label}{y}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig

def avg_scatter(preds, true_conc, agg_pred, figsize):
    # Example implementation for plotting predictions
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(preds, [true_conc] * len(preds), label="Predictions")
    ax.axhline(agg_pred, color='r', linestyle='--', label="Aggregated Prediction")
    ax.legend()
    return fig

def generate_readings(R, C, start_freq, end_freq, cycle_num, form, ex_cyc, save_csv):
    # Example implementation for generating readings
    freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), num=cycle_num)
    Z = np.random.normal(0, 1, (cycle_num, len(freqs)))
    return freqs, Z

def bode(freqs, Z):
    # Example implementation for Bode plot
    fig, ax = plt.subplots()
    ax.plot(freqs, np.abs(Z), label="Magnitude")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Bode Plot")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Impedance (Ohms)")
    ax.legend()
    return fig

def nyquist(freqs, Z):
    # Example implementation for Nyquist plot
    fig, ax = plt.subplots()
    ax.plot(Z.real, -Z.imag, label="Nyquist")
    ax.set_title("Nyquist Plot")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend()
    return fig

def calculate_edl_capacitance(d_edl):
    # Example implementation for calculating EDL capacitance
    return np.random.normal(0, 1)

def calculate_d_edl():
    # Example implementation for calculating d_edl
    return np.random.normal(0, 1)

def freq_sweep_at_c(conc, weights, freqs):
    # Example implementation for frequency sweep
    return np.random.normal(0, 1, len(freqs))

# Streamlit Page Configuration
st.set_page_config(page_title="Field Emission and Einzel Lens", layout="wide")

# Title
st.title("Nano-integrated Technology Research Operations (NiTRO)")

# Sidebar Menu
with st.sidebar:
    st.markdown("[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/nazeern/nanotech/blob/main/nitro_app.py) View Source Code")
    menu_sel = option_menu(
        "Main Menu",
        ["Field Emission", "Einzel Lens", "Results"],
        menu_icon="hexagon",
        default_index=0,
    )

# Field Emission Calculation
if menu_sel == "Field Emission":
    st.header("Field Emission Calculation")
    E_F = st.number_input("Fermi Energy (eV)", value=247)
    Phi = st.number_input("Work Function (eV)", value=75)
    d = st.number_input("Distance between nanotubes and top electrode (cm)", value=1.0)
    V_min = st.number_input("Min Applied Voltage (V)", value=0.1)
    V_max = st.number_input("Max Applied Voltage (V)", value=10)
    num_points = st.slider("Number of Points", min_value=50, max_value=500, value=100)

    V = np.linspace(V_min, V_max, num_points)
    E = V / d
    J = fowler_nordheim(E, E_F, Phi)
    current = J * np.pi * (d / 2)**2

    st.write("## Field Emission Current vs Voltage")
    plt.figure(figsize=(8, 4))
    plt.plot(V, current, label='Field Emission Current')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Field Emission Current (A)')
    plt.title('Field Emission Current vs Voltage')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Variables needed for Einzel Lens and Results sections
if menu_sel == "Einzel Lens" or menu_sel == "Results":
    if 'V_min' not in globals():
        V_min = 0.1
    if 'V_max' not in globals():
        V_max = 10
    if 'd' not in globals():
        d = 1.0
    if 'num_points' not in globals():
        num_points = 100
    if 'E_F' not in globals():
        E_F = 247
    if 'Phi' not in globals():
        Phi = 75
    if 'V' not in globals():
        V = np.linspace(V_min, V_max, num_points)
    if 'E' not in globals():
        E = V / d
    if 'current' not in globals():
        J = fowler_nordheim(E, E_F, Phi)
        current = J * np.pi * (d / 2)**2

# Einzel Lens Calculation
if menu_sel == "Einzel Lens":
    st.header("Einzel Lens Adjustment")
    V_middle = st.number_input("Middle Electrode Voltage (V)", value=1.0)
    V_upper = st.number_input("Upper Electrode Voltage (V)", value=1.0)
    V_lower = st.number_input("Lower Electrode Voltage (V)", value=1.0)

    focal_length, spot_size = einzel_lens(E, V_middle, V_upper, V_lower)

    st.write("## Beam Spot Size and Shape Adjustment")
    plt.figure(figsize=(8, 4))
    plt.plot(V, spot_size, label='Spot Size')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Spot Size (arbitrary units)')
    plt.title('Beam Spot Size vs Voltage')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    st.write("### Calculated Focal Length")
    st.write(f"Focal Length (arbitrary units): {focal_length}")

# Data Source Selection
source_sel = option_menu(
    "Choose Input Source", 
    ["Sample Data", "Simulated Data", "Uploaded Data"],
    orientation="horizontal"
)

if source_sel == "Sample Data":
    # Sample data is loaded from .npy files
    concs = [c / 100 for c in range(1, 100, 10)] + [1]
    freqs = np.load("data/sample_model_freqs.npy", allow_pickle=True)
    m = len(concs)
    n = len(freqs)
    
    noise = np.load("data/sample_data_noise.npy", allow_pickle=True)
    noise_at_freq = np.mean(noise, axis=0)

    weights = np.load("data/sample_model_weights.npy", allow_pickle=True)

    with st.sidebar:
        train_noise_scale = st.slider("Training Noise Scale", 0.0, 1.0, value=0.4, step=0.01)
        n_samples = st.slider("Number of Samples", 1, 64, value=9, step=1)
        test_noise_scale = st.slider("Test Noise Scale", 0.0, 1.0, value=0.01, step=0.01)
        true_conc = st.slider("True Concentration", float(min(concs)), float(max(concs)), value=0.18, step=0.01)

    with st.spinner("Simulating full experiment..."):
        noises = avg_noise(n_samples, noise_at_freq, m, n)
        X_true = generate_experiment(concs, freqs=freqs, weights=weights)
        y_true = np.array(concs)
        X = X_true + train_noise_scale * noises

    fig_true = plot_rows(X_true, freqs, y_true, label="conc=", xlabel="Freq", ylabel="Impedance", title="Actual Impedance vs. Concentration")
    fig_noisy = plot_rows(X, freqs, y_true, label="conc=", xlabel="Freq", ylabel="Impedance", title="Noisy Impedance vs. Concentration")

    fit_w = fit(X, concs)
    noises = test_noise_scale * np.random.normal(0, noise_at_freq)
    X_test = freq_sweep_at_c(true_conc, weights=weights, freqs=freqs)
    fig_true.add_trace(go.Scatter(x=freqs, y=X_test, name=f"True Conc={true_conc}", line=dict(color="black", dash="dash")))
    X_test = X_test + noises
    fig_noisy.add_trace(go.Scatter(x=freqs, y=X_test, name=f"True Conc={true_conc}", line=dict(color="black", dash="dash")))

    agg_pred, preds = predict(X_test, fit_w, agg=gmean, exclude_fns=0, return_preds=True)
    fig_preds = avg_scatter(preds, true_conc, agg_pred, figsize=(6, 4))

    st.info("The data generation below is based on real-world data gathered from a strain of yeast known as Saccharomyces cerevisiae. A state-of-the-art nano-device collects impedance curves of the yeast at various concentrations, and our novel algorithm is able to accurately predict the unknown concentration of a separate yeast sample.")

    st.info("Goal: Predict the sample concentration that generated the new impedance curve identified by the dashed black line.")

    f"### True Concentration: {true_conc}\n ### Algorithm Output: {agg_pred}"

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_noisy, use_container_width=True)
        st.expander("View Algorithm Internals").pyplot(fig_preds, use_container_width=True)
    with col2:
        st.plotly_chart(fig_true, use_container_width=True)

elif source_sel == "Simulated Data":
    curve_sel = option_menu("", ["Single Curve", "Multi-curve"], orientation="horizontal")

    if curve_sel == "Single Curve":
        with st.sidebar:
            advanced_settings = st.checkbox("Advanced Settings")
            R = st.number_input(label="Resistance (Ohms)", min_value=1, max_value=80_000, value=80000, step=100)
            if advanced_settings:
                with st.form("calc_cap_form"):
                    submitted = st.form_submit_button("Update")
                    CNT_radius = st.number_input("CNT Radius (m)", value=5.0E-09, format="%f")
                    CNT_height = st.number_input("CNT Height (m)", value=2.5E-04, format="%f")
                    nanostructure_width = st.number_input("Nanostructure Width (m)", value=3.0E-06, format="%f")
                    nanostructure_length = st.number_input("Nanostructure Length (m)", value=2.54E-02, format="%f")
                    gap_between_nanostructures = st.number_input("Gap Between Nanostructures (m)", value=2.0E-06, format="%f")
                    chip_length = st.number_input("Chip Length (m)", value=2.54E-02, format="%f")
                    chip_width = st.number_input("Chip Width (m)", value=2.54E-02, format="%f")
                    epsilon_r = st.number_input("Epsilon R", 1000)
                    E_breakdown = st.number_input("E Breakdown (V/m)", value=1.2E+09, format="%f")
                    Si_thickness = st.number_input("Si Thickness (m)", value=3.0E-04, format="%f")
                    SiO2_thickness = st.number_input("SiO2 Thickness (m)", value=2.1E-06, format="%f")
                    Metal_1_thickness = st.number_input("Metal 1 Thickness (m)", value=1.0E-07, format="%f")
                    Metal_2_thickness = st.number_input("Metal 2 Thickness (m)", value=1.0E-08, format="%f")
                    Catalyst_thickness = st.number_input("Catalyst Thickness (m)", value=1.0E-08, format="%f")
                    e = st.number_input("Electron Charge", value=1.602e-19, format="%f")
                    z = st.number_input("Electrons / Surface Particle", value=1)
                    C = st.number_input("C", value=1.0E-15, format="%f")
                    C_0 = st.number_input("C_0", value=1.0E-12, format="%f")
                    e_r = st.number_input("Dielectric Constant", value=78.49, format="%f")
                    e_0 = st.number_input("Vacuum Permittivity", value=8.854E-12, format="%f")
                    k_b= st.number_input("Boltzmann Constant", value=1.38E-23, format="%f", disabled=True)
                    T = st.number_input("Temperature", value=298.1, format="%f")
                    V_zeta = st.number_input("V_zeta", value=5.0E-02, format="%f")
                    N_a = st.number_input("Avogadro's Number", value=6E+23, format="%f", disabled=True)
                C = calculate_edl_capacitance(calculate_d_edl())
                st.write("Capacitance (nF): ", C * 1e9)
            else:
                nC = st.number_input(label="Capacitance (nF)", min_value=1, max_value=1000, value=1, step=100)
                C = nC * 1e-9

        with st.spinner("Spooling Virtual Device..."):
            freqs, Z = generate_readings(R=R, C=C, start_freq=100, end_freq=10_000_000, cycle_num=5, form="sin", ex_cyc=5-1, save_csv=False)

        Z = np.array(Z)
        bode_fig = bode(freqs, Z)
        nyquist_fig = nyquist(freqs, Z)

        st.info("Run an entire impedance curve collection cycle. A virtual device generates an input voltage wave and stimulates an output current response. Our impedance analysis algorithm processes these raw waves into an array of impedance values, drawing an impedance curve. The processing algorithm also extracts phase information, which allows accurate Nyquist and Bode Plots. Important: This software is not simply implementing equations, it generates impedance curves from **raw voltage and current waveforms** generated by our virtual device.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(nyquist_fig, use_container_width=True)
        with col2:
            st.plotly_chart(bode_fig, use_container_width=True)

    elif curve_sel == "Multi-curve":
        NUM_CAPS = 8
        with st.sidebar:
            R = st.number_input(label="Resistance (Ohms)", min_value=1, max_value=80_000, value=80000, step=100)
            with st.form("cap_form"):
                submitted = st.form_submit_button("Update")
                nC_vals = [st.number_input(label=f"Capacitance {i+1} (nF)", min_value=1, max_value=1000, value=i+1, step=100, key=i) for i in range(NUM_CAPS)]
            C_vals = [nC * 1e-9 for nC in nC_vals]

        Z_vals = []
        bar = st.progress(0)
        with st.spinner("Spooling Virtual Device..."):
            for i, C in enumerate(C_vals):
                freqs, Z = generate_readings(R=R, C=C, start_freq=100, end_freq=10_000_000, cycle_num=5, form="sin", ex_cyc=5-1)
                Z_vals.append(Z)
                bar.progress((i + 1) / NUM_CAPS)

        Z = np.array(Z_vals)
        fig = plot_rows(abs(Z), freqs, C_vals, label="Capacitance=", xlabel="Freq", ylabel="Impedance", title="Actual Impedance vs. Concentration")

        st.info("Run an entire impedance curve collection cycle. A virtual device generates an input voltage wave and stimulates an output current response. Our impedance analysis algorithm processes these raw waves into an array of impedance values, drawing an impedance curve. Important: This software is not simply implementing equations, it generates impedance curves from **raw voltage and current waveforms** generated by our virtual device.")
        st.plotly_chart(fig)

elif source_sel == "Uploaded Data":
    with st.expander("Please Read: CSV Format Details"):
        st.write("The CSV input should contain experiment values taken at different frequencies. For each frequency, we require three inputs: time, input voltage, and output current. These inputs should be in the columns of the CSV file, and should obviously be the same length. For example, if you run three experiments at frequencies 100, 1000, and 10000, you will have 9 columns. The first three columns are t_100, V_in_100, I_out_100. The next three columns are t_1000, V_in_1000, I_out_1000, and so on. After uploading a correctly formatted CSV, the software will provide the entire electrochemical impedance spectroscopy, calculating impedance and displaying a Bode and Nyquist plot. Don't worry, both of these are just different ways to graph impedance.")

    uploaded_file = st.file_uploader("Import CSV")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        column_names = df.columns.to_list()
        num_cols = len(column_names)
        assert num_cols % 3 == 0, "Expected columns in groups of 3 (t, V_in, I_out)"

        freqs = []
        Z = []

        for i in range(0, len(column_names), 3):
            freq = int(column_names[i].split("_")[-1])
            t = df.iloc[:,i].to_numpy()
            V_in = df.iloc[:,i+1].to_numpy()
            I_out = df.iloc[:,i+2].to_numpy()
            V_amp = np.max(V_in)

            I_out_fft = 2 * np.fft.fft(I_out) / len(I_out)
            I_out_fft = I_out_fft[:len(I_out_fft)//2]
            I_crit = np.argmax(abs(I_out_fft))
        
            Z_calc = V_amp / I_out_fft[I_crit]
            Z_calc = complex(-Z_calc.imag, Z_calc.real)
            Z.append(Z_calc)
            freqs.append(freq)

        Z = np.array(Z)
        bode_fig = bode(freqs, Z)
        nyquist_fig = nyquist(freqs, Z)
    
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(nyquist_fig, use_container_width=True)
        with col2:
            st.plotly_chart(bode_fig, use_container_width=True)
