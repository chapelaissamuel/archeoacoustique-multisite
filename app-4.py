import streamlit as st
import numpy as np
import requests
import json as _json
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, csd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ArchéoAcoustique | BARABAR Multi-Sites · Ic v2.1 · MAICR",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# DONNÉES ARCHÉOLOGIQUES DOCUMENTÉES
# ============================================================================
@dataclass
class SiteData:
    id: str
    name: str
    location: str
    chamber_resonance: Optional[float]
    sarcophagus_resonance: Optional[float]
    documented_frequency: Optional[float]
    infrasound: Optional[float]
    frequency_range: Optional[Tuple[float, float]]
    average_frequency: Optional[float]
    material: str
    odf: Optional[float]
    material_type: str
    sources: List[Dict]
    description: str
    beat_frequency: Optional[float] = None

    def get_resonance_badge(self) -> Tuple[str, str]:
        if self.chamber_resonance is not None:
            return (self._badge("✅ Mesuré", "#00d48a"), self._get_source_text("chamber_resonance"))
        return (self._badge("⚠️ Estimation", "#ffa726"), "Calcul acoustique théorique basé sur les dimensions")

    def get_frequency_badge(self) -> Tuple[str, str]:
        if self.documented_frequency is not None:
            return (self._badge("✅ Documenté", "#00d48a"), self._get_source_text("documented_frequency"))
        if self.frequency_range is not None:
            return (self._badge("✅ Plage mesurée", "#00d48a"), self._get_source_text("frequency_range"))
        return (self._badge("⚠️ Estimation", "#ffa726"), "Extrapolation à partir de sites comparables")

    def _badge(self, text: str, color: str) -> str:
        bg = color + "26"
        border = color + "4d"
        return (f'<span style="display:inline-block;padding:0.2rem 0.6rem;border-radius:12px;'
                f'font-size:0.68rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;'
                f'background:{bg};color:{color};border:1px solid {border};">{text}</span>')

    def _get_source_text(self, field: str) -> str:
        for source in self.sources:
            if source.get("field") == field or source.get("field") == "all":
                return f"{source['author']} ({source['year']})"
        return "Source non spécifiée"


# ============================================================================
# SITES DU MONDE — Presets pour simulation universelle
# ============================================================================
WORLD_PRESETS = {
    "Barabar - Chambre Sudama (Inde)": {
        "freq": 34.4, "odf": 4.2, "press_db": 102, "Q": 950,
        "material_type": "granite", "noise": 1.5,
        "note": "BAM 2023-2024. Battement 41.1 Hz. Validation predictive 75.5 Hz (Chapelais 2026)."
    },
    "Hal Saflieni - Oracle Room (Malte)": {
        "freq": 114.0, "odf": 2.5, "press_db": 95, "Q": 600,
        "material_type": "limestone", "noise": 1.5,
        "note": "Double pic 70 Hz + 114 Hz mesure in situ. Debertolis 2014 Univ. Trieste. EEG Cook et al. 2008."
    },
    "Grande Pyramide - Chambre du Roi (Egypte)": {
        "freq": 117.0, "odf": 3.8, "press_db": 100, "Q": 800,
        "material_type": "granite", "noise": 1.5,
        "note": "Sarcophage 114-122 Hz mesure Danley 1997. Granite Assouan f_qz~0.32."
    },
    "Newgrange - Chambre interieure (Irlande)": {
        "freq": 110.0, "odf": 2.8, "press_db": 90, "Q": 400,
        "material_type": "mixed", "noise": 1.5,
        "note": "110 Hz mesure in situ 1994. Jahn, Devereux et al. 1996 JASA. Princeton PEAR Lab."
    },
    "Lascaux - Salle des Taureaux (France)": {
        "freq": 250.0, "odf": 1.2, "press_db": 85, "Q": 150,
        "material_type": "limestone", "noise": 1.5,
        "note": "ATTENTION : resonance documentee ~250 Hz (bande 1/3 oct., Commins et al. 2020 JASA). Reznikoff 1988 utilisait la voix (D~300Hz/C~260Hz), pas de freq fixe. Freq 250 Hz = estimation conservative."
    },
    "Chichen Itza - El Castillo (Mexique)": {
        "freq": 633.0, "odf": 1.0, "press_db": 95, "Q": 80,
        "material_type": "limestone", "noise": 1.5,
        "note": "ATTENTION : effet chirp echo EXTERNE (diffraction grating escalier), 796 Hz -> 471 Hz en 177ms. Pas de resonance de chambre. Lubman 1998 JASA, Declercq 2004 JASA. Freq = centre chirp (~633 Hz). Simulation tres approximative."
    },
    "Stonehenge (Angleterre)": {
        "freq": 95.0, "odf": 2.2, "press_db": 88, "Q": 180,
        "material_type": "sandstone", "noise": 1.5,
        "note": "ATTENTION : pas de freq de resonance unique documentee. Cox, Fazenda & Greaney 2020 Journal of Archaeological Science : effets acoustiques complexes sur maquette 1:12. Freq 95 Hz = estimation basse du spectre observe."
    },
    "Chavin de Huantar - Galeries (Perou)": {
        "freq": 95.0, "odf": 2.0, "press_db": 90, "Q": 200,
        "material_type": "mixed", "noise": 1.5,
        "note": "ATTENTION : pas de freq unique documentee. Kolar et al. 2008-2013 (Stanford CCRMA) : resonances modales galeries alignees sur plage frequentielle pututus (trompettes coquillage). Freq 95 Hz = estimation basse plage vocale masculine."
    },
    "Site personnalise (libre)": {
        "freq": 40.0, "odf": 3.0, "press_db": 95, "Q": 400,
        "material_type": "granite", "noise": 1.5,
        "note": "Entrez vos propres parametres. Consultez la documentation pour les plages valides."
    },
}


SITES_DB = {
    "barabar": SiteData(
        id="barabar",
        name="Grotte de Sudama",
        location="Barabar, Bihar, Inde",
        chamber_resonance=34.4,
        sarcophagus_resonance=None,
        documented_frequency=75.5,
        infrasound=None,
        frequency_range=None,
        average_frequency=None,
        material="Granite poli",
        odf=4.2,
        material_type="granite",
        sources=[
            {"field": "chamber_resonance", "author": "Jérôme Paquereau, Patrice Pouillard",
             "year": "2023-2024", "title": "BARABAR Acoustics Mission (BAM)", "type": "Mesures acoustiques in situ"},
            {"field": "documented_frequency", "author": "Jérôme Paquereau, Patrice Pouillard",
             "year": "2023-2024", "title": "BARABAR Acoustics Mission (BAM)", "type": "Mesures acoustiques in situ"},
            {"field": "odf", "author": "Analyse géologique", "year": "2023",
             "title": "Caractérisation du granite de Barabar (CGC Gaya, A-type felsique évolué)", "type": "Analyse de texture cristalline — f_qz=0.32, ODF estimé 4.2"},
            {"field": "all", "author": "Díaz-Andreu", "year": "2025",
             "title": "Archaeoacoustics: Research on Past Musics and Sounds",
             "institution": "ICREA / Université de Barcelone", "type": "Annual Review of Anthropology 54:113-130 · Revue systématique 2025 incluant neuroacoustique sites sacrés — contexte international BARABAR"}
        ],
        description="Les grottes de Barabar (IIIe siècle av. J.-C.) sont les plus anciennes chambres rupestres d'Inde, taillées dans du granite massif avec une précision remarquable. La chambre elliptique de Sudama présente une résonance naturelle à 34,4 Hz. Un chant à 75,5 Hz génère un battement de 41,1 Hz — directement dans la bande gamma cérébrale.",
        beat_frequency=41.1
    ),
    "pyramide": SiteData(
        id="pyramide",
        name="Chambre du Roi",
        location="Grande Pyramide, Gizeh, Égypte",
        chamber_resonance=117.0,
        sarcophagus_resonance=117.0,  # Pic moyen Danley 1997 (plage 114-122 Hz, Reid cymatics)
        documented_frequency=None,
        infrasound=16.0,
        frequency_range=None,
        average_frequency=None,
        material="Granite d'Assouan",
        odf=3.8,
        material_type="granite",
        sources=[
            {"field": "chamber_resonance", "author": "Tom Danley", "year": "1997",
             "title": "Acoustical Study of the Great Pyramid", "type": "Étude acoustique scientifique"},
            {"field": "sarcophagus_resonance", "author": "Tom Danley", "year": "1997",
             "title": "Acoustical Study of the Great Pyramid", "type": "Mesures du sarcophage en granite"},
            {"field": "infrasound", "author": "Tom Danley", "year": "1997",
             "title": "Acoustical Study of the Great Pyramid", "type": "Mesures infrasonores"},
            {"field": "sarcophagus_resonance", "author": "John Stuart Reid", "year": "1997",
             "title": "Cymatics Experiment in the Great Pyramid — King's Chamber sarcophagus",
             "type": "Expérience cymatique in situ — pics résonance sarcophage 114-122 Hz, granite haute teneur quartz"},
            {"field": "all", "author": "Collins & Hale (Ancient Origins)", "year": "2019",
             "title": "The Great Pyramid Experiment — Infrasound measurements",
             "type": "Mesures spectrales chambers — King's Chamber: 30-130 Hz cluster, infrasound 15-16 Hz, sarcophage: 114-122 Hz"}
        ],
        description="La Chambre du Roi de la Grande Pyramide de Khéops (vers 2560 av. J.-C.) est entièrement construite en granite rose d'Assouan à haute teneur en quartz. Tom Danley (1997) a mesuré un eigenmode de chambre à ~121 Hz et des résonances basses à 30-49 Hz. John Stuart Reid (1997 cymatics) a documenté des pics de résonance du sarcophage entre 114 et 122 Hz — dans la plage vocale masculine. Les infrasons à ~16 Hz sont générés par les puits d'aération (Danley 1997 : vent traversant les shafts). Ces données convergent avec la plage 110-120 Hz documentée cross-culturellement par Cook 2008 et Jahn 1996 pour les chambres mégalithiques.",
        beat_frequency=101.0  # 117 Hz (chambre) - 16 Hz (infrasound shafts) = 101 Hz
    ),
    "saflieni": SiteData(
        id="saflieni",
        name="Oracle Room (Ħal Saflieni)",
        location="Paola, Malte",
        chamber_resonance=114.0,
        sarcophagus_resonance=None,
        documented_frequency=70.0,
        infrasound=None,
        frequency_range=(95.0, 120.0),
        average_frequency=110.0,
        material="Calcaire globigerina",
        odf=2.5,
        material_type="limestone",
        sources=[
            {"field": "chamber_resonance", "author": "Debertolis, Coimbra & Eneix", "year": "2014",
             "title": "Archaeoacoustic Analysis of the Ħal Saflieni Hypogeum in Malta",
             "institution": "Université de Trieste / SBRG", "type": "Analyse acoustique in situ — double résonance 70 Hz et 114 Hz"},
            {"field": "documented_frequency", "author": "Debertolis, Coimbra & Eneix", "year": "2014",
             "title": "Archaeoacoustic Analysis of the Ħal Saflieni Hypogeum in Malta",
             "institution": "Université de Trieste / SBRG", "type": "Premier pic de résonance — 70 Hz (basse masculine) — Journal of Anthropology and Archaeology 3:59-79"},
            {"field": "frequency_range", "author": "Rupert Till", "year": "2017",
             "title": "An archaeoacoustic study of the Hal Saflieni Hypogeum on Malta",
             "institution": "Université de Huddersfield", "type": "Antiquity 91(355):74-89 — DOI:10.15184/aqy.2016.258 · Étude systématique peer-reviewed"},
            {"field": "chamber_resonance", "author": "Dr. Ian Cook, Pajot & Leuchter", "year": "2008",
             "title": "Ancient Architectural Acoustic Resonance Patterns and Regional Brain Activity",
             "institution": "UCLA", "type": "Time and Mind vol.1 No.1 — EEG 110 Hz sites mégalithiques, préfrontal shift"},
            {"field": "all", "author": "Debertolis, Tirelli & Monti", "year": "2014",
             "title": "EEG volunteers 90-120 Hz — effets sur activité cérébrale",
             "institution": "Clinique Neurophysiologie Trieste", "type": "Étude EEG in situ · 90-120 Hz modulent activité neuronale"}
        ],
        description="L'Hypogée de Ħal Saflieni (3600-2500 av. J.-C.) est un complexe souterrain préhistorique unique. La Chambre de l'Oracle présente une DOUBLE résonance documentée : 70 Hz et 114 Hz (Debertolis et al. 2014, SBRG/Trieste). Une voix de basse accordée à ces fréquences stimule la résonance dans tout l'hypogée avec des échos jusqu'à 13 secondes. Les études EEG (Cook 2008 UCLA, Debertolis 2014 Trieste) montrent que les fréquences 90-120 Hz modulent l'activité cérébrale. NB : la valeur 110 Hz est la moyenne multi-sites (Cook 2008) — la valeur spécifique Saflieni est 114 Hz (deuxième pic) et 70 Hz (premier pic).",
        beat_frequency=44.0
    ),
    "megalithes": SiteData(
        id="megalithes",
        name="Sites Mégalithiques",
        location="Îles Britanniques & Europe",
        chamber_resonance=None,
        sarcophagus_resonance=None,
        documented_frequency=None,
        infrasound=None,
        frequency_range=(95.0, 120.0),
        average_frequency=110.0,
        material="Divers (grès, calcaire, granite)",
        odf=3.0,
        material_type="mixed",
        sources=[
            {"field": "frequency_range", "author": "Aaron Watson & David Keating", "year": "1999",
             "title": "Architecture and Sound: An Acoustic Analysis of Megalithic Monuments in Prehistoric Britain",
             "type": "Antiquity 73:325-336 — Méta-analyse Newgrange, Maeshowe, Wayland's Smithy · résonances 95-120 Hz"},
            {"field": "all", "author": "Robert G. Jahn et al.", "year": "1996",
             "title": "Acoustical Resonances of Assorted Ancient Structures",
             "institution": "Princeton PEAR Lab", "type": "JASA 99:649 — DOI:10.1121/1.414544 · 6 sites UK/Irlande, pattern 95-120 Hz"},
            {"field": "all", "author": "Rupert Till, Bruno Fazenda & Trevor Cox", "year": "2017",
             "title": "An archaeoacoustic study of the Hal Saflieni Hypogeum on Malta",
             "institution": "Université de Huddersfield", "type": "Antiquity 91(355):74-89 · Revue comparative internationale"},
            {"field": "frequency_range", "author": "Cox, Fazenda & Greaney", "year": "2020",
             "title": "Using scale modelling to assess the prehistoric acoustics of Stonehenge",
             "type": "Journal of Archaeological Science 122:105218 · Maquette 1/12 — effets acoustiques Stonehenge documentés"},
            {"field": "all", "author": "Díaz-Andreu", "year": "2025",
             "title": "Archaeoacoustics: Research on Past Musics and Sounds",
             "institution": "ICREA / Université de Barcelone", "type": "Annual Review of Anthropology 54:113-130 · Revue systématique 2025 — état de l'art field"}
        ],
        description="Sites mégalithiques comparatifs : Newgrange (Irlande, 3200 av. J.-C.) — résonance interne ~110 Hz mesurée par Jahn 1996 (Princeton PEAR), Stonehenge (Angleterre) — effets acoustiques documentés par Cox, Fazenda & Greaney 2020 (JASA scale model 1/12), Wayland's Smithy et cairns du Wiltshire. Toutes ces structures partagent une concentration de résonances dans la plage 95-120 Hz (Watson & Keating 1999, Jahn 1996). Le matériau varie (grès, calcaire, granite) expliquant l'ODF mixte et la piézoélectricité variable. Díaz-Andreu 2025 (Annual Review of Anthropology) légitime ce champ comparatif cross-culturel.",
        beat_frequency=None
    )
}

# ============================================================================
# CSS & THEME  (même base que Barabar v2.1)
# ============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #070b14; color: #dde3f0; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 1600px !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #0f1929 100%) !important;
        border-right: 1px solid rgba(99,157,255,0.12) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] label {
        color: #8da0c0 !important; font-size: 0.78rem !important;
        font-weight: 500 !important; text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3a7ef4 0%, #6b4ef7 100%) !important;
        border: none !important; color: #fff !important;
        font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
        font-size: 0.88rem !important; letter-spacing: 0.06em !important;
        padding: 0.65rem 1.2rem !important; border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(58,126,244,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(107,78,247,0.55) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important; padding: 1.1rem 1.3rem !important;
    }
    [data-testid="stMetricLabel"] { color: #8da0c0 !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] {
        color: #639dff !important; font-weight: 700 !important;
        font-size: 1.6rem !important; font-family: 'JetBrains Mono', monospace !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3a7ef4, #6b4ef7) !important;
        border-radius: 6px !important;
    }
    .stProgress > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 6px !important; }

    hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 1.8rem 0 !important; }
    .stSpinner > div { border-top-color: #3a7ef4 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MATPLOTLIB THEME
# ============================================================================
def setup_mpl_theme():
    plt.rcParams.update({
        'figure.facecolor': '#0d1521', 'axes.facecolor': '#0f1b2d',
        'axes.edgecolor': '#1e2d45', 'axes.labelcolor': '#8da0c0',
        'axes.titlecolor': '#dde3f0', 'axes.titlesize': 11,
        'axes.labelsize': 9.5, 'axes.titleweight': '600',
        'axes.grid': True, 'grid.color': '#1a2a40', 'grid.linewidth': 0.8,
        'xtick.color': '#5a7090', 'ytick.color': '#5a7090',
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'text.color': '#dde3f0', 'font.family': 'DejaVu Sans', 'font.size': 9.5,
        'lines.linewidth': 2.0, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 120, 'savefig.dpi': 120, 'savefig.facecolor': '#0d1521',
        'legend.facecolor': '#0f1b2d', 'legend.edgecolor': '#1e2d45',
        'legend.framealpha': 0.9, 'legend.fontsize': 8.5,
    })


# ============================================================================
# PHYSICS ENGINE
# ============================================================================
class PhysicsEngine:
    # Fraction volumique quartz par type de matériau (Bishop 1981)
    F_QZ = {
        "granite":   0.32,   # Granite A-type CGC Gaya — Bishop 1981, Barabar/Pyramide
        "limestone":  0.02,  # Calcaire globigerina — quartz trace (~2%) — Saflieni
        "sandstone":  0.18,  # Grès quartzeux intermédiaire
        "mixed":      0.15,  # Sites mégalithiques matériaux mixtes
    }
    # Permittivité effective par matériau
    EPS_EFF = {
        "granite":   6.65,   # Permittivité effective granite quartzifère
        "limestone":  8.20,  # Calcaire : permittivité plus élevée (εr ≈ 7-9)
        "sandstone":  4.80,  # Grès quartzeux
        "mixed":      6.00,  # Mixte
    }

    def __init__(self, material_type: str = "granite"):
        self.mu_0 = 4 * np.pi * 1e-7
        self.eps_0 = 8.854e-12
        self.d33 = 2.31e-12
        mat = material_type if material_type in self.F_QZ else "granite"
        self.f_qz = self.F_QZ[mat]
        self.eps_eff = self.EPS_EFF[mat] * self.eps_0
        self.sigma_skull = 0.006
        self.sigma_brain = 0.33
        self.thickness = 0.007
        self.material_type = mat

    def orientation_factor(self, odf):
        return (1 - np.exp(-(odf - 1) / 2)) * 0.978

    def piezo_conversion(self, P_pa, odf, f_hz):
        F = self.orientation_factor(odf)
        sigma = P_pa * 0.8 * F
        D3 = self.d33 * self.f_qz * (odf ** 2) * sigma * 0.7
        E = D3 / self.eps_eff
        dDdt = D3 * 2 * np.pi * f_hz
        B = (self.mu_0 / (4 * np.pi)) * dDdt / (0.3 ** 3)
        return E, B

    def skull_transmission(self, f_hz):
        w = 2 * np.pi * f_hz
        Z_s = np.sqrt(1j * w * self.mu_0 / self.sigma_skull)
        Z_b = np.sqrt(1j * w * self.mu_0 / self.sigma_brain)
        G = (Z_s - Z_b) / (Z_s + Z_b)
        delta = 0.12 * np.sqrt(40 / max(f_hz, 1))
        att = np.exp(-self.thickness / delta)
        T = np.abs(((1 - G ** 2) / (1 + G ** 2 * np.exp(-0.014 / delta))) * att)
        return T, delta

    def ellipse_gain(self, Q):
        return np.sqrt(Q * 2), 5.4


# ============================================================================
# NEURAL SIMULATION (Kuramoto N=3000)
# ============================================================================
class NeuralSim:
    def __init__(self, N=3000):
        self.N = N
        self.dt = 0.0005

    def kuramoto(self, f_hz, E_uv, noise_mv, T=2.0):
        n_steps = int(T / self.dt)
        t = np.arange(0, T, self.dt)
        omega = 2 * np.pi * np.random.normal(f_hz, 2.0, self.N)
        theta = np.random.uniform(0, 2 * np.pi, self.N)
        I_em = 0.5 * (E_uv * 1e-6) * 0.33 * 1e-4
        D = (noise_mv * 1e-3) ** 2
        noise = np.sqrt(2 * D * self.dt) * np.random.randn(n_steps, self.N)
        history = np.zeros((n_steps, self.N))
        for i in range(n_steps):
            history[i] = theta
            mean_p = np.angle(np.mean(np.exp(1j * theta)))
            dtheta = (omega + 2.0 * np.sin(mean_p - theta)
                      + I_em * np.sin(2 * np.pi * f_hz * t[i])
                      + noise[i] / self.dt)
            theta = np.mod(theta + dtheta * self.dt, 2 * np.pi)
        return t, history

    def plv(self, history, f_hz) -> float:
        n = history.shape[0]
        t = np.arange(n) * self.dt
        target = 2 * np.pi * f_hz * t
        mean_th = np.angle(np.mean(np.exp(1j * history), axis=1))
        return float(np.abs(np.mean(np.exp(1j * (mean_th - target)))))

    def sr_curve(self, E_uv, f_hz, noises):
        snrs = []
        for n in noises:
            _, h = self.kuramoto(f_hz, E_uv, n, T=1.5)
            p = self.plv(h, f_hz)
            snrs.append(p ** 2 / (n ** 2 + 0.01))
        snrs = np.array(snrs)
        return noises, snrs, float(noises[np.argmax(snrs)]), float(np.max(snrs))


# ============================================================================
# Ic ALGORITHM
# ============================================================================
class IcAlgorithm:
    """
    Ic v2.1 — Fusion Bayésienne calibrée Monte-Carlo (aligné BARABAR v4.0)
    Poids : C=0.35, T=0.30, G=0.15, PAC=0.20
    PAC ancré sur Huang et al. 2026 PNAS :
      slow gamma 30-70 Hz → retrieval hippocampique
      fast gamma 70-140 Hz → encodage entorhinal
    Seuils : H0 < 0.35 ≤ INDETERMINE < 0.65 ≤ H1
    """
    def __init__(self):
        self.w      = [0.35, 0.30, 0.15, 0.20]  # C, T, G, PAC
        self.s_low  = 0.35
        self.s_high = 0.65

    def coherence(self, s1, s2, fs=1000):
        n = min(256, len(s1) // 4)
        f, P1 = welch(s1, fs, nperseg=n)
        f, P2 = welch(s2, fs, nperseg=n)
        f, Pc = csd(s1, s2, fs, nperseg=n)
        g2 = np.abs(Pc) ** 2 / (P1 * P2 + 1e-20)
        w_snr = 1 - np.exp(-P2 / (np.mean(P2[:10]) + 1e-10) / 10)
        return float(np.clip(np.mean(g2 * w_snr), 0, 1))

    def power_law(self, Press, Bfields):
        if len(Press) < 2 or np.any(Press <= 0) or np.any(Bfields <= 0):
            return 0.0, 0.0, 0.0
        if len(np.unique(Press)) < 2 or len(np.unique(Bfields)) < 2:
            return 0.0, 0.0, 0.0
        lp, lb = np.log(Press), np.log(Bfields)
        if np.std(lp) < 1e-12 or np.std(lb) < 1e-12:
            return 0.0, 0.0, 0.0
        try:
            slope, _, r, p, _ = stats.linregress(lp, lb)
        except Exception:
            return 0.0, 0.0, 0.0
        if p >= 0.05:
            return 0.0, float(slope), float(r ** 2)
        qual = r ** 2 * np.exp(-abs(slope - 0.5) / 0.5)
        return (float(np.clip(np.nan_to_num(qual), 0, 1)),
                float(np.nan_to_num(slope)),
                float(np.nan_to_num(r ** 2)))

    def geo_factor(self, odf):
        x = np.linspace(0, 10, 500)
        p = stats.norm.pdf(x, odf, 0.5)
        p = p / np.sum(p)
        pc = 1 / (1 + np.exp(-2 * (x - 3)))
        return float(np.clip(np.sum(pc * p) * 0.85, 0, 1))

    def pac_score(self, plv: float, freq_hz: float) -> float:
        """PAC proxy — Huang et al. 2026 PNAS (slow/fast gamma)."""
        plv_c = float(np.clip(plv, 0.0, 1.0))
        if 30.0 <= freq_hz <= 70.0:
            return float(np.clip(plv_c ** 0.7, 0, 1))
        elif 70.0 < freq_hz <= 140.0:
            return float(np.clip(plv_c ** 0.85 * 0.75, 0, 1))
        else:
            return float(np.clip(plv_c ** 0.5 * 0.4, 0, 1))

    def fusion(self, C, T, G, PAC=None, odf=None, Q=None, B_nT=None,
               plv=None, acoustic_direct=None, piezo_pathway=None):
        """
        Fusion Ic v2.1 avec garde-fous protocolaires.
        H0 = ECHEC EM. H2 = bifurcation acoustique directe. H1 = SUCCES EM.
        """
        odf_v = float(odf) if odf is not None else 3.0
        Q_v   = float(Q)   if Q   is not None else 950.0
        B_v   = float(B_nT) if B_nT is not None else 0.0
        PAC_v = float(PAC) if PAC is not None else 0.0

        # GARDE-FOU 1 : Q insuffisant
        if Q_v < 20:
            return 0.05, "H₀ VALIDÉE [Q<20]", False, 0.0, "#ff4d4d", "#2d0a0a", "PIÉZO-EM", 0.0, 0.0

        # GARDE-FOU 2 : ODF < 3 → G plafonné
        G_eff = G if odf_v >= 3 else min(G, 0.15)

        # Correction T dégénéré
        T_eff = T if T >= 0.01 else float(np.clip(PAC_v * 0.85, 0, 1))

        scores = [C, T_eff, G_eff, PAC_v]
        std = float(np.std(scores))

        if std > 0.22:
            Ic = (C * 0.50 + PAC_v * 0.50) * 0.60
            conflict = True
        else:
            Ic = float(np.dot(self.w, scores))
            conflict = False

        # GARDE-FOU 3 : bonus EM si conditions D1c
        if odf_v >= 3 and B_v > 1.0:
            Ic = Ic + 0.20

        Ic = float(np.clip(np.nan_to_num(Ic), 0, 1))

        # Dual pathway
        ac_d = C * 0.6 + PAC_v * 0.4
        pz_d = T_eff * 0.7 + G_eff * 0.3
        dominant = "ACOUSTIQUE DIRECTE" if ac_d > pz_d else "PIÉZO-EM"

        if Ic < self.s_low:
            if ac_d > pz_d:
                return Ic, "H₀ → BIFURCATION H₂ (Acoustique)", conflict, std, "#e08c2a", "#120d04", dominant, ac_d, pz_d
            else:
                return Ic, "H₀ VALIDÉE [Échec EM]", conflict, std, "#ff4d4d", "#2d0a0a", dominant, ac_d, pz_d
        elif Ic < self.s_high:
            return Ic, "INDÉTERMINÉ", conflict, std, "#ffa726", "#2d1b00", dominant, ac_d, pz_d
        else:
            return Ic, "H₁ VALIDÉE", conflict, std, "#00d48a", "#002d1b", dominant, ac_d, pz_d


# ============================================================================
# SIMULATION
# ============================================================================
def run_simulation(params: dict, neural: NeuralSim, algo: IcAlgorithm):
    freq, odf, noise = params["freq"], params["odf"], params["noise"]
    press_db, Q = params["press_db"], params["Q"]
    # Recréer une instance PhysicsEngine propre au matériau du site
    # (évite la mutation de l'objet partagé entre simulations)
    mat = params.get("material_type", "granite")
    phys = PhysicsEngine(material_type=mat)

    P_pa = 20e-6 * (10 ** (press_db / 20))
    E_gran, B_gran = phys.piezo_conversion(P_pa, odf, freq)
    T_skull, delta = phys.skull_transmission(freq)
    E_cortex = E_gran * T_skull
    p_gain, e_gain = phys.ellipse_gain(Q)
    E_foyer = E_gran * e_gain
    # Conversion V/m → µV/m pour le moteur Kuramoto
    # kuramoto(E_uv) attend des µV/m : I_em = 0.5*(E_uv*1e-6)*0.33*1e-4
    E_cortex_uvm = E_cortex * 1e6   # V/m → µV/m
    E_gran_uvm   = E_gran   * 1e6   # V/m → µV/m (pour affichage)
    E_foyer_uvm  = E_foyer  * 1e6   # V/m → µV/m (pour affichage)

    n_range = np.linspace(0.1, 3.0, 12)
    n_arr, snr_arr, n_opt, snr_max = neural.sr_curve(E_cortex_uvm, freq, n_range)
    t, theta = neural.kuramoto(freq, E_cortex_uvm, noise, T=2.0)
    plv_val = neural.plv(theta, freq)

    ref_signal = np.sin(2 * np.pi * freq * t)
    mean_theta = np.angle(np.mean(np.exp(1j * theta), axis=1))
    osc_signal = np.sin(mean_theta)
    C = algo.coherence(ref_signal, osc_signal)

    press_levels = np.linspace(P_pa * 0.6, P_pa * 1.4, 5)
    b_levels = np.array([phys.piezo_conversion(p, odf, freq)[1] for p in press_levels])
    T_score, alpha, r2 = algo.power_law(press_levels, b_levels + 1e-30)
    G = algo.geo_factor(odf)

    # Ic v2.1 — PAC + dual pathway
    PAC = algo.pac_score(plv_val, freq)
    B_nT = B_gran * 1e9
    Ic, decision, conflict, std_s, color, bg, dominant, ac_d, pz_d = algo.fusion(
        C, T_score, G, PAC=PAC, odf=odf, Q=Q, B_nT=B_nT,
        plv=plv_val, acoustic_direct=C*0.6+PAC*0.4, piezo_pathway=T_score*0.7+G*0.3
    )

    return {
        "P_pa": P_pa, "E_gran": E_gran_uvm, "B_gran": B_gran, "B_nT": B_nT,
        "T_skull": T_skull, "delta": delta, "E_cortex": E_cortex_uvm,
        "E_foyer": E_foyer_uvm, "e_gain": e_gain,
        "n_arr": n_arr, "snr_arr": snr_arr, "n_opt": n_opt, "snr_max": snr_max,
        "t": t, "theta": theta, "plv": plv_val,
        "C": C, "T": T_score, "G": G, "PAC": PAC,
        "alpha": alpha, "r2": r2,
        "Ic": Ic, "decision": decision, "conflict": conflict,
        "std_s": std_s, "color": color, "bg": bg,
        "dominant": dominant, "acoustic_direct": ac_d, "piezo_pathway": pz_d,
        # Métadonnées pour check_coherence et affichage
        "odf_used": odf,
        "material_type": mat,
        "f_qz_used": PhysicsEngine.F_QZ.get(mat, 0.32),
    }


# ============================================================================
# HTML COMPONENTS (style inline — identique Barabar v2.1)
# ============================================================================
def section_header(num: str, title: str, subtitle: str = "") -> str:
    sub_html = (f'<span style="font-size:0.8rem;color:#8da0c0;margin-left:auto;font-weight:400;">'
                f'{subtitle}</span>') if subtitle else ""
    return f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.2rem;">
        <div style="width:30px;height:30px;background:linear-gradient(135deg,#3a7ef4,#6b4ef7);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:0.8rem;font-weight:700;color:#fff;flex-shrink:0;">{num}</div>
        <span style="font-size:1.05rem;font-weight:600;color:#dde3f0;letter-spacing:-0.01em;">{title}</span>
        {sub_html}
    </div>"""


def phys_card(label, value, unit, sub_label, sub_value, sub_unit, color, accent_gradient) -> str:
    return f"""
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                border-radius:16px;padding:1.4rem 1.6rem;position:relative;overflow:hidden;height:100%;">
        <div style="position:absolute;top:0;left:0;width:3px;height:100%;
                    background:{accent_gradient};border-radius:16px 0 0 16px;"></div>
        <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.6rem;color:{color};">{label}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:2.1rem;font-weight:700;
                    line-height:1.1;margin-bottom:0.25rem;color:{color};">{value}
            <span style="font-size:0.8rem;color:#8da0c0;font-weight:400;">{unit}</span>
        </div>
        <div style="font-size:0.8rem;color:#8da0c0;margin-top:0.7rem;padding-top:0.7rem;
                    border-top:1px solid rgba(255,255,255,0.06);">
            {sub_label}&nbsp;
            <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:#b0bfd8;">{sub_value}</span>
            <span style="font-size:0.72rem;color:#4a6080;"> {sub_unit}</span>
        </div>
    </div>"""


def decision_box(Ic: float, decision: str, color: str, bg: str) -> str:
    return f"""
    <div style="background:{bg};border:1px solid {color}40;border-radius:16px;
                padding:1.6rem;text-align:center;">
        <div style="font-size:0.7rem;font-weight:600;color:{color};text-transform:uppercase;
                    letter-spacing:0.12em;opacity:0.8;">Indice Ic</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:3.5rem;font-weight:700;
                    line-height:1;margin:0.5rem 0;color:{color};">{Ic:.3f}</div>
        <div style="font-size:0.85rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;margin-top:0.6rem;color:{color};">{decision}</div>
        <div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-top:1rem;
                    font-family:'JetBrains Mono',monospace;">seuils · 0.35 | 0.65</div>
    </div>"""


def metrics_table(rows: list) -> str:
    html = ('<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
            'border-radius:14px;padding:1rem 1.2rem;">')
    for key, val, note in rows:
        html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85rem;">
            <span style="color:#8da0c0;">{key}</span>
            <span>
                <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:#c8d4e8;">{val}</span>
                <span style="font-size:0.72rem;color:#4a6080;margin-left:0.4rem;">{note}</span>
            </span>
        </div>"""
    html += "</div>"
    return html


def render_site_data_native(site: SiteData):
    """Affiche les données du site — version robuste.
    Chaque ligne = un petit st.markdown simple, pas de bloc monolithique géant."""

    st.markdown(f"**📊 Données Acoustiques Documentées — {site.name}**")

    rows = []
    if site.chamber_resonance:
        _, source = site.get_resonance_badge()
        rows.append(("Chambre de résonance", f"{site.chamber_resonance:.1f} Hz", "✅ Mesuré", source))
    if site.sarcophagus_resonance:
        source = site._get_source_text("sarcophagus_resonance")
        rows.append(("Résonance sarcophage", f"{site.sarcophagus_resonance:.1f} Hz", "✅ Mesuré", source))
    if site.documented_frequency:
        _, source = site.get_frequency_badge()
        rows.append(("Fréquence documentée", f"{site.documented_frequency:.1f} Hz", "✅ Documenté", source))
    if site.infrasound:
        source = site._get_source_text("infrasound")
        rows.append(("Infrasons", f"{site.infrasound:.1f} Hz", "✅ Mesuré (non simulé — freq sim = chambre)", source))
    if site.frequency_range:
        _, source = site.get_frequency_badge()
        range_str = f"{site.frequency_range[0]:.0f} – {site.frequency_range[1]:.0f} Hz"
        rows.append(("Plage de résonance", range_str, "✅ Plage mesurée", source))
    if site.beat_frequency:
        rows.append(("Battement généré", f"{site.beat_frequency:.1f} Hz", "✅ Calculé",
                     "Calculé à partir des données mesurées"))
    if site.material:
        odf_str = f" · ODF = {site.odf}" if site.odf else ""
        rows.append(("Matériau", f"{site.material}{odf_str}", "✅ Documenté", "Analyse géologique"))

    for label, val, status, source in rows:
        st.markdown(f"**{label}** &mdash; `{val}` {status}", unsafe_allow_html=True)
        st.caption(f"↳ {source}")


# ============================================================================
# PROTOCOLE BAM — VALIDATION INPUTS (Python pur, avant tout LLM)
# ============================================================================
class InputValidationError(Exception):
    pass

def validate_inputs(freq, press_db, odf, Q, noise):
    errors = []
    if not (20.0 <= freq <= 1000.0):     errors.append(f"FREQ {freq:.1f} Hz hors plage [20–1000 Hz]")
    if not (60.0 <= press_db <= 130.0):  errors.append(f"PRESS {press_db} dBSPL hors plage [60–130]")
    if not (1.0 <= odf <= 5.0):          errors.append(f"ODF {odf:.1f} hors plage [1.0–5.0]")
    if not (1 <= Q <= 1000):             errors.append(f"Q {Q} hors plage [1–1000]")
    if not (0.1 <= noise <= 3.0):        errors.append(f"NOISE {noise:.2f} mV hors plage [0.1–3.0]")
    if errors:
        raise InputValidationError("🔴 REJET PROTOCOLE :\n" + "\n".join(f"  • {e}" for e in errors))

def check_coherence(res: dict, freq: float) -> list:
    """Détection Python pur des paradoxes physiques — aucun LLM."""
    flags = []
    decision = res.get("decision", "")
    plv      = res.get("plv", 0)
    odf      = res.get("odf_used") or res.get("odf", 3.0)
    PAC      = res.get("PAC", 0)
    ac_d     = res.get("acoustic_direct", 0)
    pz_d     = res.get("piezo_pathway", 0)
    if "H₀" in decision and plv > 0.5 and odf > 3:
        flags.append(f"PLV={plv:.3f}>0.5 & ODF={odf:.1f}>3 malgré H₀ — probable sous-estimation Ic")
    if "H₁" in decision and res.get("B_nT", 0) <= 1.0:
        flags.append(f"H₁ déclaré mais B={res.get('B_nT',0):.4f} nT ≤ 1 nT — violation protocole")
    if PAC > 0.7 and not (30 <= freq <= 140):
        flags.append(f"PAC={PAC:.3f}>0.7 à {freq:.1f} Hz — hors bandes gamma Huang 2026")
    # Flag spécifique calcaire : voie piézo théoriquement nulle
    f_qz = res.get("f_qz_used", 0.32)
    if f_qz < 0.05 and res.get("piezo_pathway", 0) > 0.2:
        flags.append(
            f"Site calcaire (f_qz={f_qz:.2f}) : voie piézo={res.get('piezo_pathway',0):.3f} "
            f"suspecte — modèle piézoélectrique conçu pour granite (f_qz=0.32)"
        )
    return flags

# ============================================================================
# WHITELIST 20 REFS + PAPER_FACTS
# ============================================================================
WHITELIST_AUTHORS = {
    "cook", "reznikoff", "dauvois", "till", "fazenda", "jahn", "waller",
    "bishop", "saksala", "rubio",
    "huang", "tang", "tao", "murdock", "lahijanian", "yaghmazadeh",
    "buzsaki", "biswas", "tsai",
    "aparicio", "titterton", "diaz-andreu", "diaz_andreu",
}

PAPER_FACTS = {
    "huang_2026": {"ref": "Huang et al. 2026 PNAS vol.123 No.9", "doi": "10.1073/pnas.2513547123",
        "finding": "PAC theta-gamma MEG N=23, slow gamma 30-70 Hz retrieval hippocampique, fast gamma 70-140 Hz encodage entorhinal"},
    "tang_tao_2026": {"ref": "Tang & Tao 2026 Front.Aging Neurosci 17:1710041", "doi": "10.3389/fnagi.2025.1710041",
        "finding": "-51.8% Abeta cortex auditif, -47% pTau217, 40 Hz, souris 5xFAD"},
    "cook_2008": {"ref": "Cook et al. 2008 Time and Mind vol.1", "doi": "10.2752/175169708X374264",
        "finding": "110 Hz EEG chambres megalithiques modifient activite EEG regionale, Hal Saflieni"},
    "reznikoff_1988": {"ref": "Reznikoff & Dauvois 1988 Bull.Soc.Prehist.Fr",
        "finding": "correlation resonances grottes paleolithiques et art rupestre"},
    "jahn_1996": {"ref": "Jahn et al. 1996 JASA 99:649", "doi": "10.1121/1.414544",
        "finding": "resonances ~110 Hz cairns irlandais et dolmens gallois"},
    "bishop_1981": {"ref": "Bishop 1981 Tectonophysics 77:T17",
        "finding": "piezoelectricite mesuree dans granites quartzites gneiss naturels"},
    "saksala_2023": {"ref": "Saksala et al. 2023 Rock Mech Rock Eng",
        "finding": "affaiblissement 10% granite par excitation piezo quartz 274.4 kHz"},
    "aparicio_2025_fnhum": {"ref": "Aparicio-Torres et al. 2025 Front.HN 19:1574836", "doi": "10.3389/fnhum.2025.1574836",
        "finding": "PLV mesure directe entrainement cerebral, correle etats alteres conscience, EEG N=20"},
    "aparicio_2025_nyas": {"ref": "Aparicio-Torres et al. 2025 Annals NYAS", "doi": "10.1111/nyas.15403",
        "finding": "entrainement thalamo-cortical basses frequences = mecanisme neurobiologique etats alteres"},
    "titterton_2026": {"ref": "Titterton et al. 2026 Adv.Mat.Interfaces 13:e00552", "doi": "10.1002/admi.202500552",
        "finding": "activation piezo acoustique +46% synapses excitatrices +58% inhibitrices hNSC humaines"},
    "diaz_andreu_2025": {"ref": "Diaz-Andreu 2025 Annual Review Anthropology vol.54 pp.113-130", "doi": "10.1146/annurev-anthro-071323-113540",
        "finding": "revue systematique archeoacoustique 2025, neuroacoustique sites sacres prehistoriques"},
    "buzsaki_2026": {"ref": "Yaghmazadeh & Buzsaki 2026 Brain Stimulation 19:103032", "doi": "10.1016/j.brs.2026.103032",
        "finding": "TRFS RF 945 MHz, suppression interneurones PV thermique ≠ piezo acoustique"},
    "lahijanian_2024": {"ref": "Lahijanian et al. 2024 Nature Sci Rep",
        "finding": "40 Hz restaure connectivite DMN, mesure PLV"},
}

def parse_json_safe(raw: str) -> dict:
    """Parse JSON robuste — deux couches de protection."""
    if not raw:
        return {}
    try:
        raw2 = raw.strip()
        for marker in ["```json", "```JSON", "```"]:
            if raw2.startswith(marker):
                raw2 = raw2[len(marker):]
        raw2 = raw2.rstrip("`").strip()
        return _json.loads(raw2)
    except Exception:
        pass
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return _json.loads(raw[start:end])
    except Exception:
        pass
    return {}

# ============================================================================
# APPEL LLM — OpenRouter · Claude Sonnet 4.5 · temperature=0.0
# ============================================================================
def call_claude(api_key: str, system: str, user: str, max_tokens: int = 600) -> str:
    payload = {
        "model": "anthropic/claude-sonnet-4-5",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://barabar.app",
        "X-Title": "ArchéoAcoustique Multi-Sites v2.0",
    }
    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                             headers=headers, json=payload, timeout=90)
        if resp.status_code == 401:
            raise ValueError("Clé API invalide (401)")
        if resp.status_code == 429:
            raise ValueError("Limite de taux dépassée (429)")
        if not resp.ok:
            raise ValueError(f"Erreur HTTP {resp.status_code} : {resp.text[:200]}")
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Réponse OpenRouter inattendue : {str(data)[:200]}")
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise ValueError("Délai dépassé (90s)")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Erreur OpenRouter : {e}")

# ============================================================================
# PIPELINE MAICR 5 AGENTS — identique BARABAR v4.0
# ============================================================================
class WebProspector:
    SYSTEM = (
        "Tu es l'Agent WebProspector du consortium MAICR — spécialiste archéoacoustique multi-sites. "
        "REGLE ABSOLUE N1 : tu ne cites QUE les references de cette liste blanche. "
        "Si une reference n'est PAS dans cette liste, tu ecris auteur='NON_AUTORISEE'. "
        "LISTE BLANCHE (20 references autorisees) :\n"
        "=== ARCHÉOACOUSTIQUE ===\n"
        "1. Cook et al. 2008 (Time and Mind) — 110 Hz EEG megalithes, Hal Saflieni.\n"
        "2. Reznikoff & Dauvois 1988 — resonances grottes paleolithiques.\n"
        "3. Till & Fazenda 2012 (JASA) — acoustique Stonehenge.\n"
        "4. Jahn et al. 1996 (JASA 99:649) — resonances ~110 Hz cairns irlandais.\n"
        "5. Waller 1993 (World Archaeology) — diffraction acoustique sites rupestres.\n"
        "6. Diaz-Andreu 2025 (Annual Review Anthropology vol.54) — revue archeoacoustique 2025.\n"
        "=== PIEZOELECTRICITE GRANITE ===\n"
        "7. Bishop 1981 (Tectonophysics T17) — piezo granites/quartzites naturels.\n"
        "8. Saksala et al. 2023 (Rock Mech Rock Eng) — piezo quartz 274.4 kHz granite.\n"
        "9. Rubio Ruiz et al. 2024 (Rock Mech Rock Eng) — affaiblissement progressif granite.\n"
        "10. Titterton et al. 2026 (Adv.Mat.Interfaces) — piezo acoustique +46% synapses hNSC.\n"
        "=== NEUROSCIENCES 40 Hz / PAC ===\n"
        "11. Huang et al. 2026 PNAS vol.123 No.9 — PAC theta-gamma MEG N=23, slow gamma 30-70 Hz.\n"
        "12. Tang & Tao 2026 Front.Aging Neurosci 17:1710041 — 40 Hz -51.8% Abeta.\n"
        "13. Murdock et al. 2024 Nature — 40 Hz clearance glymphatique VIP/AQP4.\n"
        "14. Lahijanian et al. 2024 Nature Sci Rep — 40 Hz restaure DMN via PLV.\n"
        "15. Yaghmazadeh & Buzsaki 2026 Brain Stimulation 19:103032 — TRFS RF 945 MHz.\n"
        "16. Biswas et al. 2026 MIT Press — meditants gamma spontane 24-34 Hz.\n"
        "17. Tsai & Park 2025 PLOS Biology/MIT — GENUS 10 ans, 40 Hz Phase III.\n"
        "18. Aparicio-Torres 2025 Front.HN 19:1574836 — PLV correle etats alteres conscience.\n"
        "19. Aparicio-Torres 2025 Annals NYAS — thalamo-cortical basses freq etats alteres.\n"
        "REGLE ABSOLUE N2 : toute reference hors liste = auteur='NON_AUTORISEE'.\n"
        "REGLE ABSOLUE N3 : ne jamais inventer un resultat numerique ni un titre.\n"
        "Reponds UNIQUEMENT en JSON valide, sans markdown."
    )

    def prospecter(self, api_key: str, site_name: str, freq: float, plv: float,
                   Ic: float, decision: str, dominant: str) -> dict:
        band = "slow gamma Huang [30-70 Hz]" if 30<=freq<=70 else ("fast gamma Huang [70-140 Hz]" if 70<freq<=140 else "hors bandes gamma")
        user = (
            f"Site : {site_name}\n"
            f"Frequence : {freq:.1f} Hz ({band})\n"
            f"PLV : {plv:.4f}\n"
            f"Ic v2.1 : {Ic:.4f} — {decision}\n"
            f"Voie dominante : {dominant}\n\n"
            "Choisis MAX 2 references de la liste blanche pertinentes pour ce site.\n"
            '{"references_cles":[{"auteur":"NOM_EXACT","annee":2025,"titre":"..."}],'
            '"coherence_parametres":"...","support_litteraire":"FORT|PARTIEL|ABSENT",'
            '"limites_connaissance":"..."}'
        )
        raw = call_claude(api_key, self.SYSTEM, user, 400)
        result = parse_json_safe(raw)
        # Post-parsing whitelist
        refs = result.get("references_cles", [])
        valid, hallucinated = [], []
        for ref in refs if isinstance(refs, list) else []:
            auteur = str(ref.get("auteur","")).lower()
            if any(w in auteur for w in WHITELIST_AUTHORS):
                valid.append(ref)
            else:
                hallucinated.append(auteur)
        result["references_cles"] = valid
        if hallucinated:
            result["limites_connaissance"] = (
                result.get("limites_connaissance","") +
                f" | Refs supprimées (hors whitelist) : {', '.join(hallucinated)}"
            ).strip(" |")
        return result


class ControleurJuge:
    SYSTEM = (
        "Tu es le Contrôleur-Juge MAICR pour l'archéoacoustique multi-sites. "
        "Génère un débat court entre :\n"
        "Sceptique : cite les failles méthodologiques. "
        "Il cite Buzsaki 2026 (TRFS thermique ≠ piezo acoustique) et questionne "
        "si le champ piézo granite atteint le seuil neuronal. "
        "Il distingue rigueur d'un site principal (Barabar) vs extrapolation multi-sites.\n"
        "Enthousiaste : cite UNIQUEMENT ces références vérifiées : "
        "Huang 2026 PNAS (PAC theta-gamma slow gamma 30-70 Hz), "
        "Cook 2008 (EEG 110 Hz Hal Saflieni), "
        "Reznikoff 1988 (resonances grottes paleolithiques), "
        "Aparicio-Torres 2025 Frontiers HN (PLV = entrainement cerebral etats alteres), "
        "Aparicio-Torres 2025 NYAS (thalamo-cortical basses freq), "
        "Titterton 2026 (piezo acoustique +46% synapses hNSC), "
        "Diaz-Andreu 2025 Annual Review Anthropology.\n"
        "Consensus MAICR : objectif, distingue voie acoustique vs piézo, niveau publication.\n"
        "MAX 120 chars par champ. Phrases complètes.\n\n"
        "=== DÉFINITIONS PROTOCOLAIRES STRICTES ===\n"
        "H₀ = ÉCHEC mécanisme EM. RÉSULTAT NÉGATIF. Jamais valoriser.\n"
        "H₂ = BIFURCATION ACOUSTIQUE. Son seul actif sans mediation EM.\n"
        "H₁ = SUCCÈS mécanisme EM. ODF>3, B>1nT, SNR>20dB requis.\n"
        "RÈGLES ANTI-HALLUCINATION :\n"
        "  1. Si H₀ ET PLV>0.5 ET ODF>3 : signaler paradoxe dans consensus.\n"
        "  2. H₀ ne jamais valorisé positivement.\n"
        "  3. Consensus nomme la décision Ic telle quelle.\n"
    )

    def deliberer(self, api_key: str, site_name: str, freq: float,
                  plv: float, Ic: float, decision: str, C: float,
                  T: float, G: float, PAC: float, dominant: str,
                  refs: list) -> dict:
        user = (
            f"Site : {site_name} · Freq={freq:.1f}Hz\n"
            f"PLV={plv:.4f}, Ic={Ic:.4f}, decision={decision}\n"
            f"C={C:.4f}, T={T:.4f}, G={G:.4f}, PAC={PAC:.4f}\n"
            f"Dominant: {dominant}\n"
            f"Refs WebProspector : {[r.get('auteur','?') for r in refs]}\n\n"
            '{"physicien_sceptique":"...","neuro_acousticien_enthousiaste":"...","consensus_maicr":"..."}'
        )
        return parse_json_safe(call_claude(api_key, self.SYSTEM, user, 600))


class ExpertScribe:
    SYSTEM = (
        "Tu es l'Expert Scribe MAICR. "
        "Rédige un résumé exécutif pour un rapport d'archéoacoustique multi-sites niveau CNRS. "
        "Contexte : algorithme Ic v2.1 (4 scores C·T·G·PAC, poids 0.35/0.30/0.15/0.20), "
        "dual pathway (voie acoustique directe vs voie piézo-électrique), bifurcation H2. "
        "Convergences 2025-2026 : Aparicio-Torres 2025 (PLV → etats alteres), "
        "Titterton 2026 (piezo acoustique → synapses hNSC), "
        "Diaz-Andreu 2025 (Annual Review archéoacoustique), "
        "Huang 2026 PNAS (PAC theta-gamma slow gamma 30-70 Hz). "
        "Style : académique, factuel, sans émotion, sans spéculer au-delà des données. Max 180 mots. "
        "Reponds UNIQUEMENT en JSON : {\"resume_executif\":\"...\"}"
    )

    def rediger(self, api_key: str, site_name: str, freq: float,
                Ic: float, decision: str, plv: float, PAC: float,
                dominant: str, consensus: str) -> dict:
        user = (
            f"Site : {site_name}\n"
            f"Frequence : {freq:.1f} Hz · Ic v2.1 : {Ic:.4f} — {decision}\n"
            f"PLV : {plv:.4f} · PAC Huang 2026 : {PAC:.4f}\n"
            f"Voie dominante : {dominant}\n"
            f"Consensus MAICR : {consensus[:250]}\n\n"
            '{"resume_executif":"..."}'
        )
        return parse_json_safe(call_claude(api_key, self.SYSTEM, user, 400))


prospector = WebProspector()
juge       = ControleurJuge()
scribe     = ExpertScribe()

def main():
    inject_css()
    setup_mpl_theme()

    if 'selected_site' not in st.session_state:
        st.session_state.selected_site = 'barabar'

    neural = NeuralSim(N=3000)
    algo   = IcAlgorithm()
    # PhysicsEngine est recréé par run_simulation() selon le matériau du site

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("ArcheoAcoustique")
        st.caption("v2.0 · Ic v2.1 · MAICR 5 Agents")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Simulation par Site", "Comparaison Multi-Sites", "Documentation"],
        )

        st.markdown("---")
        st.markdown("**Cle API OpenRouter**")
        api_key = st.text_input(
            "Cle API",
            placeholder="sk-or-v1-...",
            type="password",
            label_visibility="collapsed",
            help="Pour activer le pipeline MAICR 5 agents"
        )

        st.markdown("---")
        st.markdown("""
**Chaine causale Ic v2.1**

Acoustique → Piezo (T·G) → Os → Kuramoto → PLV → PAC Huang 2026 → Dual Pathway → Ic → MAICR
        """)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — SIMULATION PAR SITE
    # ════════════════════════════════════════════════════════════════════════
    if "Simulation par Site" in page:

        # Header
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(58,126,244,0.08) 0%,rgba(107,78,247,0.06) 100%);
                    border:1px solid rgba(99,157,255,0.1);border-radius:20px;
                    padding:2rem 2.5rem;margin-bottom:2rem;">
            <div style="font-size:0.72rem;font-weight:600;color:#00d48a;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">
                Moteur Kuramoto N=3 000
            </div>
            <h1 style="margin:0;font-size:1.9rem;font-weight:700;
                       letter-spacing:-0.03em;color:#dde3f0;line-height:1.2;">
                🏛️ ArchéoAcoustique
            </h1>
            <p style="margin:0.4rem 0 0;color:#5a7090;font-size:0.85rem;">
                Moteur de Simulation Multi-Sites · Analyse acoustique de sites archéologiques
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Sélecteur de site ──
        site = SITES_DB[st.session_state.selected_site]

        st.markdown(section_header("3", "Paramètres & Simulation",
                    "Preset mondial ou paramètres libres"), unsafe_allow_html=True)

        # ── Sélecteur de site mondial ──────────────────────────────────────
        preset_names = list(WORLD_PRESETS.keys())
        # Trouver le preset correspondant au site sélectionné
        default_preset_idx = 0
        if site.id == "barabar": default_preset_idx = 0
        elif site.id == "saflieni": default_preset_idx = 1
        elif site.id == "pyramide": default_preset_idx = 2
        elif site.id == "megalithes": default_preset_idx = 3

        selected_preset = st.selectbox(
            "Site mondial (preset)",
            preset_names,
            index=default_preset_idx,
            key=f"preset_{site.id}",
            help="Charge les parametres documentes du site"
        )
        preset = WORLD_PRESETS[selected_preset]
        # Synchroniser selected_site avec le preset choisi
        _preset_to_site = {
            "Barabar - Chambre Sudama (Inde)": "barabar",
            "Hal Saflieni - Oracle Room (Malte)": "saflieni",
            "Grande Pyramide - Chambre du Roi (Egypte)": "pyramide",
            "Newgrange - Chambre interieure (Irlande)": "megalithes",
        }
        if selected_preset in _preset_to_site:
            _mapped = _preset_to_site[selected_preset]
            if _mapped != st.session_state.selected_site:
                st.session_state.selected_site = _mapped
                for k in ["last_results", "last_params"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
        # Mettre a jour site apres eventuel rerun
        site = SITES_DB[st.session_state.selected_site]
        if preset.get("note"):
            st.caption(preset["note"])

        # Donnees documentees du site (remplace l'ancienne section 2)
        with st.expander("Donnees acoustiques documentees — " + site.name, expanded=True):
            st.caption(site.description)
            if site.chamber_resonance:
                _, source = site.get_resonance_badge()
                st.markdown(f"**Resonance chambre** : `{site.chamber_resonance:.1f} Hz` ✅ — *{source}*")
            if site.documented_frequency:
                _, source = site.get_frequency_badge()
                st.markdown(f"**Frequence documentee** : `{site.documented_frequency:.1f} Hz` ✅ — *{source}*")
            if site.sarcophagus_resonance:
                st.markdown(f"**Resonance sarcophage** : `{site.sarcophagus_resonance:.1f} Hz` ✅")
            if site.infrasound:
                st.markdown(f"**Infrasons** : `{site.infrasound:.1f} Hz` — non simule (freq sim = chambre)")
            if site.frequency_range:
                st.markdown(f"**Plage** : `{site.frequency_range[0]:.0f} – {site.frequency_range[1]:.0f} Hz` ✅")
            if site.beat_frequency:
                st.markdown(f"**Battement genere** : `{site.beat_frequency:.1f} Hz` — calcule")
            st.markdown(f"**Materiau** : {site.material or 'N/A'} — ODF = {site.odf or 'N/A'}")
            # Sources
            if site.sources:
                st.markdown("**Sources :**")
                for s in site.sources[:3]:
                    st.markdown(f"- {s['author']} ({s['year']}) — {s['title'][:60]}")

        st.markdown("---")

        # Matériau personnalisé
        mat_options = {"granite": "🪨 Granite (f_qz=0.32)", "limestone": "🏛️ Calcaire (f_qz=0.02)",
                       "sandstone": "🏜️ Grès (f_qz=0.18)", "mixed": "🌀 Mixte (f_qz=0.15)"}
        mat_keys = list(mat_options.keys())
        preset_mat = preset.get("material_type", site.material_type or "granite")
        mat_idx = mat_keys.index(preset_mat) if preset_mat in mat_keys else 0
        selected_mat = st.selectbox(
            "Matériau dominant",
            mat_keys,
            index=mat_idx,
            format_func=lambda x: mat_options[x],
            key=f"mat_{site.id}_{selected_preset}"
        )

        # Nom de site libre
        site_name_custom = st.text_input(
            "Nom du site (libre)",
            value=selected_preset.split("·")[0].strip().replace("🏔️","").replace("🇲🇹","")
                  .replace("🇪🇬","").replace("🇮🇪","").replace("🇫🇷","").replace("🇲🇽","")
                  .replace("🏴󠁧󠁢󠁥󠁮󠁧󠁿","").replace("🇵🇪","").replace("🌍","").strip(),
            key=f"sitename_{site.id}_{selected_preset}",
            placeholder="Ex: Grotte de Font-de-Gaume"
        )

        st.markdown("---")

        # Valeurs par défaut depuis le preset ou le site
        default_freq = preset.get("freq", site.chamber_resonance or site.average_frequency or 110.0)
        odf_default  = preset.get("odf", site.odf or 3.5)
        default_press = preset.get("press_db", 102)
        default_Q    = preset.get("Q", 500)
        default_noise= preset.get("noise", 1.5)

        col1, col2 = st.columns(2)
        with col1:
            freq = st.slider("Fréquence (Hz)", 20.0, 800.0, float(default_freq), 0.5,
                             key=f"freq_{site.id}", help="Fréquence de résonance principale du site")
            odf = st.slider("ODF — Orientation cristalline", 1.0, 5.0, float(odf_default), 0.1,
                            key=f"odf_{site.id}_{selected_preset}", help="1=aléatoire (piézo nul) · 5=parfait · Barabar≈4.2")
        with col2:
            noise = st.slider("Bruit synaptique (mV)", 0.1, 3.0, float(default_noise), 0.1,
                              key=f"noise_{site.id}_{selected_preset}")
            press_db = st.slider("Pression (dBSPL)", 60, 130, int(default_press), 1,
                                 key=f"press_{site.id}_{selected_preset}",
                                 help="102 dBSPL = voix forte Barabar · 85 = voix normale")

        Q = st.slider("Facteur Q", 20, 1000, int(default_Q), 10, key=f"Q_{site.id}_{selected_preset}",
                     help="Facteur qualité acoustique · Barabar≈950 · Newgrange≈400")

        run = st.button("▶ Lancer la Simulation", type="primary", use_container_width=True)

        if run:
            try:
                validate_inputs(freq, press_db, odf, Q, noise)
            except InputValidationError as e:
                st.error(str(e))
                st.stop()
            with st.spinner("Simulation du réseau neuronal en cours…"):
                params = {"freq": freq, "odf": odf, "noise": noise,
                          "press_db": press_db, "Q": Q,
                          "material_type": selected_mat,
                          "site_name_custom": site_name_custom}
                res = run_simulation(params, neural, algo)
                st.session_state.last_results = res
                st.session_state.last_params = params
                st.session_state.last_site_id = site.id
                st.session_state.last_site_name = site.name
        elif 'last_results' not in st.session_state:
            st.markdown("""
            <div style="text-align:center;padding:2rem 1rem;
                        background:rgba(255,255,255,0.015);
                        border:1px dashed rgba(255,255,255,0.08);
                        border-radius:16px;color:#3a5a80;">
                <div style="font-size:2rem;margin-bottom:0.4rem;opacity:0.4;">◉</div>
                <div style="font-size:0.85rem;font-weight:500;">
                    Ajustez les paramètres puis cliquez sur
                    <strong style="color:#639dff;">▶ Lancer la simulation</strong>
                </div>
                <div style="font-size:0.75rem;margin-top:0.3rem;opacity:0.6;">
                    Réseau de Kuramoto N=3 000 · ~5–10 s
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Résultats ── (pleine largeur sous les colonnes)
        if 'last_results' in st.session_state:
            # Vérifier que les résultats appartiennent au site actuel
            if st.session_state.get('last_site_id') != site.id:
                st.markdown("""
                <div style="text-align:center;padding:2rem;
                            background:rgba(255,165,0,0.06);
                            border:1px solid rgba(255,165,0,0.2);
                            border-radius:16px;color:#ffa726;">
                    <div style="font-size:1.5rem;margin-bottom:0.4rem;">⚠️</div>
                    <div style="font-size:0.9rem;font-weight:500;">
                        Les derniers résultats sont pour un autre site.<br>
                        <strong>Lancez la simulation pour ce site.</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                res = st.session_state.last_results
                params = st.session_state.last_params
                st.markdown("---")

                # Badge site des résultats
                st.markdown(f"""
                <div style="display:inline-block;padding:0.3rem 0.9rem;margin-bottom:1rem;
                            background:rgba(0,212,136,0.1);border:1px solid rgba(0,212,136,0.3);
                            border-radius:20px;font-size:0.75rem;color:#00d48a;
                            font-family:'JetBrains Mono',monospace;">
                    ◉ Résultats · {st.session_state.get('last_site_name', site.name)}
                </div>
                """, unsafe_allow_html=True)

                # Section physique
                st.markdown(section_header("4", "Conversion Physique",
                            f"P = {res['P_pa']*1e3:.2f} mPa · δ = {res['delta']*1e3:.1f} mm"),
                            unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(phys_card(
                        "Champ Électromagnétique", f"{res['E_gran']:.2f}", "µV/m",
                        "Induction B", f"{res['B_gran']*1e15:.2f}", "fT",
                        "#4f8ef7", "linear-gradient(180deg,#3a7ef4,#1a56c4)"
                    ), unsafe_allow_html=True)
                with c2:
                    st.markdown(phys_card(
                        "Transmission Osseuse", f"{res['T_skull']*100:.1f}", "%",
                        "Champ cortex E", f"{res['E_cortex']:.3f}", "µV/m",
                        "#a78bfa", "linear-gradient(180deg,#9b59b6,#6b4ef7)"
                    ), unsafe_allow_html=True)
                with c3:
                    st.markdown(phys_card(
                        "Focalisation Elliptique", f"×{res['e_gain']:.1f}", "",
                        "Champ foyer E", f"{res['E_foyer']:.2f}", "µV/m",
                        "#fb923c", "linear-gradient(180deg,#ff7043,#e91e63)"
                    ), unsafe_allow_html=True)

                st.markdown("---")

                # Section neuronale
                st.markdown(section_header("5", "Dynamique Neuronale",
                            "Réseau de Kuramoto · N = 3 000"), unsafe_allow_html=True)

                left, right = st.columns([2, 1])
                with left:
                    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={'wspace': 0.35})
                    ax = axes[0]
                    ax.plot(res['n_arr'], res['snr_arr'], 'o-', color='#4f8ef7',
                            linewidth=2.2, markersize=6, markerfacecolor='#0f1b2d',
                            markeredgewidth=1.8, zorder=3)
                    ax.fill_between(res['n_arr'], res['snr_arr'], alpha=0.12, color='#4f8ef7')
                    ax.axvline(res['n_opt'], color='#00d48a', linestyle='--',
                               linewidth=1.5, alpha=0.9, label=f'Optimal {res["n_opt"]:.2f} mV')
                    ax.axvline(params['noise'], color='#ff6b6b', linestyle=':',
                               linewidth=1.5, alpha=0.8, label=f'Sélection {params["noise"]:.2f} mV')
                    ax.set_xlabel('Bruit synaptique (mV)')
                    ax.set_ylabel('SNR')
                    ax.set_title('Résonance Stochastique')
                    ax.legend(frameon=True)

                    ax2 = axes[1]
                    counts, bins, patches = ax2.hist(res['theta'][-400:].flatten(), bins=36,
                                                      density=True, edgecolor='none', alpha=0.0)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    for patch, cv in zip(patches, bin_centers / (2 * np.pi)):
                        patch.set_facecolor(plt.cm.plasma(cv))
                        patch.set_alpha(0.85)
                    ax2.set_xlabel('Phase (rad)')
                    ax2.set_ylabel('Densité')
                    ax2.set_title(f'Distribution des phases · PLV = {res["plv"]:.3f}')
                    ax2.set_xlim(0, 2 * np.pi)
                    ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                    ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with right:
                    st.metric("PLV Synchronisation", f"{res['plv']:.3f}")
                    st.metric("Bruit optimal", f"{res['n_opt']:.2f} mV")
                    st.metric("Gain SR max", f"{res['snr_max']:.2f} ×")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if res['plv'] > 0.5:
                        st.success("**Synchronisation forte**")
                    elif res['plv'] > 0.3:
                        st.info("**Synchronisation modérée**")
                    else:
                        st.warning("**Synchronisation faible**")

                st.markdown("---")

                # Section Ic
                st.markdown(section_header("6", "Algorithme Ic v2.1",
                            "Fusion bayésienne · 4 sous-scores C·T·G·PAC · Dual Pathway"), unsafe_allow_html=True)

                col_chart, col_metrics, col_ic = st.columns([2, 1, 1])

                with col_chart:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    cats = ['C\nCohérence', 'T\nPiézo', 'G\nGéo', 'PAC\nHuang2026']
                    vals = [res['C'], res['T'], res['G'], res['PAC']]
                    palette = ['#4f8ef7', '#a78bfa', '#00d48a', '#ffa726']
                    bars = ax.bar(cats, vals, color=palette, alpha=0.85, width=0.55,
                                  edgecolor='none', zorder=3)
                    ax.axhline(y=0.65, color='#00d48a', linestyle='--', linewidth=1.2,
                               alpha=0.6, zorder=1)
                    ax.axhline(y=0.35, color='#ff6b6b', linestyle='--', linewidth=1.2,
                               alpha=0.6, zorder=1)
                    ax.text(3.45, 0.66, 'H₁', fontsize=7.5, color='#00d48a', alpha=0.7, va='bottom')
                    ax.text(3.45, 0.36, 'H₀', fontsize=7.5, color='#ff6b6b', alpha=0.7, va='bottom')
                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                                fontweight='700', color='#dde3f0')
                    if res['conflict']:
                        ax.text(0.5, 0.5, '⚠ CONFLIT DÉTECTÉ', transform=ax.transAxes,
                                ha='center', va='center', fontsize=13, color='#ffa726',
                                alpha=0.4, fontweight='700', rotation=12)
                    ax.set_ylim(0, 1.12)
                    ax.set_ylabel('Score normalisé')
                    ax.set_title("Ic v2.1 — 4 scores · Dual Pathway")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with col_metrics:
                    st.markdown(metrics_table([
                        ("C Cohérence spectrale", f"{res['C']:.4f}", "×0.35"),
                        ("T Transfert piézo", f"{res['T']:.4f}", "×0.30"),
                        ("G Facteur géologique", f"{res['G']:.4f}", "×0.15"),
                        ("PAC Huang 2026", f"{res['PAC']:.4f}", "×0.20"),
                        ("Acoustique directe", f"{res['acoustic_direct']:.4f}", "C×0.6+PAC×0.4"),
                        ("Voie piézo", f"{res['piezo_pathway']:.4f}", "T×0.7+G×0.3"),
                        ("Dominant", res['dominant'], ""),
                        ("σ Dispersion", f"{res['std_s']:.4f}",
                         "⚠ conflit" if res['conflict'] else "✓ cohérent"),
                        ("Matériau", res.get("material_type", site.material_type or "granite"),
                         f"f_qz={res.get('f_qz_used', 0.32):.2f} — {'⚠ calcaire' if res.get('f_qz_used',0.32) < 0.05 else '✓ granite'}"),
                    ]), unsafe_allow_html=True)

                with col_ic:
                    st.markdown(decision_box(res['Ic'], res['decision'],
                                             res['color'], res['bg']),
                                unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(float(res['Ic']))
                    st.markdown(f"""
                    <div style="margin-top:1rem;font-size:0.72rem;color:#4a5568;
                                font-family:'JetBrains Mono',monospace;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                            <span>C</span><span>T</span><span>G</span><span>PAC</span>
                        </div>
                        <div style="display:flex;gap:2px;height:6px;border-radius:4px;overflow:hidden;">
                            <div style="flex:{res['C']:.2f};background:#4f8ef7;border-radius:4px 0 0 4px;"></div>
                            <div style="flex:{res['T']:.2f};background:#a78bfa;"></div>
                            <div style="flex:{res['G']:.2f};background:#00d48a;"></div>
                            <div style="flex:{res['PAC']:.2f};background:#ffa726;border-radius:0 4px 4px 0;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── PIPELINE MAICR 5 AGENTS ────────────────────────────────
                st.markdown("---")
                st.markdown(section_header("7", "Pipeline MAICR 5 Agents",
                            "WebProspector · Juge · Scribe · temperature=0.0 · Whitelist 20 Refs"),
                            unsafe_allow_html=True)

                if not api_key:
                    st.info("🔑 Entrez votre clé API OpenRouter dans la sidebar pour activer le pipeline MAICR.")
                else:
                    run_maicr = st.button("🧠 Lancer le pipeline MAICR", type="primary", use_container_width=True)
                    if run_maicr:
                        # Utiliser le site associé aux derniers résultats
                        # (pas selected_site qui peut avoir changé entre-temps)
                        maicr_site_id = st.session_state.get("last_site_id",
                                            st.session_state.selected_site)
                        site = SITES_DB.get(maicr_site_id, SITES_DB["barabar"])
                        freq_sim = params["freq"]

                        # Coherence check — odf_used déjà dans res depuis run_simulation
                        flags = check_coherence(res, freq_sim)
                        if flags:
                            st.warning("⚠️ **Alertes cohérence Python (non-LLM) :**\n" + "\n".join(f"  • {f}" for f in flags))

                        try:
                            # Agent WebProspector
                            with st.spinner("📚 Agent WebProspector — triangulation bibliographique…"):
                                web_res = prospector.prospecter(
                                    api_key, site.name, freq_sim, res['plv'],
                                    res['Ic'], res['decision'], res['dominant']
                                )
                            st.markdown("**📚 Agent WebProspector**")
                            refs_list = web_res.get("references_cles", [])
                            if refs_list:
                                for ref in refs_list:
                                    st.markdown(f"  • **{ref.get('auteur','')} ({ref.get('annee','')})** — {ref.get('titre','')}")
                            support = web_res.get("support_litteraire", "—")
                            st.markdown(f"Support littéraire : **{support}**")
                            if web_res.get("limites_connaissance"):
                                st.caption(f"ℹ️ {web_res['limites_connaissance']}")

                            # Agent Juge
                            with st.spinner("⚖️ Agent Juge — débat MAICR…"):
                                juge_res = juge.deliberer(
                                    api_key, site.name, freq_sim, res['plv'],
                                    res['Ic'], res['decision'],
                                    res['C'], res['T'], res['G'], res['PAC'],
                                    res['dominant'], refs_list
                                )
                            col_s, col_e = st.columns(2)
                            with col_s:
                                st.markdown("**🔴 Sceptique**")
                                st.markdown(juge_res.get("physicien_sceptique", "—"))
                            with col_e:
                                st.markdown("**🟢 Enthousiaste**")
                                st.markdown(juge_res.get("neuro_acousticien_enthousiaste", "—"))
                            st.markdown("**🔵 Consensus MAICR**")
                            st.success(juge_res.get("consensus_maicr", "—"))

                            # Agent Scribe
                            with st.spinner("📄 Agent Scribe — résumé exécutif…"):
                                scribe_res = scribe.rediger(
                                    api_key, site.name, freq_sim, res['Ic'],
                                    res['decision'], res['plv'], res['PAC'],
                                    res['dominant'], juge_res.get("consensus_maicr","")
                                )
                            st.markdown("**📄 Résumé exécutif (style CNRS)**")
                            st.info(scribe_res.get("resume_executif", "—"))

                        except ValueError as e:
                            st.error(f"Erreur pipeline MAICR : {e}")

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — COMPARAISON MULTI-SITES
    # ════════════════════════════════════════════════════════════════════════
    elif "Comparaison" in page:

        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(58,126,244,0.08) 0%,rgba(107,78,247,0.06) 100%);
                    border:1px solid rgba(99,157,255,0.1);border-radius:20px;
                    padding:2rem 2.5rem;margin-bottom:2rem;">
            <h1 style="margin:0;font-size:1.9rem;font-weight:700;
                       letter-spacing:-0.03em;color:#dde3f0;">
                🔬 Comparaison Multi-Sites
            </h1>
            <p style="margin:0.4rem 0 0;color:#5a7090;font-size:0.85rem;">
                Simulation parallèle · 4 sites · Analyse comparative Ic
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(section_header("1", "Paramètres communs"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: noise_c = st.slider("Bruit synaptique (mV)", 0.1, 3.0, 1.5, 0.1)
        with col2: press_db_c = st.slider("Pression (dBSPL)", 80, 120, 102, 1)
        with col3: Q_c = st.slider("Facteur Q", 20, 1000, 500, 10)

        run_comp = st.button("▶ Lancer la Comparaison", type="primary", use_container_width=True)

        if run_comp:
            # Vérification avec paramètres représentatifs (Saflieni odf=2.5 → Q minimal)
            first_site = list(SITES_DB.values())[0]
            try:
                validate_inputs(
                    first_site.chamber_resonance or first_site.average_frequency or 110.0,
                    press_db_c, 2.0, Q_c, noise_c
                )
            except InputValidationError as e:
                st.error(str(e))
                st.stop()
            all_res = {}
            prog = st.progress(0, text="Simulation en cours…")
            for i, (sid, site) in enumerate(SITES_DB.items()):
                prog.progress(i / len(SITES_DB), text=f"Calcul {site.name}…")
                f = site.chamber_resonance or site.average_frequency or 110.0
                p = {"freq": f, "odf": site.odf or 3.0,
                     "noise": noise_c, "press_db": press_db_c, "Q": Q_c,
                     "material_type": site.material_type or "granite"}
                all_res[sid] = run_simulation(p, neural, algo)
            prog.progress(1.0, text="Terminé ✓")
            prog.empty()

            st.markdown("---")
            st.markdown(section_header("2", "Résultats comparatifs"), unsafe_allow_html=True)

            labels = [SITES_DB[s].name for s in all_res]
            metrics_names = ["Cohérence C", "Transfert T", "Géo G", "PAC", "PLV", "Ic"]
            data_matrix = np.array([[r["C"], r["T"], r["G"], r["PAC"], r["plv"], r["Ic"]]
                                     for r in all_res.values()])
            colors_bar = ['#4f8ef7', '#a78bfa', '#00d48a', '#ffa726']

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            x = np.arange(len(metrics_names))
            w = 0.18
            for j, (label, row, col_) in enumerate(zip(labels, data_matrix, colors_bar)):
                ax.bar(x + j * w - 1.5 * w, row, w, label=label, color=col_, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, fontsize=8.5)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score normalisé")
            ax.set_title("Scores comparatifs par site")
            ax.legend(fontsize=8)
            ax.axhline(0.65, color='#00d48a', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axhline(0.35, color='#ff4d4d', linestyle='--', linewidth=0.8, alpha=0.5)

            ax2 = axes[1]
            ic_vals = [r["Ic"] for r in all_res.values()]
            decisions = [r["decision"] for r in all_res.values()]
            bar_colors2 = [r["color"] for r in all_res.values()]
            bars2 = ax2.barh(labels, ic_vals, color=bar_colors2, alpha=0.85, height=0.5)
            ax2.axvline(0.65, color='#00d48a', linestyle='--', linewidth=1, alpha=0.6)
            ax2.axvline(0.35, color='#ff4d4d', linestyle='--', linewidth=1, alpha=0.6)  # seuil H₀ = 0.35
            for bar, val, dec in zip(bars2, ic_vals, decisions):
                ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f} — {dec}", va='center', fontsize=8.5, color='#dde3f0')
            ax2.set_xlim(0, 1.25)
            ax2.set_xlabel("Indice Ic")
            ax2.set_title("Indice de Cohérence par Site")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            st.markdown(section_header("3", "Récapitulatif"), unsafe_allow_html=True)
            for sid, r in all_res.items():
                s = SITES_DB[sid]
                ca, cb, cc, cd, ce, cf = st.columns(6)
                ca.metric("Site", s.name)
                cb.metric("Ic v2.1", f"{r['Ic']:.3f}")
                cc.metric("PLV", f"{r['plv']:.3f}")
                cd.metric("PAC", f"{r['PAC']:.3f}")
                ce.metric("Décision", r["decision"])
                cf.metric("Dominant", r.get("dominant", "—"))

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 2rem;
                        background:rgba(255,255,255,0.012);
                        border:1px dashed rgba(99,157,255,0.15);
                        border-radius:16px;">
                <div style="font-size:2.5rem;margin-bottom:0.6rem;opacity:0.3;">◉</div>
                <div style="font-size:0.9rem;font-weight:500;color:#4a6a90;">
                    Cliquez sur <strong style="color:#639dff;">▶ Lancer la Comparaison</strong>
                </div>
                <div style="font-size:0.78rem;margin-top:0.4rem;color:#3a5070;">
                    4 simulations Kuramoto N=3 000 · ~20–30 s
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — DOCUMENTATION MAICR COMPLÈTE
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(58,126,244,0.08) 0%,rgba(107,78,247,0.06) 100%);
                    border:1px solid rgba(99,157,255,0.1);border-radius:20px;
                    padding:2rem 2.5rem;margin-bottom:2rem;">
            <div style="font-size:0.72rem;font-weight:600;color:#00d48a;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">
                BARABAR v4.0 · Ic v2.1 · MAICR 5 Agents · Whitelist 20 Refs
            </div>
            <h1 style="margin:0;font-size:1.9rem;font-weight:700;
                       letter-spacing:-0.03em;color:#dde3f0;">
                📚 Référentiel Scientifique MAICR
            </h1>
            <p style="margin:0.4rem 0 0;color:#5a7090;font-size:0.85rem;">
                Sources documentées par site · Pipeline MAICR · Algorithme Ic v2.1 · Samuel Chapelais Veillet
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Section Sources par site ──────────────────────────────────────
        st.markdown(section_header("1", "Sources archéologiques par site"), unsafe_allow_html=True)
        for site_id, site in SITES_DB.items():
            with st.expander(f"🏛️ {site.name} — {site.location}"):
                st.markdown(f"""
                <div style="background:rgba(58,126,244,0.06);border:1px solid rgba(58,126,244,0.15);
                            border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;
                            font-size:0.83rem;color:#8da0c0;line-height:1.6;">
                    {site.description}
                </div>
                """, unsafe_allow_html=True)
                for source in site.sources:
                    inst = source.get('institution', '')
                    inst_str = f" — {inst}" if inst else ""
                    st.markdown(f"""
                    <div style="background:rgba(99,157,255,0.05);border-left:3px solid #639dff;
                                border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin:0.5rem 0;">
                        <div style="font-size:0.8rem;font-weight:600;color:#639dff;">
                            {source['author']} ({source['year']}){inst_str}
                        </div>
                        <div style="font-size:0.8rem;color:#a0b4c8;margin-top:0.2rem;">
                            {source['title']}
                        </div>
                        <div style="font-size:0.75rem;color:#5a7090;margin-top:0.2rem;">
                            {source['type']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Whitelist 20 Références annotées ─────────────────────────────
        st.markdown(section_header("2", "Whitelist MAICR — 20 références peer-reviewed",
                    "Publications autorisées · Post-parsing anti-hallucination"), unsafe_allow_html=True)

        REFS_DOC = [
            ("ARCHÉOACOUSTIQUE", [
                ("[01] Reznikoff & Dauvois (1988)", "Bull. Soc. Préhistorique Française",
                 "Première démonstration systématique de la corrélation entre qualité acoustique et densité de peintures rupestres paléolithiques dans les grottes de Font-de-Gaume, Niaux, Portel, Pech-Merle.",
                 "Pertinence BAM : fondateur du champ — sites sacrés délibérément choisis pour leurs propriétés acoustiques. Analogie directe avec Barabar.", "FONDATEUR", None),
                ("[02] Cook, Pajot & Leuchter (2008)", "Time and Mind, vol.1 No.1 · DOI: 10.2752/175169708X374264",
                 "Les fréquences de résonance des chambres mégalithiques (110 Hz, Ħal Saflieni) modifient l'activité EEG régionale. Basculement préfrontal vers dominance droite à 110 Hz (N=30, UCLA).",
                 "Pertinence BAM : protocole EEG + résonance chambre souterraine = protocole exact Phase 1 Bihar. Référence centrale.", "FONDATEUR", "10.2752/175169708X374264"),
                ("[03] Jahn et al. (1996)", "JASA 99:649 · DOI: 10.1121/1.414544",
                 "Premières mesures acoustiques systématiques dans des sites mégalithiques européens (cairns irlandais, dolmens gallois). Résonances ~110 Hz dans toutes les chambres testées.",
                 "Pertinence BAM : convergence 110 Hz cross-culturelle. Contexte comparatif pour Barabar (34.4 Hz = granite indien).", "FONDATEUR", "10.1121/1.414544"),
                ("[04] Till & Fazenda (2012)", "JASA — Acoustique de Stonehenge",
                 "Modélisation acoustique de Stonehenge par éléments finis (réplique 1/12). Filtrage acoustique complexe et réverbération intentionnels.",
                 "Pertinence BAM : valide l'approche modélisation computationnelle de l'acoustique des sites anciens.", None, None),
                ("[05] Waller (1993)", "World Archaeology 25:117-128",
                 "Hypothèse de diffraction acoustique : sites rupestres localisés aux points d'interférence de sources naturelles.",
                 "Pertinence BAM : argument alternatif utilisé par le Sceptique dans le débat MAICR.", None, None),
                ("[06] Díaz-Andreu (2025)", "Annual Review of Anthropology vol.54 pp.113-130 · DOI: 10.1146/annurev-anthro-071323-113540",
                 "Première revue systématique de l'archéoacoustique dans une revue annuelle de référence mondiale. Intègre les résultats ERC Artsoundscapes 2019-2025.",
                 "Pertinence BAM : légitime formellement le champ en 2025. Citer dans paper BAM co-signé Chapelais/Paquereau.", "NOUVEAU 2025", "10.1146/annurev-anthro-071323-113540"),
            ]),
            ("PIÉZOÉLECTRICITÉ GRANITE", [
                ("[07] Bishop (1981)", "Tectonophysics 77:T17-T22",
                 "Première mesure de la piézoélectricité dans des roches naturelles : granites, quartzites, gneiss. Champ E mesurable sous contrainte mécanique.",
                 "Pertinence BAM : fonde la voie T du modèle. Valide d33=2.31e-12 C/N et f_qz=0.32 pour granite Barabar.", "FONDATEUR", None),
                ("[08] Saksala et al. (2023)", "Rock Mechanics and Rock Engineering",
                 "Affaiblissement de 10% de la résistance en compression du granite par excitation piézoélectrique du quartz à 274.4 kHz.",
                 "Pertinence BAM : valide que l'activation acoustique du granite produit un effet piézoélectrique mesurable.", None, None),
                ("[09] Rubio Ruiz et al. (2024)", "Rock Mechanics and Rock Engineering · DOI: 10.1007/s00603-024-03948-w",
                 "Preuve expérimentale de l'affaiblissement progressif du granite Kuru par excitation piézoélectrique AC. Tests Hopkinson bar + imagerie synchrotron ESRF.",
                 "Pertinence BAM : valide la chaîne causale pression acoustique → piézoélectricité → effet mécanique mesurable.", None, "10.1007/s00603-024-03948-w"),
                ("[10] Titterton et al. (2026)", "Advanced Materials Interfaces 13:e00552 · DOI: 10.1002/admi.202500552",
                 "Activation piézoélectrique acoustique de scaffolds sur hNSC in vitro : +46% synapses excitatrices, +58% synapses inhibitrices. Formation de gaines de myéline.",
                 "Pertinence BAM : preuve directe sur cellules humaines que pression acoustique → piézo → effet synaptique neuronal. Valide la voie T de BARABAR.", "NOUVEAU 2026", "10.1002/admi.202500552"),
            ]),
            ("NEUROSCIENCES — GAMMA & PAC", [
                ("[11] Huang, Bisby, Burgess & Bush (2026)", "PNAS vol.123 No.9 · DOI: 10.1073/pnas.2513547123",
                 "PAC thêta-gamma dans l'hippocampe humain mesuré par MEG non-invasif (N=23). Slow gamma 30-70 Hz = retrieval mémoire hippocampique. Fast gamma 70-140 Hz = encodage entorhinal.",
                 "Pertinence BAM : fonde la définition des bandes gamma du score PAC (Ic v2.1). La fréquence Sudama 34.4 Hz tombe en slow gamma. Le PLV Kuramoto est le proxy de ce PAC.", "FONDATEUR", "10.1073/pnas.2513547123"),
                ("[12] Tang & Tao (2026)", "Front. Aging Neurosci. 17:1710041 · DOI: 10.3389/fnagi.2025.1710041",
                 "Revue systématique 40 Hz. -51.8% Aβ1-42 cortex auditif, -47% pTau217. La stimulation auditive seule suffit (sans visuel). Souris 5xFAD.",
                 "Pertinence BAM : stimulation auditive seule active effets neurobiologiques — pertinent pour grottes sans lumière.", None, "10.3389/fnagi.2025.1710041"),
                ("[13] Murdock et al. (2024)", "Nature",
                 "Stimulation gamma 40 Hz induit clearance glymphatique via interneurones VIP et canaux AQP4. Flux LCR augmenté.",
                 "Pertinence BAM : mécanisme glymphatique distinct du piézo BAM mais valide les effets cellulaires de la stimulation acoustique gamma.", None, None),
                ("[14] Lahijanian et al. (2024)", "Nature Scientific Reports",
                 "Stimulation auditive 40 Hz restaure la connectivité Default Mode Network (DMN) mesurée par PLV chez des patients atteints de démence.",
                 "Pertinence BAM : validation directe de l'usage du PLV comme proxy de synchronisation neuronale par stimulation auditive — même métrique que Ic v2.1.", None, None),
                ("[15] Yaghmazadeh & Buzsaki (2026)", "Brain Stimulation 19:103032 · DOI: 10.1016/j.brs.2026.103032",
                 "TRFS (Transcranial Radio Frequency Stimulation) à 945 MHz : suppression interneurones PV par chauffage thermique (ΔT>2°C). Mécanisme thermique RF ≠ piézo acoustique.",
                 "Pertinence BAM : le Sceptique MAICR cite cette étude pour questionner l'analogie RF/piézo. Valide aussi le principe général de neuromodulation transcranienne.", None, "10.1016/j.brs.2026.103032"),
                ("[16] Biswas et al. (2026)", "Imaging Neuroscience (MIT Press) · DOI: 10.1162/IMAG.a.1145",
                 "Méditants long-terme montrent gamma spontané renforcé 24-34 Hz sans stimulation externe. Pente spectrale apériodique plus raide.",
                 "Pertinence BAM : gamma 24-34 Hz adjacent à 34.4 Hz Sudama peut être renforcé durablement par expositions répétées.", None, "10.1162/IMAG.a.1145"),
                ("[17] Tsai & Park (2025)", "PLOS Biology / MIT Picower Institute",
                 "Revue des 10 ans de stimulation GENUS 40 Hz : clearance amyloïde confirmée, essais humains Phase III (Cognito Therapeutics) en cours.",
                 "Pertinence BAM : état de l'art 2025 sur la stimulation sensorielle gamma. Valide la pertinence de la bande 40 Hz.", None, None),
                ("[18] Aparicio-Terrés et al. (2025) — Frontiers HN", "Front. Human Neurosci. 19:1574836 · DOI: 10.3389/fnhum.2025.1574836",
                 "Première démonstration systématique du lien PLV → états altérés de conscience. N=20, EEG + questionnaires ASC, musique électronique 1.65-2.85 Hz.",
                 "Pertinence BAM : valide que le PLV (métrique centrale d'Ic v2.1) prédit les états altérés. Si Barabar → PLV=0.87-0.97 → prédiction état altéré.", "NOUVEAU 2025", "10.3389/fnhum.2025.1574836"),
                ("[19] Aparicio-Terrés et al. (2025) — NYAS", "Annals of the New York Academy of Sciences · DOI: 10.1111/nyas.15403",
                 "Revue neurobiologique exhaustive des états altérés induits par percussion et sons rythmiques. Proposition : entrainement thalamo-cortical basses fréquences = mécanisme sous-jacent.",
                 "Pertinence BAM : base théorique du mécanisme BAM — transmission crânienne (T_skull=26%) entraîne circuits thalamo-corticaux dans bande gamma.", "NOUVEAU 2025", "10.1111/nyas.15403"),
            ]),
        ]

        def ref_card(num_title, journal, finding, pertinence, badge, doi):
            badge_html = ""
            if badge:
                color = "#00a878" if "NOUVEAU" in badge else "#004B8D"
                badge_html = f'<span style="background:rgba(0,168,120,0.12);border:1px solid rgba(0,168,120,0.3);color:{color};font-size:0.63rem;font-weight:700;padding:1px 7px;border-radius:4px;margin-left:8px;letter-spacing:0.08em;">{badge}</span>'
            doi_html = f'<div style="font-size:0.7rem;color:#4a7090;margin-top:3px;font-family:monospace;">DOI : {doi}</div>' if doi else ""
            return f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(99,157,255,0.1);
                        border-radius:12px;padding:0.9rem 1rem;margin:0.5rem 0;">
                <div style="font-size:0.82rem;font-weight:700;color:#639dff;margin-bottom:3px;">
                    {num_title} {badge_html}
                </div>
                <div style="font-size:0.75rem;color:#5a7090;margin-bottom:6px;font-style:italic;">{journal}</div>
                {doi_html}
                <div style="font-size:0.78rem;color:#a0b4c8;margin-top:6px;line-height:1.5;">{finding}</div>
                <div style="font-size:0.75rem;color:#00a878;margin-top:6px;line-height:1.4;
                            border-left:2px solid rgba(0,168,120,0.4);padding-left:8px;">
                    {pertinence}
                </div>
            </div>"""

        for cat_name, refs in REFS_DOC:
            st.markdown(f"""
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.15em;color:#639dff;margin:1.5rem 0 0.5rem;">
                {cat_name}
            </div>""", unsafe_allow_html=True)
            for args in refs:
                st.markdown(ref_card(*args), unsafe_allow_html=True)

        st.markdown("---")

        # ── Algorithme Ic v2.1 ────────────────────────────────────────────
        st.markdown(section_header("3", "Algorithme Ic v2.1 — Référentiel complet",
                    "4 scores · Dual Pathway · Garde-fous · H0/H1/H2"), unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(0,212,184,0.05);border:1px solid rgba(0,212,184,0.2);
                    border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;">
            <div style="font-size:0.8rem;font-weight:700;color:#00d48a;margin-bottom:0.5rem;">
                Formule générale · Ic = 0.35·C + 0.30·T + 0.15·G + 0.20·PAC
            </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(metrics_table([
                ("C — Cohérence spectrale", "poids 0.35", "Cohérence de Welch signal acoustique / Kuramoto"),
                ("T — Transfert piézo", "poids 0.30", "Régression log-log pression→champ B (Bishop 1981)"),
                ("G — Facteur géologique", "poids 0.15", "Intégrale bayésienne ODF sur sigmoïde cristalline"),
                ("PAC — Phase-Amplitude", "poids 0.20", "PLV^0.7 slow γ (30-70 Hz) · Huang 2026 PNAS"),
            ]), unsafe_allow_html=True)
        with col_b:
            st.markdown(metrics_table([
                ("Acoustique directe", "C×0.6 + PAC×0.4", "Voie son seul (sans médiation EM)"),
                ("Voie piézo-EM", "T×0.7 + G×0.3", "Voie piézoélectrique granite"),
                ("H₀ VALIDÉE", "Ic < 0.35", "ÉCHEC EM — résultat négatif"),
                ("H₁ VALIDÉE", "Ic ≥ 0.65", "SUCCÈS EM — ODF>3, B>1nT, SNR>20dB"),
            ]), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,165,0,0.05);border:1px solid rgba(255,165,0,0.2);
                    border-radius:12px;padding:1rem 1.2rem;margin:0.5rem 0;">
            <div style="font-size:0.78rem;font-weight:700;color:#ffa726;margin-bottom:0.4rem;">
                ⚠️ BIFURCATION H₂ — Condition
            </div>
            <div style="font-size:0.78rem;color:#a0b4c8;line-height:1.6;">
                H₀ VALIDÉE ET voie acoustique directe > voie piézo-EM<br>
                → Le son seul synchronise les oscillateurs gamma sans médiation piézoélectrique.<br>
                → Référence : Aparicio-Terrés 2025 NYAS (entrainement thalamo-cortical basses fréquences).<br>
                → Résultat publiable comme preuve computationnelle de l'hypothèse acoustique directe.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Pipeline MAICR ────────────────────────────────────────────────
        st.markdown(section_header("4", "Pipeline MAICR 5 Agents",
                    "temperature=0.0 · Whitelist post-parsing · validate_inputs()"), unsafe_allow_html=True)

        agents_info = [
            ("01 — WebProspector", "Triangulation bibliographique whitelist 20 refs. Post-parsing Python supprime toute hallucination hors liste. MAX 2 refs pertinentes.",
             "support_litteraire : FORT | PARTIEL | ABSENT · references_cles[]"),
            ("02 — ControleurJuge", "Débat Sceptique (Buzsaki 2026, limites piézo) vs Enthousiaste (Huang 2026, Cook 2008, Aparicio-Terrés 2025, Titterton 2026). H₀ = résultat négatif, JAMAIS valorisé.",
             "physicien_sceptique · neuro_acousticien_enthousiaste · consensus_maicr"),
            ("03 — ExpertScribe", "Résumé exécutif style CNRS (180 mots max). Mentionne dual pathway, mécanisme dominant, convergences 2025-2026.",
             "resume_executif : 180 mots · style CNRS"),
        ]

        for agent_name, desc, output in agents_info:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border-left:3px solid #639dff;
                        border-radius:0 10px 10px 0;padding:0.8rem 1rem;margin:0.5rem 0;">
                <div style="font-size:0.82rem;font-weight:700;color:#639dff;">Agent {agent_name}</div>
                <div style="font-size:0.78rem;color:#a0b4c8;margin-top:4px;line-height:1.5;">{desc}</div>
                <div style="font-size:0.73rem;color:#5a7090;margin-top:4px;font-family:monospace;">↳ {output}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem;font-size:0.72rem;color:#4a5568;text-align:center;
                    font-family:'JetBrains Mono',monospace;">
            Samuel Chapelais Veillet — Opérateur MAICR · BARABAR v4.0 · Laboratoire BAM · 28 mars 2026
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
