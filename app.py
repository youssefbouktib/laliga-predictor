"""Streamlit dashboard for the La Liga predictor.

    streamlit run app.py

Reads the files written by `run.py evaluate`, `predict` and `settle`; it never
trains a model itself, so it opens instantly. When reports/ is missing (e.g. a
hosted copy of the repo) it falls back to the snapshot in site_data/.
"""
from __future__ import annotations

import datetime as dt
import html
import pathlib
import urllib.parse

import altair as alt
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).parent
REPORTS = ROOT / "reports"
if not (REPORTS / "model_comparison.csv").exists() and (ROOT / "site_data").exists():
    REPORTS = ROOT / "site_data"

# ---------------------------------------------------------------- design tokens
INK, MUTED, FAINT = "#15201a", "#4a554e", "#6f7a72"
LINE, GRID, SURFACE = "#d9ddd5", "#e2e5de", "#ffffff"
ACCENT, ACCENT_SOFT = "#1d6b4c", "#e1efe7"
GOOD, BAD, WARN_SOFT = "#1d7a3e", "#b3352f", "#fbf0dc"
C_MODEL, C_BOOKIE = "#2a78d6", "#eb6834"                      # model vs market, everywhere
C_HOME, C_DRAW, C_AWAY = "#2a78d6", "#a7aca3", "#6250d6"     # home / neutral draw / away
GOLD, RED = "#f1bf00", "#c8102e"                              # Spanish accents

# club kit colours for the monogram badges: (code, fill, ring, text)
TEAMS = {
    "Alaves": ("ALA", "#0761af", "#ffffff", "#ffffff"), "Almeria": ("ALM", "#ee1119", "#ffffff", "#ffffff"),
    "Ath Bilbao": ("ATH", "#ee2523", "#ffffff", "#ffffff"), "Ath Madrid": ("ATM", "#cb3524", "#272e61", "#ffffff"),
    "Barcelona": ("BAR", "#a50044", "#004d98", "#ffffff"), "Betis": ("BET", "#0bb363", "#ffffff", "#ffffff"),
    "Cadiz": ("CAD", "#f4d000", "#0045a7", "#0045a7"), "Celta": ("CEL", "#8ac3ee", "#e5254e", "#0b2a4a"),
    "Elche": ("ELC", "#05642c", "#ffffff", "#ffffff"), "Espanol": ("ESP", "#007fc8", "#ffffff", "#ffffff"),
    "Getafe": ("GET", "#005999", "#ffffff", "#ffffff"), "Girona": ("GIR", "#cd2534", "#ffffff", "#ffffff"),
    "Granada": ("GRA", "#c31632", "#ffffff", "#ffffff"), "La Coruna": ("DEP", "#0067b1", "#ffffff", "#ffffff"),
    "Las Palmas": ("LPA", "#ffe400", "#0049a1", "#0049a1"), "Leganes": ("LEG", "#0b4ea2", "#ffffff", "#ffffff"),
    "Levante": ("LEV", "#b4053f", "#004a9f", "#ffffff"), "Malaga": ("MAL", "#6bb6e8", "#ffffff", "#0b2a4a"),
    "Mallorca": ("MLL", "#e20613", "#111111", "#ffffff"), "Osasuna": ("OSA", "#d91a21", "#0a346f", "#ffffff"),
    "Oviedo": ("OVI", "#0047ab", "#ffffff", "#ffffff"), "Real Madrid": ("RMA", "#ffffff", "#febe10", "#00529f"),
    "Santander": ("RAC", "#00843d", "#ffffff", "#ffffff"), "Sevilla": ("SEV", "#ffffff", "#d4021d", "#d4021d"),
    "Sociedad": ("RSO", "#0067b1", "#ffffff", "#ffffff"), "Valencia": ("VAL", "#ffffff", "#ee3524", "#111111"),
    "Valladolid": ("VLL", "#921b88", "#ffffff", "#ffffff"), "Vallecano": ("RAY", "#ffffff", "#e53027", "#e53027"),
    "Villarreal": ("VIL", "#ffe667", "#005187", "#005187"),
}

PITCH_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 440" fill="none" stroke="#ffffff" stroke-opacity=".22" stroke-width="3">
<rect x="10" y="10" width="660" height="420" rx="4"/><line x1="340" y1="10" x2="340" y2="430"/>
<circle cx="340" cy="220" r="62"/><circle cx="340" cy="220" r="4" fill="#ffffff" fill-opacity=".22"/>
<rect x="10" y="110" width="104" height="220"/><rect x="10" y="168" width="36" height="104"/>
<rect x="566" y="110" width="104" height="220"/><rect x="634" y="168" width="36" height="104"/>
<path d="M114 180 A50 50 0 0 1 114 260"/><path d="M566 180 A50 50 0 0 0 566 260"/></svg>"""
PITCH_BG = 'url("data:image/svg+xml,' + urllib.parse.quote(PITCH_SVG.replace("\n", "")) + '")'

st.set_page_config(page_title="La Liga Predictor", page_icon="⚽", layout="wide")

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], button, input {
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2.2rem; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3 { font-family: "Bricolage Grotesque", "Segoe UI", sans-serif !important; letter-spacing: -.01em; }

.lp-eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: #1d6b4c;
  letter-spacing: .08em; text-transform: uppercase; }
.lp-title { font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 46px;
  line-height: 1.04; color: #15201a; margin: 6px 0 10px; }
.lp-lede { font-size: 17px; color: #4a554e; max-width: 68ch; margin: 0; line-height: 1.55; }
.lp-chip { display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 12px;
  color: #4a554e; background: #eceee8; border-radius: 99px; padding: 3px 10px; margin-top: 14px; }

.lp-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 26px 0 8px; }
@media (max-width: 900px) { .lp-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.lp-kpi { background: #fff; border: 1px solid #d9ddd5; border-radius: 12px; padding: 16px 18px; }
.lp-kpi.hero { background: #1d6b4c; border-color: #1d6b4c; }
.lp-kpi .k { font-size: 12.5px; color: #6f7a72; text-transform: uppercase; letter-spacing: .06em; }
.lp-kpi .v { font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 34px;
  color: #15201a; line-height: 1.1; margin: 6px 0 4px; font-variant-numeric: tabular-nums; }
.lp-kpi .s { font-size: 13.5px; color: #4a554e; line-height: 1.4; }
.lp-kpi.hero .k { color: #cfe6da; } .lp-kpi.hero .v { color: #fff; } .lp-kpi.hero .s { color: #e1efe7; }

button[data-baseweb="tab"] p { font-size: 15px; font-weight: 500; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #d9ddd5; border-radius: 12px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: "Bricolage Grotesque", sans-serif; font-variant-numeric: tabular-nums; }

.lp-fx { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; margin-top: 8px; }
.lp-card { background: #fff; border: 1px solid #d9ddd5; border-radius: 12px; padding: 16px 18px;
  display: flex; flex-direction: column; gap: 10px; }
.lp-card.value { border-color: #1d6b4c; box-shadow: inset 4px 0 0 #1d6b4c; }
.lp-card .when { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: #6f7a72; }
.lp-card .teams { font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 20px;
  color: #15201a; line-height: 1.2; }
.lp-card .teams span { color: #6f7a72; font-weight: 500; font-size: 15px; padding: 0 6px; }
.lp-bar { display: flex; height: 10px; border-radius: 99px; overflow: hidden; gap: 2px; background: #fff; }
.lp-bar i { display: block; height: 100%; }
.lp-probs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 13px; color: #4a554e; }
.lp-probs b { display: block; font-family: "IBM Plex Mono", monospace; font-weight: 500; font-size: 15px; color: #15201a; }
.lp-probs .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.lp-pill { align-self: flex-start; font-size: 13px; font-weight: 600; border-radius: 99px; padding: 4px 11px; }
.lp-pill.yes { background: #e1efe7; color: #1d6b4c; }
.lp-pill.no { background: #eceee8; color: #6f7a72; font-weight: 500; }

.lp-note { background: #fbf0dc; border-radius: 10px; padding: 12px 16px; color: #5b4310; font-size: 14px; line-height: 1.5; }
.lp-steps { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.lp-step { background: #fff; border: 1px solid #d9ddd5; border-radius: 12px; padding: 14px 16px; }
.lp-step .n { font-family: "IBM Plex Mono", monospace; color: #1d6b4c; font-size: 13px; }
.lp-step h4 { font-family: "Bricolage Grotesque", sans-serif; margin: 4px 0 6px; font-size: 17px; color: #15201a; }
.lp-step p { margin: 0; font-size: 14px; color: #4a554e; line-height: 1.5; }

/* ---------- football layer ---------- */
[data-testid="stAppViewContainer"] {
  background: repeating-linear-gradient(90deg, #f6f7f4 0 140px, #f1f3ee 140px 280px); }
.lp-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 34px 36px 26px;
  background: PITCH_BG right -30px center / auto 118% no-repeat,
              repeating-linear-gradient(90deg, #17603f 0 72px, #1b6b47 72px 144px); color: #fff; }
.lp-hero::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 6px;
  background: linear-gradient(90deg, #c8102e 0 33%, #f1bf00 33% 67%, #c8102e 67%); }
.lp-hero .lp-eyebrow { color: #f1bf00; }
.lp-hero .lp-title { color: #fff; font-size: 52px; }
.lp-hero .lp-lede { color: #dcebe2; position: relative; }
.lp-hero .lp-chip { background: rgba(255,255,255,.13); color: #e9f3ed; }
.lp-ticker { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 20px; position: relative; }
.lp-ball { display: inline-block; animation: lp-spin 6s linear infinite; }
@keyframes lp-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .lp-ball { animation: none; } }

.lp-badge { display: inline-flex; align-items: center; justify-content: center; border-radius: 50%;
  font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; letter-spacing: .02em;
  flex: none; box-shadow: 0 1px 2px rgba(0,0,0,.18); }
.lp-badge.sm { width: 34px; height: 34px; font-size: 10.5px; border: 2px solid; }
.lp-badge.lg { width: 56px; height: 56px; font-size: 15px; border: 3px solid; }

.lp-kpi { border-top: 4px solid #c8102e; }
.lp-kpi.hero { background: #f1bf00; border-color: #f1bf00; border-top-color: #c8102e; }
.lp-kpi.hero .k { color: #5a4600; } .lp-kpi.hero .v { color: #15201a; } .lp-kpi.hero .s { color: #3d3000; }

[role="tablist"] { gap: 8px; flex-wrap: wrap; border: none !important; box-shadow: none !important; }
[role="tablist"] > :not([data-testid="stTab"]) { display: none; }
[data-testid="stTab"] { background: #fff; border: 1px solid #d9ddd5 !important; border-radius: 99px;
  padding: 6px 16px !important; margin: 0 !important; }
[data-testid="stTab"][aria-selected="true"] { background: #15201a; border-color: #15201a !important; }
[data-testid="stTab"][aria-selected="true"] p { color: #fff !important; }
.lp-card { padding: 0; gap: 0; overflow: hidden; }
.lp-card.value { box-shadow: 0 6px 18px rgba(29,107,76,.14); border-color: #1d6b4c; }
.lp-mtop { display: flex; justify-content: space-between; align-items: center; padding: 9px 16px;
  background: #15201a; color: #cfd8d2; font-family: "IBM Plex Mono", monospace; font-size: 12px; }
.lp-mtop .tag { background: #f1bf00; color: #15201a; font-weight: 600; border-radius: 4px; padding: 1px 7px; }
.lp-mbody { padding: 16px 18px 16px; display: flex; flex-direction: column; gap: 14px; }
.lp-match { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px; }
.lp-match .side { display: flex; flex-direction: column; align-items: center; gap: 7px; text-align: center; }
.lp-match .nm { font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 16px; color: #15201a; }
.lp-match .vs { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: #6f7a72;
  border: 1px solid #d9ddd5; border-radius: 99px; padding: 3px 9px; }
.lp-odds { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.lp-odd { border: 1px solid #d9ddd5; border-radius: 10px; padding: 7px 8px; text-align: center; background: #fafbf8; }
.lp-odd .l { font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: #6f7a72; }
.lp-odd .p { font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 20px; color: #15201a; }
.lp-odd .o { font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: #4a554e; }
.lp-odd.pick { border: 2px solid #f1bf00; background: #fffaea; }
.lp-pill.yes { background: #1d6b4c; color: #fff; }

.lp-table { width: 100%; border-collapse: separate; border-spacing: 0 5px; font-size: 14.5px; }
.lp-table th { text-align: left; font-weight: 500; font-size: 12px; color: #6f7a72; text-transform: uppercase;
  letter-spacing: .06em; padding: 0 12px; }
.lp-table td { background: #fff; padding: 10px 12px; border-top: 1px solid #e2e5de; border-bottom: 1px solid #e2e5de; }
.lp-table td:first-child { border-left: 5px solid #d9ddd5; border-radius: 8px 0 0 8px;
  font-family: "IBM Plex Mono", monospace; color: #4a554e; width: 52px; }
.lp-table td:last-child { border-right: 1px solid #e2e5de; border-radius: 0 8px 8px 0; }
.lp-table td.num { font-family: "IBM Plex Mono", monospace; text-align: right; }
.lp-table tr.champ td { background: #fffaea; } .lp-table tr.champ td:first-child { border-left-color: #f1bf00; }
.lp-table tr.europe td:first-child { border-left-color: #1d6b4c; }
.lp-table tr.releg td:first-child { border-left-color: #c8102e; }
.lp-zones { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: #4a554e; margin: 4px 0 18px; }
.lp-zones i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.lp-step .n { display: inline-flex; width: 34px; height: 34px; border-radius: 50%; background: #15201a;
  color: #f1bf00; align-items: center; justify-content: center; font-weight: 500; }
</style>
""".replace("PITCH_BG", PITCH_BG))


# ---------------------------------------------------------------- data helpers

@st.cache_data
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:
    # mtime is part of the cache key: a rewritten report is re-read at once
    return pd.read_csv(path, **kw)


def read(name: str, **kw) -> pd.DataFrame | None:
    f = REPORTS / name
    return _read_csv(str(f), f.stat().st_mtime, **kw) if f.exists() else None


def pct(x, digits=1):
    return f"{x:+.{digits}%}" if pd.notna(x) else "–"


def esc(x) -> str:
    return html.escape(str(x))


def style(chart: alt.Chart, height: int = 300) -> alt.Chart:
    """One quiet chart style: recessive axes and grid, text in ink tokens."""
    return (chart.properties(height=height)
            .configure(font="IBM Plex Sans")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal", gridColor=GRID, domainColor=LINE, tickColor=LINE)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, orient="top",
                              title=None, symbolStrokeWidth=0))


def badge(team: str, size: str = "sm") -> str:
    """Monogram badge in club colours (no real crests)."""
    code, fill, ring, text = TEAMS.get(team, (str(team)[:3].upper(), "#4a554e", "#ffffff", "#ffffff"))
    return (f'<span class="lp-badge {size}" title="{esc(team)}" '
            f'style="background:{fill};border-color:{ring};color:{text}">{esc(code)}</span>')





def chart_title(title: str, sub: str | None = None):
    st.markdown(f"#### {title}")
    if sub:
        st.caption(sub)


# ---------------------------------------------------------------- load
comp = read("model_comparison.csv")
if comp is None:
    st.warning("No reports yet. Run `python run.py download` then `python run.py evaluate`.")
    st.stop()

bets = read("strategy_bets.csv", parse_dates=["date"])
by_season = read("strategy_by_season.csv")
control = read("control_clv.csv")
bt = read("backtest_summary.csv")
metrics = read("probability_metrics.csv")
cal = read("calibration.csv")

main_model = bets.model.iloc[0] if bets is not None and len(bets) else comp.model.iloc[0]
main_price = bets.price.iloc[0] if bets is not None and len(bets) else "best"
control_val = None
if control is not None and main_price in set(control.price):
    control_val = float(control.set_index("price").avg_clv_fair[main_price])
main_row = None
if bt is not None and "price" in bt.columns:
    m = bt[(bt.model == main_model) & (bt.price == main_price) & (bt.staking == "flat")
           & (bt.min_edge.round(3) == 0.05)]
    main_row = m.iloc[0] if len(m) else None

# ---------------------------------------------------------------- header
updated = dt.datetime.fromtimestamp((REPORTS / "model_comparison.csv").stat().st_mtime)
preds = read("predictions.csv", usecols=["season", "home"])
season_now, teams_now = "", []
if preds is not None and len(preds):
    last = preds.season.iloc[-1]
    teams_now = sorted(preds.loc[preds.season == last, "home"].unique())
    season_now = f"{int(last)}/{(int(last) + 1) % 100:02d}" if str(last).isdigit() else str(last)
st.html(f"""
<div class="lp-hero">
<div class="lp-eyebrow"><span class="lp-ball">⚽</span> Primera División{' · ' + esc(season_now) if season_now else ''} · value detection</div>
<div class="lp-title">La Liga Predictor</div>
<p class="lp-lede">Nine models forecast every match, then the system hunts for prices the market
will later correct. Judged against the sharp closing line, not by accuracy or lucky profit.</p>
<span class="lp-chip">reports updated {updated:%d %b %Y, %H:%M} · main model {esc(main_model)} · {esc(main_price)} price</span>
<div class="lp-ticker">{''.join(badge(t) for t in teams_now)}</div>
</div>""")

if main_row is not None:
    clv = main_row.get("avg_clv_fair")
    pos = main_row.get("pct_clv_fair_pos")
    lo, hi = main_row.get("roi_ci_low"), main_row.get("roi_ci_high")
    st.html(f"""
<div class="lp-kpis">
  <div class="lp-kpi hero"><div class="k">Beats the closing line by</div>
    <div class="v">{pct(clv)}</div><div class="s">average CLV of {int(main_row.bets)} backtest bets</div></div>
  <div class="lp-kpi"><div class="k">Bets beating the close</div>
    <div class="v">{pos:.0%}</div><div class="s">share of bets with positive CLV</div></div>
  <div class="lp-kpi"><div class="k">Random bets, same prices</div>
    <div class="v">{pct(control_val)}</div><div class="s">the bar the model has to clear</div></div>
  <div class="lp-kpi"><div class="k">Backtest ROI</div>
    <div class="v">{pct(main_row.roi)}</div><div class="s">95% range {pct(lo, 0)} to {pct(hi, 0)}: not yet proven</div></div>
</div>""")

tab_week, tab_edge, tab_models, tab_paper, tab_how = st.tabs(
    ["⚽ This week", "📈 The edge", "🏆 Model table", "📝 Paper trades", "📖 How it works"])

# ---------------------------------------------------------------- this week
with tab_week:
    up = read("upcoming.csv", parse_dates=["date"])
    if up is None or up.empty:
        st.info("No upcoming fixtures. Run `python run.py download` and `python run.py predict`.")
    else:
        has_best = {"best_h", "best_d", "best_a"}.issubset(up.columns)
        c = st.columns([1, 1, 1])
        min_edge = c[0].slider("Minimum edge", 0.0, 0.20, 0.05, 0.01, format="%.2f")
        max_edge = c[1].slider("Maximum edge", 0.05, 1.0, 0.15, 0.01, format="%.2f",
                               help="Bigger disagreements with the market are usually model errors.")
        price = c[2].radio("Price", ["best", "book"] if has_best else ["book"], horizontal=True,
                           help="best = best price across all bookmakers (line shopping)")
        prefix = "best" if price == "best" else "odds"
        names = {"h": "Home", "d": "Draw", "a": "Away"}
        cards, rows = [], []
        for r in up.sort_values("date").itertuples():
            probs = {o: getattr(r, f"p_{o}") for o in "hda"}
            odds = {o: getattr(r, f"{prefix}_{o}") for o in "hda"}
            evs = {o: probs[o] * odds[o] - 1 for o in "hda" if pd.notna(odds[o])}
            pick = max(evs, key=evs.get) if evs else None
            ev = evs[pick] if pick else float("nan")
            is_value = pick is not None and min_edge < ev <= max_edge
            bar = "".join(f'<i style="width:{probs[o] * 100:.1f}%;background:{col}"></i>'
                          for o, col in zip("hda", (C_HOME, C_DRAW, C_AWAY)))
            sym = {"h": "1 · home", "d": "X · draw", "a": "2 · away"}
            cells = "".join(
                f'<div class="lp-odd{" pick" if is_value and o == pick else ""}">'
                f'<div class="l">{sym[o]}</div><div class="p">{probs[o]:.0%}</div>'
                f'<div class="o">{f"@ {odds[o]:.2f}" if pd.notna(odds[o]) else "no price"}</div></div>'
                for o in "hda")
            pill = (f'<span class="lp-pill yes">Value bet: {names[pick]} @ {odds[pick]:.2f} · EV {ev:+.1%}</span>'
                    if is_value else
                    f'<span class="lp-pill no">No value · best EV {ev:+.1%}</span>' if pick else
                    '<span class="lp-pill no">No prices yet</span>')
            cards.append(f"""<div class="lp-card{' value' if is_value else ''}">
  <div class="lp-mtop"><span>MATCHDAY · {r.date:%a %d %b %Y}</span>{'<span class="tag">VALUE</span>' if is_value else ''}</div>
  <div class="lp-mbody">
  <div class="lp-match">
    <div class="side">{badge(r.home, "lg")}<div class="nm">{esc(r.home)}</div></div>
    <div class="vs">VS</div>
    <div class="side">{badge(r.away, "lg")}<div class="nm">{esc(r.away)}</div></div>
  </div>
  <div class="lp-bar" role="img" aria-label="home {probs['h']:.0%}, draw {probs['d']:.0%}, away {probs['a']:.0%}">{bar}</div>
  <div class="lp-odds">{cells}</div>
  {pill}
  </div>
</div>""")
            rows.append({"date": r.date.date(), "match": f"{r.home} v {r.away}",
                         "P(H)": probs["h"], "P(D)": probs["d"], "P(A)": probs["a"],
                         "odds H": odds["h"], "odds D": odds["d"], "odds A": odds["a"],
                         "best EV": ev, "value bet": names[pick] if is_value else ""})
        n_value = sum(1 for r in rows if r["value bet"])
        st.caption(f"{len(rows)} upcoming game{'s' if len(rows) != 1 else ''} · {n_value} value "
                   f"bet{'s' if n_value != 1 else ''} · model {up.model.iloc[0]} · prices: "
                   + ("best available across bookmakers" if price == "best" else str(up.book.iloc[0])))
        st.html(f'<div class="lp-fx">{"".join(cards)}</div>')
        with st.expander("Table view"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={**{c: st.column_config.NumberColumn(format="%.2f") for c in
                                           ["P(H)", "P(D)", "P(A)", "odds H", "odds D", "odds A"]},
                                        "best EV": st.column_config.NumberColumn(format="%+.3f")})
        if price == "best":
            st.html('<div class="lp-note">Best price = the highest price found across bookmakers when '
                    'the data was collected. Check it is still available before you paper-trade it.</div>')

# ---------------------------------------------------------------- the edge
with tab_edge:
    if by_season is None or by_season.empty:
        st.info("Run `python run.py evaluate` to build the backtest.")
    else:
        chart_title("Do the bets beat the closing line?",
                    f"Average CLV vs the sharp closing line per season · {main_model}, {main_price} price, "
                    "edge 5–15% · dashed line: betting every outcome with no model")
        d = by_season.assign(clv=by_season.avg_clv_fair * 100,
                             pos=by_season.pct_clv_fair_pos * 100, roi=by_season.roi * 100)
        base = alt.Chart(d).encode(x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(color=C_MODEL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                             size=46).encode(
            y=alt.Y("clv:Q", title="average CLV (%)"),
            tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("bets:Q", title="Bets"),
                     alt.Tooltip("clv:Q", title="CLV %", format="+.2f"),
                     alt.Tooltip("pos:Q", title="Beat the close %", format=".0f"),
                     alt.Tooltip("roi:Q", title="ROI %", format="+.1f")])
        labels = base.mark_text(dy=-8, color=MUTED, fontSize=12).encode(
            y=alt.Y("lab_y:Q"), text=alt.Text("clv:Q", format="+.1f")).transform_calculate(
            lab_y="max(datum.clv, 0)")
        layers = [bars, labels, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=LINE).encode(y="y:Q")]
        if control_val is not None:
            layers.append(alt.Chart(pd.DataFrame({"y": [control_val * 100]}))
                          .mark_rule(color=C_BOOKIE, strokeDash=[6, 4], strokeWidth=2)
                          .encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="No-model CLV %", format="+.2f")]))
        st.altair_chart(style(alt.layer(*layers), 320), width="stretch")

    if bets is not None and len(bets):
        cols = st.columns(2, gap="large")
        with cols[0]:
            chart_title("Flat stakes", "Cumulative profit, 1 unit per bet")
            line = alt.Chart(bets).mark_line(color=C_MODEL, strokeWidth=2).encode(
                x=alt.X("date:T", title=None), y=alt.Y("cum_profit:Q", title="profit (units)"),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("home:N", title="Home"),
                         alt.Tooltip("away:N", title="Away"), alt.Tooltip("pick:N", title="Pick"),
                         alt.Tooltip("bet_odds:Q", title="Odds", format=".2f"),
                         alt.Tooltip("cum_profit:Q", title="Running profit", format="+.1f")])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=LINE).encode(y="y:Q")
            st.altair_chart(style(alt.layer(zero, line), 260), width="stretch")
        with cols[1]:
            chart_title("Quarter Kelly", "Bankroll from 100, stakes capped at 5%")
            line = alt.Chart(bets).mark_line(color=ACCENT, strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("kelly_bankroll:Q", title="bankroll", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("kelly_bankroll:Q", title="Bankroll", format=".1f")])
            start = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color=LINE).encode(y="y:Q")
            st.altair_chart(style(alt.layer(start, line), 260), width="stretch")
        if main_row is not None:
            st.html(f'<div class="lp-note">Positive CLV but a 95% ROI range of {pct(main_row.roi_ci_low, 0)} '
                    f'to {pct(main_row.roi_ci_high, 0)}: the bets beat the market\'s final price, but '
                    f'{int(main_row.bets)} bets are too few to prove profit. Paper trading is the real test.</div>')

    if bt is not None:
        with st.expander("Every strategy tested"):
            st.dataframe(bt, hide_index=True, width="stretch")

# ---------------------------------------------------------------- models
with tab_models:
    best = comp.iloc[0]
    c = st.columns(3)
    c[0].metric("Best model", best.model)
    c[1].metric("Its log loss", f"{best.logloss:.4f}")
    c[2].metric("Gap to bookmaker", f"{best.gap_logloss:+.4f}",
                help="Model log loss minus the sharp closing line's. Above zero = the market is better.")

    chart_title("The standings", "Every model ranked by log loss over all test matches, like a league table. "
                "Lower is better, and the sharp closing line sits top.")
    table = [("🏆", "Sharp closing line", "the market's final price", float(comp.bookie_logloss.iloc[0]), None, "champ")]
    ranked = comp.sort_values("logloss").reset_index(drop=True)
    for i, r in ranked.iterrows():
        market = bool(r.get("uses_market", False))
        zone = "releg" if i >= len(ranked) - 3 and len(ranked) > 4 else "europe" if market else ""
        table.append((f"{i + 2}", r.model, ("opening odds only" if r.model == "market" else "opening odds + football") if market else "football data only",
                      float(r.logloss), float(r.gap_logloss), zone))
    st.html('<table class="lp-table"><tr><th>Pos</th><th>Model</th><th>Inputs</th>'
            '<th style="text-align:right">Log loss</th><th style="text-align:right">Gap</th></tr>'
            + "".join(f'<tr class="{z}"><td>{p}</td><td><b>{esc(m)}</b></td><td>{esc(k)}</td>'
                      f'<td class="num">{ll:.4f}</td><td class="num">{"–" if g is None else f"{g:+.4f}"}</td></tr>'
                      for p, m, k, ll, g, z in table) + "</table>"
            f'<div class="lp-zones"><span><i style="background:{GOLD}"></i>Champion: the benchmark</span>'
            f'<span><i style="background:{ACCENT}"></i>Market-aware models</span>'
            f'<span><i style="background:{RED}"></i>Relegation zone</span></div>')

    chart_title("Every model vs the closing line",
                "Log loss over all test matches, lower is better · the line shows each model's gap to the market")
    uses = comp["uses_market"].astype(bool) if "uses_market" in comp.columns else pd.Series(False, index=comp.index)
    d = comp.assign(kind=uses.map({True: "uses market odds", False: "football only"}))
    order = d.sort_values("logloss").model.tolist()
    y = alt.Y("model:N", sort=order, title=None)
    gap = alt.Chart(d).mark_rule(color=LINE, strokeWidth=3).encode(y=y, x="bookie_logloss:Q", x2="logloss:Q")
    pts = alt.Chart(d).mark_circle(size=110, color=C_MODEL, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
        y=y, x=alt.X("logloss:Q", title="log loss", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("kind:N", title="Inputs"),
                 alt.Tooltip("logloss:Q", title="Log loss", format=".4f"),
                 alt.Tooltip("gap_logloss:Q", title="Gap to market", format="+.4f")])
    mkt = alt.Chart(d).mark_circle(size=110, color=C_BOOKIE, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
        y=y, x="bookie_logloss:Q",
        tooltip=[alt.Tooltip("bookie_logloss:Q", title="Closing line log loss", format=".4f")])
    lab = alt.Chart(d).mark_text(align="left", dx=10, color=MUTED, fontSize=12).encode(
        y=y, x="logloss:Q", text=alt.Text("gap_logloss:Q", format="+.4f"))
    st.altair_chart(style(alt.layer(gap, pts, mkt, lab), 34 * len(d) + 30), width="stretch")
    st.html(f'<div style="display:flex;gap:18px;font-size:13px;color:{MUTED}">'
            f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{C_MODEL};margin-right:6px"></span>Model</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{C_BOOKIE};margin-right:6px"></span>Sharp closing line</span></div>')

    cols = st.columns(2, gap="large")
    with cols[0]:
        if metrics is not None:
            model_name = st.selectbox("Per-season detail", comp.model.tolist(),
                                      index=comp.model.tolist().index(main_model) if main_model in comp.model.tolist() else 0)
            m = metrics[metrics.model == model_name]
            long = pd.concat([m[["season", "model_logloss"]].rename(columns={"model_logloss": "v"}).assign(who=model_name),
                              m[["season", "bookie_logloss"]].rename(columns={"bookie_logloss": "v"}).assign(who="closing line")])
            ch = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
                x=alt.X("season:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("v:Q", title="log loss", scale=alt.Scale(zero=False)),
                color=alt.Color("who:N", scale=alt.Scale(domain=[model_name, "closing line"],
                                                         range=[C_MODEL, C_BOOKIE])),
                tooltip=[alt.Tooltip("season:N", title="Season"), alt.Tooltip("who:N", title="Source"),
                         alt.Tooltip("v:Q", title="Log loss", format=".4f")])
            st.altair_chart(style(ch, 280), width="stretch")
    with cols[1]:
        if cal is not None and len(cal):
            st.markdown(f"**Calibration** · {main_model}")
            st.caption("When the model says 40%, does it happen 40% of the time?")
            c2 = cal.assign(who=cal.source.map({"model": main_model, "bookie": "closing line"}))
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                color=FAINT, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
            ch = alt.Chart(c2).mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=2).encode(
                x=alt.X("predicted:Q", title="predicted probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("observed:Q", title="observed frequency", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("who:N", scale=alt.Scale(domain=[main_model, "closing line"],
                                                         range=[C_MODEL, C_BOOKIE])),
                tooltip=[alt.Tooltip("who:N", title="Source"), alt.Tooltip("n:Q", title="Outcomes"),
                         alt.Tooltip("predicted:Q", format=".3f", title="Predicted"),
                         alt.Tooltip("observed:Q", format=".3f", title="Observed")])
            st.altair_chart(style(alt.layer(diag, ch), 280), width="stretch")
    with st.expander("Full comparison table"):
        st.dataframe(comp, hide_index=True, width="stretch")

# ---------------------------------------------------------------- paper trades
with tab_paper:
    raw = read("paper_trades.csv", parse_dates=["date"])
    settled = read("paper_trades_settled.csv", parse_dates=["date"])
    if raw is None and settled is None:
        st.info("No paper trades yet. `python run.py predict` logs value bets automatically.")
    else:
        # the settled file is a snapshot from the last `settle`: join it onto the
        # full log so trades logged since then still show up (as open)
        if settled is None:
            st.info("Run `python run.py settle` to attach results and closing odds.")
            pt = raw.assign(status="open", profit=float("nan"), clv=float("nan"))
        elif raw is None:
            pt = settled
        else:
            key = ["date", "home", "away"]
            extra = settled.drop(columns=[c for c in raw.columns if c not in key], errors="ignore")
            pt = raw.merge(extra.drop_duplicates(key), on=key, how="left")
            pt["status"] = pt.status.fillna("open")
        done = pt[pt.status != "open"]
        c = st.columns(4)
        c[0].metric("Bets logged", len(pt))
        c[1].metric("Settled", len(done))
        c[2].metric("Profit (units)", f"{done.profit.sum():+.2f}" if len(done) else "–")
        clv = (done.clv_fair if "clv_fair" in done.columns else done.clv).dropna()
        c[3].metric("Avg CLV vs sharp close", pct(clv.mean()) if len(clv) else "–",
                    help="Bet odds x fair closing probability - 1. Positive over many bets "
                         "is the best evidence of a real edge.")
        if len(done):
            series = done.sort_values("date").assign(cum=lambda x: x.profit.cumsum())
            ch = alt.Chart(series).mark_line(color=C_MODEL, strokeWidth=2, point=True).encode(
                x=alt.X("date:T", title=None), y=alt.Y("cum:Q", title="profit (units)"),
                tooltip=["date:T", "home:N", "away:N", "pick:N", alt.Tooltip("cum:Q", format="+.2f")])
            st.altair_chart(style(ch, 240), width="stretch")
        show = [c for c in ["date", "home", "away", "pick", "price", "bet_odds", "p_model", "ev",
                            "status", "profit", "clv_fair", "model", "logged_at"] if c in pt.columns]
        st.dataframe(pt[show], hide_index=True, width="stretch",
                     column_config={"bet_odds": st.column_config.NumberColumn("odds", format="%.2f"),
                                    "p_model": st.column_config.NumberColumn("model prob.", format="%.3f"),
                                    "ev": st.column_config.NumberColumn("EV", format="%+.3f"),
                                    "clv_fair": st.column_config.NumberColumn("CLV", format="%+.3f"),
                                    "profit": st.column_config.NumberColumn(format="%+.2f")})

# ---------------------------------------------------------------- how it works
with tab_how:
    steps = [
        ("Data", "14 seasons of results, odds and shots from football-data.co.uk. For every match: "
                 "one bookmaker's opening and closing price, the best price anywhere, and the sharp "
                 "closing line used as the benchmark."),
        ("Features", "Elo ratings, last-5 form and rolling shots on target, built in one pass "
                     "through time so a match never sees itself or the future."),
        ("Models", "Logistic regression, LightGBM, Dixon-Coles and stacked combinations. The main "
                   "model starts from the sharp opening odds and adds Dixon-Coles and Elo."),
        ("Walk-forward test", "Every season is predicted by models trained only on earlier seasons. "
                              "Never a random split."),
        ("Value detection", "Bet the outcome with the highest expected value when it is 5–15%, at "
                            "the best price. Bigger edges are usually model errors."),
        ("Judging", "Closing line value against the sharp close, compared with random bets, plus a "
                    "bootstrap range for ROI. Profit alone can be luck."),
    ]
    st.html('<div class="lp-steps">' + "".join(
        f'<div class="lp-step"><div class="n">{i}</div><h4>{esc(t)}</h4><p>{esc(b)}</p></div>'
        for i, (t, b) in enumerate(steps, 1)) + "</div>")
    st.html('<div class="lp-note" style="margin-top:14px">Educational project, paper trading only. '
            'A positive backtest is not a guarantee of profit, and bookmakers limit accounts that '
            'consistently beat the closing line.</div>')
