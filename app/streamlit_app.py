import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

st.set_page_config(
    page_title="Cinecast — Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4dc;
}

.stApp { background-color: #0a0a0f; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #e8e4dc;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
    color: #e8e4dc;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #8a8070;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.metric-card {
    background: #13131a;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}

.metric-label {
    font-size: 0.72rem;
    color: #6a6460;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8b84b;
}

.metric-value-secondary {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #a0998a;
}

.insight-card {
    background: #13131a;
    border: 1px solid #2a2a35;
    border-left: 3px solid #e8b84b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.insight-title {
    font-size: 0.75rem;
    color: #e8b84b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 0.3rem;
}

.insight-text {
    font-size: 0.92rem;
    color: #c8c0b0;
    line-height: 1.5;
}

.suggestion-card {
    background: #13131a;
    border: 1px solid #2a2a35;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}

.suggestion-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.suggestion-icon { font-size: 1.1rem; }

.suggestion-title {
    font-size: 0.8rem;
    font-weight: 500;
    color: #e8e4dc;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.suggestion-text {
    font-size: 0.88rem;
    color: #8a8070;
    line-height: 1.55;
}

.tag {
    display: inline-block;
    background: #1e1e28;
    border: 1px solid #3a3a48;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    color: #8a8070;
    margin-right: 0.3rem;
    margin-top: 0.3rem;
}

.tag-green {
    background: #0f1f12;
    border-color: #2a5c30;
    color: #5cb868;
}

.tag-amber {
    background: #1f1a08;
    border-color: #5c480a;
    color: #e8b84b;
}

.tag-red {
    background: #1f0e0e;
    border-color: #5c1a1a;
    color: #e85c5c;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4a4840;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e28;
}

.stButton > button {
    background: #e8b84b !important;
    color: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #f0c85c !important;
}

div[data-testid="stForm"] {
    background: #0f0f18;
    border: 1px solid #1e1e28;
    border-radius: 14px;
    padding: 1.5rem;
}

label { color: #8a8070 !important; font-size: 0.82rem !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider {
    background: #0a0a0f !important;
}

hr { border-color: #1e1e28 !important; }

.watermark {
    font-size: 0.68rem;
    color: #3a3830;
    text-align: center;
    margin-top: 2rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

def fmt_revenue(n):
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    return f"${n/1e6:.1f}M"

def roi_color(roi):
    if roi >= 3:   return "tag-green"
    if roi >= 1.5: return "tag-amber"
    return "tag-red"

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# Genre median revenues from Stage 2 EDA
GENRE_MEDIANS = {
    "Animation": 197, "Adventure": 129, "Action": 89,
    "Thriller": 60,   "Comedy": 57,     "Horror": 43,
    "Romance": 42,    "Drama": 38
}

# Genre median ROI from Stage 2
GENRE_ROI = {
    "Horror": 2.9,    "Animation": 2.8, "Adventure": 2.4,
    "Romance": 2.4,   "Comedy": 2.3,    "Thriller": 2.2,
    "Action": 2.15,   "Drama": 2.05
}

MONTH_MEDIANS = {
    1:36, 2:50, 3:65, 4:47, 5:81,
    6:112, 7:91, 8:43, 9:27, 10:41,
    11:85, 12:95
}

def generate_suggestions(budget_m, genres, is_franchise,
                          release_month, predicted_m, low_m, high_m):
    suggestions = []
    roi = predicted_m / budget_m if budget_m > 0 else 0

    # ── Release timing ────────────────────────────────────────────────────
    current_window = MONTH_MEDIANS.get(release_month, 50)
    best_month = max(MONTH_MEDIANS, key=MONTH_MEDIANS.get)
    best_val   = MONTH_MEDIANS[best_month]

    if release_month in [9, 10, 1, 4]:
        alt_months = sorted(MONTH_MEDIANS, key=MONTH_MEDIANS.get, reverse=True)[:3]
        alt_names  = [MONTH_NAMES[m-1] for m in alt_months]
        suggestions.append({
            "icon": "📅",
            "title": "Reconsider release timing",
            "text": (
                f"{MONTH_NAMES[release_month-1]} has a median box office of "
                f"${current_window}M — one of the weakest windows of the year. "
                f"Moving to {', '.join(alt_names)} could lift revenue "
                f"by {int((best_val/current_window - 1)*100)}% based on historical patterns."
            ),
            "urgency": "high"
        })
    elif release_month in [6, 7, 12]:
        suggestions.append({
            "icon": "✅",
            "title": "Strong release window",
            "text": (
                f"{MONTH_NAMES[release_month-1]} is one of the highest-grossing months "
                f"(${current_window}M median). You're targeting the right window — "
                f"make sure marketing spend is front-loaded to win opening weekend."
            ),
            "urgency": "low"
        })

    # ── Budget vs predicted ROI ───────────────────────────────────────────
    if roi < 1.5 and budget_m > 80:
        suggestions.append({
            "icon": "⚠️",
            "title": "Budget-to-revenue risk",
            "text": (
                f"At ${budget_m:.0f}M budget, the model predicts ${predicted_m:.0f}M — "
                f"an ROI of {roi:.2f}×. Studios typically need 2.5–3× to break even "
                f"after P&A spend. Consider trimming production budget by 15–20% "
                f"or boosting pre-release marketing to lift opening weekend share."
            ),
            "urgency": "high"
        })
    elif roi >= 3:
        suggestions.append({
            "icon": "💰",
            "title": "Healthy ROI profile",
            "text": (
                f"Predicted ROI of {roi:.2f}× on a ${budget_m:.0f}M budget is strong. "
                f"This film has the financial profile of a profitable release. "
                f"Protect this margin by keeping production costs disciplined."
            ),
            "urgency": "low"
        })

    # ── Genre strategy ────────────────────────────────────────────────────
    if genres:
        best_genre    = max(genres, key=lambda g: GENRE_MEDIANS.get(g, 0))
        best_genre_m  = GENRE_MEDIANS.get(best_genre, 50)
        best_roi_g    = max(genres, key=lambda g: GENRE_ROI.get(g, 0))

        if "Horror" in genres and budget_m > 30:
            suggestions.append({
                "icon": "🎃",
                "title": "Horror works best lean",
                "text": (
                    f"Horror has the highest median ROI of any genre (2.9×) — but that's "
                    f"driven by low-budget productions. At ${budget_m:.0f}M, you're above "
                    f"the sweet spot. The most profitable Horror films keep budgets under $20M. "
                    f"Consider whether the scale justifies the genre."
                ),
                "urgency": "medium"
            })

        if "Animation" in genres:
            suggestions.append({
                "icon": "🎨",
                "title": "Animation demands franchise thinking",
                "text": (
                    f"Animation has the highest median revenue ($197M) but also the "
                    f"highest budgets. Standalone animated films underperform sequels "
                    f"by nearly 2×. If this is an original IP, build in sequel hooks "
                    f"and merchandise strategy from day one."
                ),
                "urgency": "medium"
            })

        if len(genres) >= 3:
            suggestions.append({
                "icon": "🎭",
                "title": "Genre clarity drives marketing",
                "text": (
                    f"You've selected {len(genres)} genres. Films with 1–2 clear genres "
                    f"are easier to market with a single positioning statement. "
                    f"Identify your primary genre ({best_genre} leads on revenue) "
                    f"and use others as secondary descriptors only."
                ),
                "urgency": "medium"
            })

    # ── Franchise leverage ────────────────────────────────────────────────
    if not is_franchise and predicted_m < 80:
        suggestions.append({
            "icon": "🔗",
            "title": "Build franchise potential",
            "text": (
                f"Franchise films earn a 1.95× median premium over standalones. "
                f"Even for an original story, structuring the ending to allow a sequel "
                f"— and signalling this to distributors — can improve greenlight "
                f"chances and marketing positioning."
            ),
            "urgency": "medium"
        })

    if is_franchise and predicted_m > 150:
        suggestions.append({
            "icon": "🚀",
            "title": "Franchise premium is working",
            "text": (
                f"The franchise flag is contributing significantly to this prediction. "
                f"Protect the brand: consistent tone, returning cast, and strong "
                f"opening weekend are critical — franchise audiences reward loyalty "
                f"but punish quality drops with sharp sequel drop-offs."
            ),
            "urgency": "low"
        })

    # ── Uncertainty advice ────────────────────────────────────────────────
    uncertainty = high_m / low_m
    if uncertainty > 6:
        suggestions.append({
            "icon": "📊",
            "title": "High prediction uncertainty",
            "text": (
                f"The confidence range is ${low_m:.0f}M – ${high_m:.0f}M "
                f"({uncertainty:.1f}× spread). This is wider than typical, "
                f"suggesting the film sits in an unusual combination of features. "
                f"Run sensitivity analysis on release month and budget — "
                f"small changes may have outsized impact."
            ),
            "urgency": "medium"
        })

    return suggestions


def make_comparison_chart(genres, budget_m, predicted_m):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5),
                             facecolor='#0a0a0f')

    # Chart 1 — genre revenue context
    ax1 = axes[0]
    ax1.set_facecolor('#0f0f18')
    sorted_genres = sorted(GENRE_MEDIANS.items(), key=lambda x: x[1])
    g_names = [g for g,v in sorted_genres]
    g_vals  = [v for g,v in sorted_genres]
    colors  = ['#e8b84b' if g in genres else '#2a2a35' for g in g_names]

    bars = ax1.barh(g_names, g_vals, color=colors, height=0.6)
    ax1.axvline(predicted_m, color='#e8b84b', linestyle='--',
                linewidth=1, alpha=0.6, label=f'Your prediction ${predicted_m:.0f}M')
    ax1.set_xlabel('Median Revenue ($M)', color='#6a6460', fontsize=8)
    ax1.set_title('Genre revenue context', color='#8a8070',
                  fontsize=9, pad=8, loc='left')
    ax1.tick_params(colors='#6a6460', labelsize=8)
    ax1.spines[:].set_color('#2a2a35')
    ax1.legend(fontsize=7, labelcolor='#8a8070',
               facecolor='#0f0f18', edgecolor='#2a2a35')

    # Chart 2 — monthly revenue calendar
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f18')
    months = list(range(1, 13))
    vals   = [MONTH_MEDIANS[m] for m in months]
    month_colors = []
    for m in months:
        if m == 6:           month_colors.append('#e8b84b')
        elif m in [6,7,8]:   month_colors.append('#d4a030')
        elif m in [11,12]:   month_colors.append('#c87832')
        else:                month_colors.append('#2a2a35')

    # re-color with release_month highlight handled outside
    month_colors = []
    for m in months:
        if m in [6, 7, 8]:    month_colors.append('#3a4020')
        elif m in [11, 12]:   month_colors.append('#3a2a10')
        else:                  month_colors.append('#2a2a35')

    ax2.bar(months, vals, color=month_colors, width=0.7)
    ax2.set_xticks(months)
    ax2.set_xticklabels(MONTH_NAMES, fontsize=7, color='#6a6460')
    ax2.set_ylabel('Median Revenue ($M)', color='#6a6460', fontsize=8)
    ax2.set_title('Seasonal revenue pattern', color='#8a8070',
                  fontsize=9, pad=8, loc='left')
    ax2.tick_params(colors='#6a6460', labelsize=8)
    ax2.spines[:].set_color('#2a2a35')

    summer_patch  = mpatches.Patch(color='#3a4020', label='Summer')
    holiday_patch = mpatches.Patch(color='#3a2a10', label='Holiday')
    ax2.legend(handles=[summer_patch, holiday_patch],
               fontsize=7, labelcolor='#8a8070',
               facecolor='#0f0f18', edgecolor='#2a2a35')

    plt.tight_layout(pad=1.5)
    return fig


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
  <div class="hero-sub">AI · Box Office Intelligence</div>
  <div class="hero-title">CinecastTM</div>
  <p style="color:#6a6460; font-size:0.9rem; margin-top:0.5rem; max-width:520px;">
    Predict opening weekend revenue before a single frame is shot.
    Built on 3,200+ films · XGBoost · R² = 0.80
  </p>
</div>
<hr/>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="section-label">Film parameters</div>',
                unsafe_allow_html=True)

    with st.form("predict_form"):
        title = st.text_input("Film title", "Untitled Project")

        c1, c2 = st.columns(2)
        with c1:
            budget = st.number_input("Budget ($M)", 1.0, 500.0, 50.0, step=5.0)
        with c2:
            runtime = st.slider("Runtime (min)", 70, 210, 110)

        c3, c4 = st.columns(2)
        with c3:
            popularity = st.slider("TMDB popularity", 1.0, 100.0, 15.0)
        with c4:
            director_score = st.number_input("Director score", 10.0, 25.0, 17.5,
                                              help="Median log-revenue of director's past films. Leave 17.5 if unknown.")

        c5, c6 = st.columns(2)
        with c5:
            release_month = st.selectbox(
                "Release month",
                list(range(1, 13)),
                format_func=lambda m: MONTH_NAMES[m-1],
                index=5
            )
        with c6:
            release_dow = st.selectbox(
                "Release day",
                [0,1,2,3,4,5,6],
                format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d],
                index=4
            )

        is_franchise = st.radio(
            "Franchise / sequel?",
            [0, 1],
            format_func=lambda x: "Yes — sequel or franchise entry" if x else "No — original / standalone",
            horizontal=True
        )

        genres = st.multiselect(
            "Genres",
            ["Action","Comedy","Drama","Thriller",
             "Animation","Horror","Romance","Adventure"],
            default=["Action"]
        )

        submitted = st.form_submit_button("Run prediction →")


# ── Right column — results ────────────────────────────────────────────────────
with right_col:
    if not submitted:
        st.markdown("""
        <div style="height:200px; display:flex; align-items:center;
                    justify-content:center; color:#3a3830; font-size:0.85rem;
                    letter-spacing:0.1em; text-transform:uppercase;">
            Fill in parameters and run prediction
        </div>
        """, unsafe_allow_html=True)

    else:
        budget_dollars = budget * 1e6

        payload = {
            "title":          title,
            "budget":         budget_dollars,
            "runtime":        runtime,
            "popularity":     popularity,
            "is_franchise":   is_franchise,
            "director_score": director_score,
            "release_month":  release_month,
            "release_dow":    release_dow,
            "genres":         genres,
        }

        with st.spinner("Running model..."):
            try:
                r   = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                res = r.json()

                predicted_m = res['predicted_revenue'] / 1e6
                low_m       = res['low_estimate']      / 1e6
                high_m      = res['high_estimate']     / 1e6
                roi         = predicted_m / budget if budget > 0 else 0

                # ── Prediction metrics ────────────────────────────────────
                st.markdown('<div class="section-label">Revenue forecast</div>',
                            unsafe_allow_html=True)

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Low estimate</div>
                        <div class="metric-value-secondary">{fmt_revenue(res['low_estimate'])}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">{fmt_revenue(res['predicted_revenue'])}</div>
                    </div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">High estimate</div>
                        <div class="metric-value-secondary">{fmt_revenue(res['high_estimate'])}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ROI + quick tags
                roi_cls = roi_color(roi)
                is_summer_flag  = release_month in [6,7,8]
                is_holiday_flag = release_month in [11,12]

                tag_html = f'<span class="tag {roi_cls}">ROI {roi:.2f}×</span>'
                tag_html += f'<span class="tag {"tag-green" if is_franchise else "tag"}">{"Franchise" if is_franchise else "Standalone"}</span>'
                if is_summer_flag:
                    tag_html += '<span class="tag tag-green">Summer window</span>'
                if is_holiday_flag:
                    tag_html += '<span class="tag tag-amber">Holiday window</span>'
                for g in genres:
                    tag_html += f'<span class="tag">{g}</span>'

                st.markdown(f'<div style="margin-bottom:1.2rem">{tag_html}</div>',
                            unsafe_allow_html=True)

                # ── Key insights ──────────────────────────────────────────
                st.markdown('<div class="section-label">Model insights</div>',
                            unsafe_allow_html=True)

                # Budget efficiency
                best_genre_rev = max([GENRE_MEDIANS.get(g, 0) for g in genres], default=50)
                insight_budget = (
                    f"Your ${budget:.0f}M budget predicts {predicted_m/budget:.1f}× return. "
                    f"Historical median for {genres[0] if genres else 'this genre'} films "
                    f"is ${best_genre_rev}M."
                ) if genres else f"${budget:.0f}M budget → {predicted_m/budget:.1f}× predicted return."

                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">💵 Budget efficiency</div>
                    <div class="insight-text">{insight_budget}</div>
                </div>""", unsafe_allow_html=True)

                # Timing insight
                month_perf = MONTH_MEDIANS.get(release_month, 50)
                best_m_val = max(MONTH_MEDIANS.values())
                timing_pct = int((month_perf / best_m_val) * 100)
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">📅 Release window strength</div>
                    <div class="insight-text">
                        {MONTH_NAMES[release_month-1]} ranks at {timing_pct}% of peak month (June).
                        Historical median for this month: ${month_perf}M.
                        {"Summer premium adds ~1.5× vs off-peak." if is_summer_flag else
                         "Holiday window adds ~1.7× vs weakest months." if is_holiday_flag else
                         "Consider shifting to Jun–Jul or Nov–Dec for a revenue lift."}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Franchise insight
                franchise_text = (
                    "Franchise films earn a 1.95× median premium over standalones in this dataset. "
                    "This premium is already baked into the prediction."
                    if is_franchise else
                    "As a standalone, you're competing without the franchise safety net. "
                    "Strong opening weekend marketing is critical — there's no sequel halo effect."
                )
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">🔗 Franchise effect</div>
                    <div class="insight-text">{franchise_text}</div>
                </div>""", unsafe_allow_html=True)

                # ── Charts ────────────────────────────────────────────────
                st.markdown('<div class="section-label">Market context</div>',
                            unsafe_allow_html=True)

                fig = make_comparison_chart(genres, budget, predicted_m)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ── Filmmaker suggestions ─────────────────────────────────
                st.markdown('<div class="section-label">Filmmaker recommendations</div>',
                            unsafe_allow_html=True)

                suggestions = generate_suggestions(
                    budget, genres, is_franchise,
                    release_month, predicted_m, low_m, high_m
                )

                if not suggestions:
                    st.markdown("""
                    <div class="insight-card">
                        <div class="insight-title">✅ No major risk flags</div>
                        <div class="insight-text">
                            This film's parameters are well-optimised across budget,
                            timing, and genre. Focus on execution quality and
                            front-loaded marketing spend.
                        </div>
                    </div>""", unsafe_allow_html=True)

                for s in suggestions:
                    border_color = {
                        "high":   "#e85c5c",
                        "medium": "#e8b84b",
                        "low":    "#5cb868"
                    }.get(s.get("urgency","medium"), "#e8b84b")

                    st.markdown(f"""
                    <div class="suggestion-card" style="border-left: 3px solid {border_color};">
                        <div class="suggestion-header">
                            <span class="suggestion-icon">{s['icon']}</span>
                            <span class="suggestion-title">{s['title']}</span>
                        </div>
                        <div class="suggestion-text">{s['text']}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Scenario comparison ───────────────────────────────────
                st.markdown('<div class="section-label">What-if scenarios</div>',
                            unsafe_allow_html=True)

                sc1, sc2, sc3 = st.columns(3)
                # Scenario: move to June
                june_boost = MONTH_MEDIANS[6] / MONTH_MEDIANS.get(release_month, 50)
                june_pred  = predicted_m * june_boost

                # Scenario: make it a franchise
                franchise_boost = 1.95 if not is_franchise else 1.0
                franchise_pred  = predicted_m * franchise_boost

                # Scenario: cut budget 20%
                budget_cut_pred = predicted_m * 0.88  # approximate log-linear effect

                with sc1:
                    delta = june_pred - predicted_m
                    color = "#5cb868" if delta > 0 else "#e85c5c"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">If released in June</div>
                        <div class="metric-value-secondary">{fmt_revenue(june_pred*1e6)}</div>
                        <div style="font-size:0.75rem; color:{color}; margin-top:0.3rem;">
                            {"+" if delta>0 else ""}{fmt_revenue(delta*1e6)}
                        </div>
                    </div>""", unsafe_allow_html=True)

                with sc2:
                    delta2 = franchise_pred - predicted_m
                    color2 = "#5cb868" if delta2 > 0 else "#8a8070"
                    label2 = "If franchise entry" if not is_franchise else "Already franchise"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label2}</div>
                        <div class="metric-value-secondary">{fmt_revenue(franchise_pred*1e6)}</div>
                        <div style="font-size:0.75rem; color:{color2}; margin-top:0.3rem;">
                            {"+" if delta2>0 else ""}{fmt_revenue(delta2*1e6)}
                        </div>
                    </div>""", unsafe_allow_html=True)

                with sc3:
                    delta3 = budget_cut_pred - predicted_m
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">If budget cut 20%</div>
                        <div class="metric-value-secondary">{fmt_revenue(budget_cut_pred*1e6)}</div>
                        <div style="font-size:0.75rem; color:#e85c5c; margin-top:0.3rem;">
                            {fmt_revenue(delta3*1e6)}
                        </div>
                    </div>""", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure FastAPI is running: `uvicorn main:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("""
<div class="watermark">
    CinecastTM · XGBoost R²=0.80 · 3,213 films · TMDB dataset · for educational use
</div>
""", unsafe_allow_html=True)