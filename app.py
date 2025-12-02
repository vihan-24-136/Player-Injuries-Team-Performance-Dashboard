# Streamlit app: Player Injuries & Team Performance Dashboard
# File: app.py
# Author: Generated for student project (IADAI102)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Player Injuries & Team Performance", layout="wide")

# ---------------------- Helper functions ----------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        # Create a small synthetic sample dataset so app still runs
        rng = pd.date_range("2023-01-01", periods=200, freq='D')
        players = ["A. Silva", "B. Kumar", "C. Jones", "D. Lee", "E. Patel"]
        clubs = ["Lions", "Tigers", "Eagles", "Sharks"]
        data = []
        for i in range(200):
            player = np.random.choice(players)
            club = np.random.choice(clubs)
            date = rng[i]
            rating = np.clip(6 + np.random.randn() * 1.2, 4, 9)
            injured = np.random.choice([0, 1], p=[0.95, 0.05])
            if injured:
                injury_len = np.random.randint(7, 50)
                injury_start = date
                injury_end = date + pd.Timedelta(days=injury_len)
            else:
                injury_start = pd.NaT
                injury_end = pd.NaT
            data.append({
                "match_date": date,
                "player_name": player,
                "club": club,
                "rating": rating,
                "goals": np.random.poisson(0.15),
                "injury_start": injury_start,
                "injury_end": injury_end,
                "age": np.random.randint(18, 36),
                "injury_id": np.random.choice([np.nan, "I-"+str(np.random.randint(1,200))], p=[0.9,0.1])
            })
        df = pd.DataFrame(data)
        return df
    else:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Unable to load file: {e}")
            return pd.DataFrame()


@st.cache_data
def preprocess(df):
    df = df.copy()
    # Standardize column names to lowercase for safety
    df.columns = [c.strip() for c in df.columns]

    # Try common column names if different
    # Guarantee required columns exist or create placeholders
    for col in ["match_date", "injury_start", "injury_end"]:
        if col not in df.columns:
            # attempt some common variants
            if col.replace('_', ' ') in df.columns:
                df[col] = pd.to_datetime(df[col.replace('_', ' ')])
            else:
                df[col] = pd.NaT

    # Convert date columns
    for date_col in ["match_date", "injury_start", "injury_end"]:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Ensure numeric columns
    for num_col in ["rating", "goals", "age"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
        else:
            df[num_col] = np.nan

    # Fill simple missing values
    df["club"] = df.get("club", pd.Series(["Unknown"]*len(df)))
    df["player_name"] = df.get("player_name", pd.Series(["Unknown"]*len(df)))

    # Compute injury flag for each match row: whether player was injured at that match_date
    df["injured_during_match"] = False
    mask = (~df["injury_start"].isna()) & (~df["injury_end"].isna())
    # If there are explicit injury windows, mark rows with match_date inside window
    for idx in df[mask].index:
        s = df.at[idx, "injury_start"]
        e = df.at[idx, "injury_end"]
        player = df.at[idx, "player_name"]
        # mark all other rows of same player that fall inside that window as injured
        player_mask = (df["player_name"] == player) & (df["match_date"].between(s, e))
        df.loc[player_mask, "injured_during_match"] = True

    # For players missing injury windows but with an injury_id, leave injured_during_match as False

    # Feature engineering: rolling averages before and after injury
    df.sort_values(["player_name", "match_date"], inplace=True)

    # Pre-injury avg rating (5-match window ending before injury start)
    df["pre_injury_avg"] = df.groupby("player_name")["rating"].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    # Post-injury avg rating (5-match window starting after injury end) - approximate by forward rolling
    df["post_injury_avg"] = df.groupby("player_name")["rating"].transform(lambda x: x.shift(-1).rolling(5, min_periods=1).mean())

    df["performance_drop"] = df["pre_injury_avg"] - df["post_injury_avg"]

    # Extract month and year for heatmap
    df["injury_month"] = df["match_date"].dt.month
    df["injury_year"] = df["match_date"].dt.year

    return df


# ---------------------- App UI ----------------------
st.title("⚽ Player Injuries & Team Performance Dashboard")
st.markdown(
    "Use this dashboard to explore how player injuries influence match ratings and team performance. Upload your dataset (CSV/XLSX) or use the sample data provided." 
)

with st.sidebar:
    st.header("Data & Filters")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)
    st.markdown("---")
    min_date = st.date_input("Start date (data window)", value=None)
    max_date = st.date_input("End date (data window)", value=None)
    club_filter = st.multiselect("Select clubs", options=[], default=None)
    player_filter = st.multiselect("Select players", options=[], default=None)
    age_range = st.slider("Player age range", 15, 45, (15, 40))
    st.markdown("---")
    st.write("Project: FootLens Analytics — Student Demo")

# Load data
df_raw = load_data(uploaded_file)
if df_raw.empty:
    st.warning("No data available. Upload a CSV/XLSX file with match records or modify the sample generator.")

# Preprocess
df = preprocess(df_raw)

# Populate filter options dynamically
all_clubs = sorted(df["club"].dropna().unique())
all_players = sorted(df["player_name"].dropna().unique())

# Update sidebar multiselects if they are empty
if not club_filter:
    club_filter = st.sidebar.multiselect("Select clubs", options=all_clubs, default=all_clubs)
else:
    # user may have already selected, keep it
    pass

if not player_filter:
    player_filter = st.sidebar.multiselect("Select players", options=all_players, default=all_players[:10])

# Apply filters
filter_mask = df["club"].isin(club_filter) & df["player_name"].isin(player_filter)
if min_date:
    filter_mask &= df["match_date"] >= pd.to_datetime(min_date)
if max_date:
    filter_mask &= df["match_date"] <= pd.to_datetime(max_date)
filter_mask &= df["age"].between(age_range[0], age_range[1])

df_filtered = df[filter_mask].copy()

st.sidebar.markdown(f"**Records:** {len(df_filtered):,}")

# ---------------------- KPIs ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_injuries = df_filtered["injury_id"].notna().sum()
    st.metric("Reported injury events", f"{total_injuries}")
with col2:
    avg_drop = df_filtered["performance_drop"].mean()
    st.metric("Avg. performance drop", f"{avg_drop:.2f}")
with col3:
    injured_matches = df_filtered["injured_during_match"].sum()
    st.metric("Matches with injured players", f"{injured_matches}")
with col4:
    players_affected = df_filtered["player_name"].nunique()
    st.metric("Players in view", f"{players_affected}")

st.markdown("---")

# ---------------------- Visual 1: Bar Chart - Top 10 injuries with highest team performance drop ----------------------
st.subheader("Top injuries by team performance drop")
impact_table = (
    df_filtered.dropna(subset=["injury_id"]).groupby(["injury_id", "player_name", "club"])["performance_drop"].mean().reset_index()
)
impact_table = impact_table.sort_values("performance_drop", ascending=False).head(10)
if not impact_table.empty:
    fig1 = px.bar(impact_table, x="player_name", y="performance_drop", color="club",
                  hover_data=["injury_id"], title="Top injuries causing team performance drop")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Not enough injury-event data to show impact chart.")

# ---------------------- Visual 2: Line Chart - Player performance timeline (before/after injury) ----------------------
st.subheader("Player performance timeline (select a player)")
player_choice = st.selectbox("Choose player for timeline", options=all_players, index=0)
player_df = df[df["player_name"] == player_choice].sort_values("match_date")

if not player_df.empty:
    fig2 = px.line(player_df, x="match_date", y="rating", title=f"Rating timeline: {player_choice}")
    # annotate injury windows
    injury_rows = player_df.dropna(subset=["injury_start", "injury_end"]).drop_duplicates(subset=["injury_start","injury_end"])
    for _, row in injury_rows.iterrows():
        fig2.add_vrect(x0=row["injury_start"], x1=row["injury_end"], fillcolor="red", opacity=0.15, layer="below", line_width=0)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No data for selected player.")

# ---------------------- Visual 3: Heatmap - Injury frequency by month and club ----------------------
st.subheader("Injury frequency: month vs club")
heat_data = df_filtered.copy()
heat_data = heat_data[heat_data["injury_id"].notna()]
if not heat_data.empty:
    pivot = heat_data.pivot_table(index="club", columns=heat_data["match_date"].dt.month, values="injury_id", aggfunc="count", fill_value=0)
    # Ensure months 1-12 present
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot.sort_index()
    fig3 = px.imshow(pivot, labels=dict(x="Month", y="Club", color="Injury count"), x=pivot.columns, y=pivot.index, aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No injuries tracked in filtered data to build heatmap.")

# ---------------------- Visual 4: Scatter - Age vs performance drop ----------------------
st.subheader("Age vs Performance Drop")
scatter_df = df_filtered.dropna(subset=["performance_drop", "age"]).groupby(["player_name", "age"]).agg(performance_drop=("performance_drop","mean"), occurrences=("performance_drop","count")).reset_index()
if not scatter_df.empty:
    fig4 = px.scatter(scatter_df, x="age", y="performance_drop", size="occurrences", hover_name="player_name", title="Age vs Avg. Performance Drop")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Insufficient data for scatter plot.")

# ---------------------- Visual 5: Leaderboard - Comeback players ----------------------
st.subheader("Comeback leaderboard: players with largest rating improvement after injury")
leaderboard = df_filtered.dropna(subset=["pre_injury_avg", "post_injury_avg"]).groupby("player_name").agg(pre_avg=("pre_injury_avg","mean"), post_avg=("post_injury_avg","mean"))
if not leaderboard.empty:
    leaderboard["improvement"] = leaderboard["post_avg"] - leaderboard["pre_avg"]
    lb = leaderboard.sort_values("improvement", ascending=False).reset_index().head(20)
    st.dataframe(lb.style.format({"pre_avg":"{:.2f}", "post_avg":"{:.2f}", "improvement":"{:.2f}"}))
else:
    st.info("No pre/post injury rolling averages computed; leaderboard unavailable.")

st.markdown("---")

# ---------------------- Additional EDA / Raw data ----------------------
st.header("Data Explorer & Export")
with st.expander("Show filtered data (first 200 rows)"):
    st.dataframe(df_filtered.head(200))

csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data as CSV", data=csv, file_name="filtered_injuries.csv", mime="text/csv")

st.markdown("---")
st.caption("This demo app generates a sample dataset when no upload is provided. For real analysis, upload a CSV/XLSX file with columns: match_date, player_name, club, rating, goals, injury_start, injury_end, age, injury_id.")

# ---------------------- End ----------------------

