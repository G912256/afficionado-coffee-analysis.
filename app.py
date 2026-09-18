"""
Afficionado Coffee Roasters - Sales Trend & Time-Based Performance Dashboard
===========================================================================
Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coffee_clean.csv"
RAW = ROOT / "data" / "Afficionado_Coffee_Roasters.xlsx"

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BUCKETS = ["Morning", "Afternoon", "Evening", "Late hours"]
BROWN = ["#6F4E37", "#C08552", "#2E5266", "#9B2226", "#4F772D"]
SERVICE_RATE_DEFAULT = 25

st.set_page_config(page_title="Afficionado Coffee Roasters - Time Analytics",
                   page_icon="☕", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
h1, h2, h3 {color: #4a3728;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- DATA LOADING
def bucket(h):
    if 6 <= h <= 11:
        return "Morning"
    if 12 <= h <= 16:
        return "Afternoon"
    if 17 <= h <= 21:
        return "Evening"
    return "Late hours"


def build_from_raw(path):
    """Rebuild the clean dataset from the raw workbook (date reconstruction included)."""
    df = pd.read_excel(path, sheet_name="Transactions").sort_values("transaction_id").reset_index(drop=True)
    secs = df["transaction_time"].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    df["_s"] = secs
    idx = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby("store_id"):
        idx.loc[g.index] = (g["_s"].diff().fillna(0) < 0).cumsum().values
    df["transaction_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(idx, unit="D")
    ts = df["transaction_date"] + pd.to_timedelta(df["transaction_time"].astype(str))
    df["revenue"] = df.transaction_qty * df.unit_price
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_name()
    df["day_of_week_num"] = ts.dt.dayofweek
    df["is_weekend"] = df.day_of_week_num >= 5
    df["month"] = ts.dt.month
    df["month_name"] = ts.dt.month_name()
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["time_bucket"] = df.hour.map(bucket)
    return df.drop(columns=["_s"])


@st.cache_data(show_spinner="Loading transaction data…")
def load_data():
    if DATA.exists():
        df = pd.read_csv(DATA, parse_dates=["transaction_date"])
    elif RAW.exists():
        df = build_from_raw(RAW)
    else:
        st.error(f"No data found. Place coffee_clean.csv in {DATA.parent} "
                 f"or the raw workbook at {RAW}.")
        st.stop()
    return df


df = load_data()

# -------------------------------------------------------------------- SIDEBAR
st.sidebar.title("☕ Filters")
st.sidebar.caption("Every chart on every tab responds to these controls.")

stores = sorted(df.store_location.unique())
sel_stores = st.sidebar.multiselect("Store location", stores, default=stores)

sel_days = st.sidebar.multiselect("Day of week", DOW, default=DOW)

hr_min, hr_max = int(df.hour.min()), int(df.hour.max())
sel_hours = st.sidebar.slider("Hour range", hr_min, hr_max, (hr_min, hr_max))

dmin, dmax = df.transaction_date.min().date(), df.transaction_date.max().date()
sel_dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(sel_dates, (list, tuple)) and len(sel_dates) == 2:
    d_from, d_to = [pd.Timestamp(d) for d in sel_dates]
else:
    d_from, d_to = pd.Timestamp(dmin), pd.Timestamp(dmax)

cats = sorted(df.product_category.unique())
sel_cats = st.sidebar.multiselect("Product category", cats, default=cats)

st.sidebar.markdown("---")
metric = st.sidebar.radio("Metric", ["Revenue", "Quantity", "Transactions"], horizontal=False)
norm = st.sidebar.checkbox("Normalise store comparison (% of store total)", value=True,
                           help="Removes size differences so demand *shapes* can be compared.")

MET = {"Revenue": ("revenue", "sum", "Revenue (USD)", "$,.0f"),
       "Quantity": ("transaction_qty", "sum", "Units sold", ",.0f"),
       "Transactions": ("transaction_id", "count", "Transactions", ",.0f")}
col, how, label, fmt = MET[metric]

mask = (df.store_location.isin(sel_stores) & df.day_of_week.isin(sel_days)
        & df.hour.between(*sel_hours) & df.transaction_date.between(d_from, d_to)
        & df.product_category.isin(sel_cats))
f = df[mask]

if f.empty:
    st.warning("No transactions match the current filters. Widen the selection.")
    st.stop()


def agg(frame, by):
    return frame.groupby(by).agg(value=(col, how)).reset_index()


n_days = f.transaction_date.nunique()

# --------------------------------------------------------------------- HEADER
st.title("Afficionado Coffee Roasters")
st.markdown("#### Sales Trend and Time-Based Performance Analysis · Jan–Jun 2025")

k = st.columns(5)
k[0].metric("Revenue", f"${f.revenue.sum():,.0f}")
k[1].metric("Transactions", f"{len(f):,}")
k[2].metric("Units sold", f"{int(f.transaction_qty.sum()):,}")
k[3].metric("Avg ticket", f"${f.revenue.mean():,.2f}")
k[4].metric("Avg revenue / day", f"${f.revenue.sum()/max(n_days,1):,.0f}")

tabs = st.tabs(["📈 Sales trend", "📅 Day of week", "🕐 Hourly demand",
                "🗺️ Location comparison", "👥 Staffing planner", "🔎 Data & method"])

# ------------------------------------------------------------ TAB 1 · TRENDS
with tabs[0]:
    st.subheader("Overall sales trend")
    daily = agg(f, "transaction_date").sort_values("transaction_date")
    daily["ma7"] = daily.value.rolling(7).mean()
    x = np.arange(len(daily))
    slope, intercept = np.polyfit(x, daily.value, 1) if len(daily) > 1 else (0, daily.value.mean())

    fig = go.Figure()
    fig.add_scatter(x=daily.transaction_date, y=daily.value, name="Daily",
                    line=dict(color="#D6BFA6", width=1))
    fig.add_scatter(x=daily.transaction_date, y=daily.ma7, name="7-day moving avg",
                    line=dict(color=BROWN[0], width=3))
    fig.add_scatter(x=daily.transaction_date, y=intercept + slope * x, name="Linear trend",
                    line=dict(color=BROWN[3], width=2, dash="dash"))
    fig.update_layout(height=420, yaxis_title=label, xaxis_title=None,
                      legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        mon = f.groupby(["month", "month_name"]).agg(value=(col, how)).reset_index()
        mon["MoM %"] = mon.value.pct_change() * 100
        fig = px.bar(mon, x="month_name", y="value", text_auto=".3s",
                     color_discrete_sequence=[BROWN[0]], title="Monthly totals")
        fig.update_layout(height=340, yaxis_title=label, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with c2:
        wk = f.groupby(["store_location", "week_of_year"]).agg(value=(col, how)).reset_index()
        fig = px.line(wk, x="week_of_year", y="value", color="store_location", markers=True,
                      color_discrete_sequence=BROWN, title="Weekly trend by store")
        fig.update_layout(height=340, yaxis_title=label, xaxis_title="ISO week",
                          legend=dict(orientation="h", y=-0.25, title=None))
        st.plotly_chart(fig, width="stretch")

    growth = (mon.value.iloc[-1] / mon.value.iloc[0] - 1) * 100 if len(mon) > 1 else 0
    st.info(f"**Trend:** {slope:+,.1f} {label.lower()} per day on the fitted line · "
            f"first-to-last month change **{growth:+.1f}%**.")

# -------------------------------------------------------- TAB 2 · DAY OF WEEK
with tabs[1]:
    st.subheader("Day-of-week performance")
    per_day = f.groupby(["transaction_date", "day_of_week"]).agg(value=(col, how)).reset_index()
    d = (per_day.groupby("day_of_week").value.agg(["mean", "std", "count"])
         .reindex([x for x in DOW if x in per_day.day_of_week.values]).reset_index())
    d.columns = ["day_of_week", "avg", "std", "n_days"]
    d["se"] = d["std"] / np.sqrt(d.n_days)

    fig = go.Figure(go.Bar(x=d.day_of_week, y=d.avg, marker_color=BROWN[0],
                           error_y=dict(type="data", array=1.96 * d.se, color="#555", width=4)))
    fig.add_hline(y=d.avg.mean(), line_dash="dash", line_color="grey",
                  annotation_text="overall mean")
    lo, hi = d.avg.min(), d.avg.max()
    fig.update_layout(height=400, yaxis_title=f"Avg {label.lower()} per day",
                      yaxis_range=[lo * 0.9, hi * 1.06], margin=dict(t=30),
                      title="Average per day, with 95% confidence intervals")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([2, 1])
    with c1:
        we = f.groupby("is_weekend").agg(value=(col, how))
        nd = f.groupby("is_weekend").transaction_date.nunique()
        cmp_df = pd.DataFrame({"Segment": ["Weekday (Mon–Fri)", "Weekend (Sat–Sun)"],
                               "Avg per day": [(we.value / nd).get(False, np.nan),
                                               (we.value / nd).get(True, np.nan)]})
        fig = px.bar(cmp_df, x="Segment", y="Avg per day", text_auto=".4s",
                     color="Segment", color_discrete_sequence=[BROWN[0], BROWN[1]])
        fig.update_layout(height=320, showlegend=False, yaxis_title=f"Avg {label.lower()} per day",
                          title="Weekday vs weekend")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Ranking**")
        rank = d[["day_of_week", "avg"]].sort_values("avg", ascending=False).reset_index(drop=True)
        rank.index += 1
        rank.columns = ["Day", f"Avg {metric.lower()}"]
        st.dataframe(rank.style.format({f"Avg {metric.lower()}": "{:,.0f}"}),
                     width="stretch")
        spread = (d.avg.max() / d.avg.min() - 1) * 100
        st.metric("Best vs worst day gap", f"{spread:.1f}%")
        if spread < 10:
            st.caption("Gap is small — day of week is a weak scheduling lever here. "
                       "See the Hourly tab for the strong one.")

# ------------------------------------------------------- TAB 3 · HOURLY DEMAND
with tabs[2]:
    st.subheader("Time-of-day demand")
    h = agg(f, "hour")
    h["per_day"] = h.value / max(n_days, 1)
    peak = int(h.loc[h.value.idxmax(), "hour"])
    colors = [BROWN[3] if hh == peak else BROWN[0] for hh in h.hour]

    fig = go.Figure(go.Bar(x=h.hour, y=h.value, marker_color=colors))
    fig.update_layout(height=380, xaxis_title="Hour of day", yaxis_title=label,
                      xaxis=dict(dtick=1), title=f"Hourly {metric.lower()} — peak hour: {peak}:00",
                      margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        piv = f.pivot_table(index="day_of_week", columns="hour", values=col, aggfunc=how)
        piv = piv.reindex([x for x in DOW if x in piv.index])
        nd = f.groupby("day_of_week").transaction_date.nunique().reindex(piv.index)
        piv = piv.div(nd.values, axis=0)
        fig = px.imshow(piv, color_continuous_scale="YlOrBr", aspect="auto",
                        labels=dict(color=f"Avg {metric.lower()}"),
                        title="Heatmap · day × hour (average per day)")
        fig.update_layout(height=380, xaxis_title="Hour", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with c2:
        b = f.groupby("time_bucket").agg(value=(col, how)).reindex(
            [x for x in BUCKETS if x in f.time_bucket.values]).reset_index()
        b["share"] = b.value / b.value.sum() * 100
        fig = px.pie(b, names="time_bucket", values="value", hole=.45,
                     color_discrete_sequence=["#6F4E37", "#C08552", "#E0C097", "#2E5266"],
                     title="Share by time bucket")
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

    top3 = h.nlargest(3, "value")
    st.info(f"**Peak window:** hours {', '.join(f'{int(x)}:00' for x in sorted(top3.hour))} carry "
            f"**{top3.value.sum()/h.value.sum()*100:.1f}%** of all {metric.lower()} in the current selection.")

# ---------------------------------------------------- TAB 4 · LOCATIONS
with tabs[3]:
    st.subheader("Cross-location temporal comparison")
    s = agg(f, "store_location").sort_values("value", ascending=False)
    cols_ = st.columns(len(s))
    for cc, (_, r) in zip(cols_, s.iterrows()):
        cc.metric(r.store_location, f"{r.value:,.0f}" if metric != "Revenue" else f"${r.value:,.0f}")

    sh = f.pivot_table(index="hour", columns="store_location", values=col, aggfunc=how)
    plot_sh = (sh / sh.sum() * 100) if norm else sh
    ytitle = "% of store total" if norm else label
    fig = go.Figure()
    for i, cname in enumerate(plot_sh.columns):
        fig.add_scatter(x=plot_sh.index, y=plot_sh[cname], name=cname, mode="lines+markers",
                        line=dict(color=BROWN[i % len(BROWN)], width=2.5))
    fig.update_layout(height=400, xaxis_title="Hour of day", yaxis_title=ytitle,
                      xaxis=dict(dtick=1), title="Hourly demand profile by store",
                      legend=dict(orientation="h", y=1.12, title=None), margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        sd = f.pivot_table(index="day_of_week", columns="store_location", values=col, aggfunc=how)
        sd = sd.reindex([x for x in DOW if x in sd.index])
        nd = f.groupby(["day_of_week", "store_location"]).transaction_date.nunique().unstack().reindex(sd.index)
        sd = sd / nd
        fig = px.bar(sd.reset_index().melt(id_vars="day_of_week"), x="day_of_week", y="value",
                     color="store_location", barmode="group", color_discrete_sequence=BROWN,
                     title="Day-of-week profile by store")
        fig.update_layout(height=360, yaxis_title=f"Avg {label.lower()} per day", xaxis_title=None,
                          legend=dict(orientation="h", y=-0.3, title=None))
        st.plotly_chart(fig, width="stretch")
    with c2:
        bs = f.pivot_table(index="time_bucket", columns="store_location", values=col, aggfunc=how)
        bs = bs.reindex([x for x in BUCKETS if x in bs.index])
        bs = bs / bs.sum() * 100
        fig = px.bar(bs.reset_index().melt(id_vars="time_bucket"), x="store_location", y="value",
                     color="time_bucket", color_discrete_sequence=["#6F4E37", "#C08552", "#E0C097"],
                     title="Share of store demand by time bucket (%)")
        fig.update_layout(height=360, yaxis_title="% of store total", xaxis_title=None,
                          legend=dict(orientation="h", y=-0.3, title=None))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Per-store hourly heatmaps**")
    for sname in sorted(f.store_location.unique()):
        sub = f[f.store_location == sname]
        p = sub.pivot_table(index="day_of_week", columns="hour", values=col, aggfunc=how)
        p = p.reindex([x for x in DOW if x in p.index])
        nd = sub.groupby("day_of_week").transaction_date.nunique().reindex(p.index)
        p = p.div(nd.values, axis=0)
        fig = px.imshow(p, color_continuous_scale="YlOrBr", aspect="auto", title=sname,
                        labels=dict(color=f"Avg {metric.lower()}"))
        fig.update_layout(height=260, xaxis_title=None, yaxis_title=None, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------- TAB 5 · STAFFING
with tabs[4]:
    st.subheader("Demand-based staffing planner")
    st.caption("Converts observed transaction load into baristas required on the floor. "
               "Rosters are forward-looking, so the default calibration month is the latest one.")

    c1, c2, c3 = st.columns(3)
    rate = c1.slider("Transactions served per barista-hour", 10, 50, SERVICE_RATE_DEFAULT)
    months = sorted(f.month_name.unique(), key=lambda m: f[f.month_name == m].month.iloc[0])
    cal = c2.selectbox("Calibration month", months, index=len(months) - 1)
    pctile = c3.select_slider("Size the roster for", [50, 75, 90, 95], value=90,
                              format_func=lambda p: f"the {p}th-percentile day")

    base = f[f.month_name == cal]
    if base.empty:
        st.warning("No data for that month under the current filters.")
    else:
        per = base.groupby(["store_location", "hour", "transaction_date"]).size().rename("txns").reset_index()
        a = per.groupby(["store_location", "hour"]).txns.agg(
            mean_txns="mean", target=lambda s: s.quantile(pctile / 100)).reset_index()
        a["baristas"] = np.ceil(a.target / rate).clip(lower=1)
        roster = a.pivot(index="hour", columns="store_location", values="baristas")

        st.markdown(f"**Baristas required on the floor · {cal} 2025 · {pctile}th-percentile day · {rate} txn/hr**")
        st.dataframe(roster.style.format("{:.0f}", na_rep="closed")
                     .background_gradient(cmap="YlOrBr", axis=None), width="stretch")

        demand_h = a.groupby("store_location").baristas.sum()
        flat_h = a.groupby("store_location").baristas.max() * a.groupby("store_location").hour.count()
        comp = pd.DataFrame({"Demand-based barista-hours/day": demand_h,
                             "Flat peak-level staffing": flat_h,
                             "Reduction %": ((1 - demand_h / flat_h) * 100).round(1)})
        st.dataframe(comp.style.format({"Demand-based barista-hours/day": "{:.0f}",
                                        "Flat peak-level staffing": "{:.0f}",
                                        "Reduction %": "{:.1f}%"}), width="stretch")
        st.success(f"Matching the roster to the demand curve instead of staffing flat at peak level "
                   f"saves **{(1 - demand_h.sum()/flat_h.sum())*100:.1f}%** of barista-hours "
                   f"while still covering the {pctile}th-percentile day.")
        st.caption("Assumption: a barista serves ~{} transactions per hour. Adjust the slider to "
                   "match Afficionado's measured service time.".format(rate))

# ------------------------------------------------------- TAB 6 · DATA & METHOD
with tabs[5]:
    st.subheader("Data quality and methodology")
    q = pd.DataFrame({
        "Check": ["Rows loaded", "Missing values", "Duplicate transaction IDs",
                  "Non-positive quantity", "Non-positive unit price", "Distinct trading days",
                  "Stores", "Date range"],
        "Result": [f"{len(df):,}", f"{int(df.isna().sum().sum()):,}",
                   f"{int(df.transaction_id.duplicated().sum()):,}",
                   f"{int((df.transaction_qty <= 0).sum()):,}",
                   f"{int((df.unit_price <= 0).sum()):,}",
                   f"{df.transaction_date.nunique():,}", f"{df.store_location.nunique()}",
                   f"{df.transaction_date.min().date()} → {df.transaction_date.max().date()}"]})
    st.table(q)

    st.markdown("""
**Calendar reconstruction.** The source workbook ships `year` and `transaction_time` but no
calendar date, which would make day-of-week and daily-trend analysis impossible. The ledger is
written in chronological order within each store, so every point where `transaction_time` steps
backwards marks a store closing and reopening — i.e. a day boundary. Counting those boundaries
returns exactly **181 trading days for each of the three stores**, with no store missing a day,
and the day index is monotonic in `transaction_id` across the whole file. A continuous 181-day
calendar starting 1 January 2025 ends on 30 June 2025, which is the period used throughout.

**Derived fields.** `revenue = transaction_qty × unit_price`; hour of day; day of week;
weekend flag; ISO week; month; and time buckets — Morning 06–11, Afternoon 12–16,
Evening 17–21, Late hours 22–05.
    """)

    st.markdown("**Filtered sample (first 300 rows)**")
    show = ["transaction_id", "transaction_date", "day_of_week", "transaction_time", "hour",
            "store_location", "product_category", "product_type", "transaction_qty",
            "unit_price", "revenue"]
    st.dataframe(f[[c for c in show if c in f.columns]].head(300), width="stretch")
    st.download_button("⬇️ Download filtered data (CSV)",
                       f.to_csv(index=False).encode(), "afficionado_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Afficionado Coffee Roasters · Sales Trend and Time-Based Performance Analysis · "
           "149,116 transactions · 3 stores · 1 Jan – 30 Jun 2025")
