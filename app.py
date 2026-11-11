import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile

st.set_page_config(page_title="Benford & KS Analysis Dashboard", layout="wide")

st.title("Benford & KS Analysis Dashboard")
st.write(
    "Upload a CSV or Excel file, filter the data, compare the actual distribution "
    "with Benford's Law, and compare two time periods with a KS statistic."
)


# -------------------------
# Functions
# -------------------------

def load_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, low_memory=False)
    return pd.read_excel(uploaded_file)


def numeric_version(series):
    return pd.to_numeric(series, errors="coerce")


def prepare_benford_amounts(series):
    amounts = numeric_version(series).dropna().abs()
    return amounts[amounts > 0]


def benford_analysis(series, first_two=False):
    amounts = prepare_benford_amounts(series)

    if len(amounts) == 0:
        return pd.DataFrame(), 0

    normalized = amounts / (10 ** np.floor(np.log10(amounts)))

    if first_two:
        digits = (normalized * 10).astype(int)
        digit_range = np.arange(10, 100)
    else:
        digits = normalized.astype(int)
        digit_range = np.arange(1, 10)

    frequency = digits.value_counts().reindex(digit_range, fill_value=0).sort_index()
    actual = frequency / frequency.sum()
    expected = np.log10(1 + (1 / digit_range))

    result = pd.DataFrame({
        "Digit": digit_range,
        "Frequency": frequency.values,
        "Actual %": actual.values * 100,
        "Benford %": expected * 100
    })

    result["Difference"] = result["Actual %"] - result["Benford %"]

    return result, len(amounts)


def benford_ks(result, sample_size):
    actual = result["Actual %"].to_numpy() / 100
    expected = result["Benford %"].to_numpy() / 100

    actual_cdf = np.cumsum(actual)
    expected_cdf = np.cumsum(expected)

    ks_stat = np.max(np.abs(actual_cdf - expected_cdf))
    critical_value = 1.36 / np.sqrt(sample_size)

    return ks_stat, critical_value, ks_stat > critical_value


def two_sample_ks(sample1, sample2):
    sample1 = numeric_version(sample1).dropna().to_numpy()
    sample2 = numeric_version(sample2).dropna().to_numpy()

    if len(sample1) == 0 or len(sample2) == 0:
        return None, None, None

    sample1 = np.sort(sample1)
    sample2 = np.sort(sample2)

    combined = np.unique(np.concatenate([sample1, sample2]))

    cdf1 = np.searchsorted(sample1, combined, side="right") / len(sample1)
    cdf2 = np.searchsorted(sample2, combined, side="right") / len(sample2)

    ks_stat = np.max(np.abs(cdf1 - cdf2))
    critical_value = 1.36 * np.sqrt(
        (len(sample1) + len(sample2)) / (len(sample1) * len(sample2))
    )

    return ks_stat, critical_value, ks_stat > critical_value


def figure_to_png(fig):
    image = BytesIO()
    fig.savefig(image, format="png", bbox_inches="tight", dpi=150)
    image.seek(0)
    return image.getvalue()


def make_results_zip(summary, first_digit, first_two, analysis_text,
                     first_png, first_two_png, period_png=None):
    output = BytesIO()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary_statistics.csv", summary.to_csv(index=False))
        z.writestr("first_digit_benford.csv", first_digit.to_csv(index=False))
        z.writestr("first_two_digit_benford.csv", first_two.to_csv(index=False))
        z.writestr("analysis.txt", analysis_text)
        z.writestr("first_digit_chart.png", first_png)
        z.writestr("first_two_digit_chart.png", first_two_png)

        if period_png is not None:
            z.writestr("time_period_ks_chart.png", period_png)

    output.seek(0)
    return output.getvalue()


# -------------------------
# Upload data
# -------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"],
    max_upload_size=500
)

if uploaded_file is None:
    st.info("Upload a file to begin the analysis.")
    st.stop()

try:
    data = load_file(uploaded_file)
except Exception as e:
    st.error(f"The file could not be loaded: {e}")
    st.stop()

st.success(f"Loaded {len(data):,} rows and {len(data.columns):,} columns.")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(25), use_container_width=True)


# -------------------------
# Select analysis columns
# -------------------------

st.sidebar.header("Analysis Settings")

columns = data.columns.tolist()

amount_matches = [
    c for c in columns
    if any(word in str(c).lower() for word in ["amount", "cost", "price", "value", "total"])
]
amount_index = columns.index(amount_matches[0]) if amount_matches else 0

amount_col = st.sidebar.selectbox(
    "Amount column",
    columns,
    index=amount_index
)

date_matches = [
    c for c in columns
    if any(word in str(c).lower() for word in ["date", "time"])
]
date_options = ["None"] + columns
date_index = date_options.index(date_matches[0]) if date_matches else 0

date_col = st.sidebar.selectbox(
    "Date column",
    date_options,
    index=date_index,
    help="A date column is needed for the two-period KS comparison."
)

working = data.copy()
working[amount_col] = numeric_version(working[amount_col])
working = working.dropna(subset=[amount_col])

if date_col != "None":
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")


# -------------------------
# Three filtering criteria
# -------------------------

st.sidebar.header("Filters")

# Filter 1: Amount range
st.sidebar.subheader("1. Amount Range")

amount_min = float(working[amount_col].min())
amount_max = float(working[amount_col].max())

col1, col2 = st.sidebar.columns(2)

with col1:
    selected_min = st.number_input(
        "Minimum",
        value=amount_min,
        format="%.2f"
    )

with col2:
    selected_max = st.number_input(
        "Maximum",
        value=amount_max,
        format="%.2f"
    )

filtered = working[
    (working[amount_col] >= selected_min) &
    (working[amount_col] <= selected_max)
].copy()


# Filter 2: Date range
st.sidebar.subheader("2. Date Range")

if date_col != "None" and filtered[date_col].notna().any():
    valid_dates = filtered[date_col].dropna()
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    selected_dates = st.sidebar.date_input(
        "Select dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates

        filtered = filtered[
            (filtered[date_col].dt.date >= start_date) &
            (filtered[date_col].dt.date <= end_date)
        ].copy()
else:
    st.sidebar.info("Select a valid date column to use the date filter.")


# Filter 3: Category
st.sidebar.subheader("3. Category")

category_candidates = [
    c for c in columns
    if c not in [amount_col, date_col]
]

if category_candidates:
    category_col = st.sidebar.selectbox(
        "Category column",
        category_candidates
    )

    category_counts = (
        filtered[category_col]
        .dropna()
        .astype(str)
        .value_counts()
    )

    if len(category_counts) > 500:
        category_options = category_counts.head(500).index.tolist()
        st.sidebar.caption("Showing the 500 most common values.")
    else:
        category_options = category_counts.index.tolist()

    selected_categories = st.sidebar.multiselect(
        "Values to include",
        category_options
    )

    if selected_categories:
        filtered = filtered[
            filtered[category_col].astype(str).isin(selected_categories)
        ].copy()
else:
    category_col = None
    selected_categories = []
    st.sidebar.info("No additional category column is available.")


# -------------------------
# Filter results and summary
# -------------------------

st.header("Filtered Data")

c1, c2, c3 = st.columns(3)
c1.metric("Rows after filtering", f"{len(filtered):,}")
c2.metric("Total amount", f"${filtered[amount_col].sum():,.2f}")
c3.metric("Average amount", f"${filtered[amount_col].mean():,.2f}")

summary = pd.DataFrame({
    "Statistic": [
        "Count",
        "Sum",
        "Mean",
        "Median",
        "Standard Deviation",
        "Minimum",
        "25th Percentile",
        "75th Percentile",
        "Maximum"
    ],
    "Value": [
        filtered[amount_col].count(),
        filtered[amount_col].sum(),
        filtered[amount_col].mean(),
        filtered[amount_col].median(),
        filtered[amount_col].std(),
        filtered[amount_col].min(),
        filtered[amount_col].quantile(.25),
        filtered[amount_col].quantile(.75),
        filtered[amount_col].max()
    ]
})

st.subheader("Summary Statistics")
st.dataframe(summary, use_container_width=True)


# -------------------------
# Benford analysis
# -------------------------

st.header("Benford Analysis")

first_digit, first_n = benford_analysis(filtered[amount_col], first_two=False)
first_two, first_two_n = benford_analysis(filtered[amount_col], first_two=True)

if first_n == 0:
    st.warning("There are no nonzero numeric amounts available for Benford analysis.")
    st.stop()

first_ks, first_critical, first_reject = benford_ks(first_digit, first_n)
two_ks, two_critical, two_reject = benford_ks(first_two, first_two_n)

tab1, tab2 = st.tabs(["First Digit", "First Two Digits"])


with tab1:
    st.subheader("First-Digit Distribution")

    st.dataframe(
        first_digit.round({
            "Actual %": 4,
            "Benford %": 4,
            "Difference": 4
        }),
        use_container_width=True
    )

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    x1 = np.arange(len(first_digit))
    width = 0.38

    ax1.bar(
        x1 - width / 2,
        first_digit["Actual %"],
        width,
        label="Actual"
    )
    ax1.bar(
        x1 + width / 2,
        first_digit["Benford %"],
        width,
        label="Benford"
    )

    ax1.set_title("Actual vs. Benford First-Digit Distribution")
    ax1.set_xlabel("First Digit")
    ax1.set_ylabel("Percent")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(first_digit["Digit"])
    ax1.legend()

    st.pyplot(fig1, use_container_width=True)
    first_png = figure_to_png(fig1)
    plt.close(fig1)

    st.write(f"**KS statistic:** {first_ks:.6f}")
    st.write(f"**5% critical value:** {first_critical:.6f}")

    if first_reject:
        st.write(
            "**Result:** The actual first-digit distribution is statistically "
            "different from the Benford distribution."
        )
    else:
        st.write(
            "**Result:** The actual first-digit distribution is not statistically "
            "different from the Benford distribution."
        )


with tab2:
    st.subheader("First-Two-Digit Distribution")

    st.dataframe(
        first_two.round({
            "Actual %": 4,
            "Benford %": 4,
            "Difference": 4
        }),
        use_container_width=True
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(
        first_two["Digit"],
        first_two["Actual %"],
        label="Actual"
    )
    ax2.plot(
        first_two["Digit"],
        first_two["Benford %"],
        label="Benford"
    )

    ax2.set_title("Actual vs. Benford First-Two-Digit Distribution")
    ax2.set_xlabel("First Two Digits")
    ax2.set_ylabel("Percent")
    ax2.legend()

    st.pyplot(fig2, use_container_width=True)
    first_two_png = figure_to_png(fig2)
    plt.close(fig2)

    st.write(f"**KS statistic:** {two_ks:.6f}")
    st.write(f"**5% critical value:** {two_critical:.6f}")

    if two_reject:
        st.write(
            "**Result:** The actual first-two-digit distribution is statistically "
            "different from the Benford distribution."
        )
    else:
        st.write(
            "**Result:** The actual first-two-digit distribution is not statistically "
            "different from the Benford distribution."
        )


# -------------------------
# Largest Benford differences
# -------------------------

st.subheader("Largest Benford Differences")

top_first = (
    first_digit.assign(Absolute_Difference=first_digit["Difference"].abs())
    .sort_values("Absolute_Difference", ascending=False)
    .head(3)
)

top_two = (
    first_two.assign(Absolute_Difference=first_two["Difference"].abs())
    .sort_values("Absolute_Difference", ascending=False)
    .head(5)
)

left, right = st.columns(2)

with left:
    st.write("**First digits with the largest differences**")
    st.dataframe(
        top_first[["Digit", "Actual %", "Benford %", "Difference"]].round(4),
        use_container_width=True
    )

with right:
    st.write("**First-two-digit values with the largest differences**")
    st.dataframe(
        top_two[["Digit", "Actual %", "Benford %", "Difference"]].round(4),
        use_container_width=True
    )


# -------------------------
# Two-period KS comparison
# -------------------------

st.header("Two-Time-Period KS Comparison")

period_png = None
period_analysis = "Two-period KS comparison was not available."

if date_col == "None":
    st.warning("Select a date column in the sidebar to compare two time periods.")

elif filtered[date_col].dropna().empty:
    st.warning("No valid dates remain after filtering.")

else:
    available_dates = filtered[date_col].dropna()
    overall_min = available_dates.min().date()
    overall_max = available_dates.max().date()

    midpoint = overall_min + ((overall_max - overall_min) / 2)

    default_p1_start = overall_min
    default_p1_end = midpoint

    if midpoint < overall_max:
        default_p2_start = midpoint + pd.Timedelta(days=1)
    else:
        default_p2_start = midpoint

    default_p2_end = overall_max

    p1_col, p2_col = st.columns(2)

    with p1_col:
        st.write("**Period 1**")
        p1_start = st.date_input(
            "Start date",
            value=default_p1_start,
            min_value=overall_min,
            max_value=overall_max,
            key="p1_start"
        )
        p1_end = st.date_input(
            "End date",
            value=default_p1_end,
            min_value=overall_min,
            max_value=overall_max,
            key="p1_end"
        )

    with p2_col:
        st.write("**Period 2**")
        p2_start = st.date_input(
            "Start date",
            value=default_p2_start,
            min_value=overall_min,
            max_value=overall_max,
            key="p2_start"
        )
        p2_end = st.date_input(
            "End date",
            value=default_p2_end,
            min_value=overall_min,
            max_value=overall_max,
            key="p2_end"
        )

    period1 = filtered[
        (filtered[date_col].dt.date >= p1_start) &
        (filtered[date_col].dt.date <= p1_end)
    ][amount_col]

    period2 = filtered[
        (filtered[date_col].dt.date >= p2_start) &
        (filtered[date_col].dt.date <= p2_end)
    ][amount_col]

    st.write(
        f"Period 1 contains **{period1.count():,}** transactions and "
        f"Period 2 contains **{period2.count():,}** transactions."
    )

    period_ks, period_critical, period_reject = two_sample_ks(period1, period2)

    if period_ks is None:
        st.warning("Both time periods must contain numeric observations.")
    else:
        st.write(f"**KS statistic:** {period_ks:.6f}")
        st.write(f"**5% critical value:** {period_critical:.6f}")

        if period_reject:
            period_result = (
                "The KS statistic is larger than the critical value, so the null "
                "hypothesis is rejected. The two time periods are statistically different."
            )
        else:
            period_result = (
                "The KS statistic is not larger than the critical value, so the null "
                "hypothesis is not rejected. There is not enough evidence to conclude "
                "that the two time periods are statistically different."
            )

        st.write(f"**Result:** {period_result}")

        p1_values = np.sort(numeric_version(period1).dropna().to_numpy())
        p2_values = np.sort(numeric_version(period2).dropna().to_numpy())

        fig3, ax3 = plt.subplots(figsize=(9, 5))

        if len(p1_values) > 0:
            p1_index = np.linspace(
                0,
                len(p1_values) - 1,
                min(5000, len(p1_values))
            ).astype(int)
            ax3.plot(
                p1_values[p1_index],
                (p1_index + 1) / len(p1_values),
                label="Period 1"
            )

        if len(p2_values) > 0:
            p2_index = np.linspace(
                0,
                len(p2_values) - 1,
                min(5000, len(p2_values))
            ).astype(int)
            ax3.plot(
                p2_values[p2_index],
                (p2_index + 1) / len(p2_values),
                label="Period 2"
            )

        ax3.set_title("Cumulative Distribution by Time Period")
        ax3.set_xlabel("Transaction Amount")
        ax3.set_ylabel("Cumulative Proportion")
        ax3.legend()

        st.pyplot(fig3, use_container_width=True)
        period_png = figure_to_png(fig3)
        plt.close(fig3)

        period_analysis = (
            f"Period 1: {p1_start} to {p1_end}\n"
            f"Period 2: {p2_start} to {p2_end}\n"
            f"Period 1 observations: {period1.count():,}\n"
            f"Period 2 observations: {period2.count():,}\n"
            f"KS statistic: {period_ks:.6f}\n"
            f"5% critical value: {period_critical:.6f}\n"
            f"Conclusion: {period_result}"
        )


# -------------------------
# Download and share results
# -------------------------

st.header("Download Results")

first_result_text = (
    "statistically different from Benford"
    if first_reject
    else "not statistically different from Benford"
)

two_result_text = (
    "statistically different from Benford"
    if two_reject
    else "not statistically different from Benford"
)

analysis_text = f"""Benford & KS Analysis Dashboard

Uploaded file: {uploaded_file.name}
Rows after filtering: {len(filtered):,}
Amount column: {amount_col}
Date column: {date_col}

First-digit Benford analysis
KS statistic: {first_ks:.6f}
5% critical value: {first_critical:.6f}
Conclusion: The actual first-digit distribution is {first_result_text}.

First-two-digit Benford analysis
KS statistic: {two_ks:.6f}
5% critical value: {two_critical:.6f}
Conclusion: The actual first-two-digit distribution is {two_result_text}.

Two-time-period KS comparison
{period_analysis}
"""

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        "Download summary statistics",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="summary_statistics.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        "Download analysis",
        data=analysis_text.encode("utf-8"),
        file_name="analysis.txt",
        mime="text/plain"
    )

with download_col3:
    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.download_button(
        "Download first-digit chart",
        data=first_png,
        file_name="first_digit_chart.png",
        mime="image/png"
    )

with chart_col2:
    st.download_button(
        "Download first-two-digit chart",
        data=first_two_png,
        file_name="first_two_digit_chart.png",
        mime="image/png"
    )

with chart_col3:
    if period_png is not None:
        st.download_button(
            "Download time-period chart",
            data=period_png,
            file_name="time_period_ks_chart.png",
            mime="image/png"
        )

results_zip = make_results_zip(
    summary,
    first_digit,
    first_two,
    analysis_text,
    first_png,
    first_two_png,
    period_png
)

st.download_button(
    "Download all results",
    data=results_zip,
    file_name="benford_ks_results.zip",
    mime="application/zip"
)
