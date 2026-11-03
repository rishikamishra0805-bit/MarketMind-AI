
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re

from src.data_utils import load_and_clean_data, build_rfm
from src.segmentation import run_segmentation
from src.recommendations import recommend_for_customer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f6fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11152e, #20264b);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] {
    background: white !important;
    border-radius: 10px;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #20264b !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: white !important;
    border: 2px dashed #a99bea !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
    color: #28245c !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: white !important;
    color: #28245c !important;
    border: 1px solid #a99bea !important;
    border-radius: 8px !important;
}

.hero {
    background: linear-gradient(120deg, #171b3d, #4435a5, #7655d9);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(45, 38, 120, 0.15);
}

.hero h1 {
    color: white !important;
    font-size: 34px !important;
    font-weight: 800 !important;
}

.hero p {
    color: #e3e4ff !important;
    font-size: 15px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 13px;
    border-radius: 30px;
    font-size: 12px;
    margin-bottom: 12px;
}

.section-heading {
    color: #20264b;
    font-size: 22px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #7b819b;
    font-size: 13px;
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e8e9f3;
    box-shadow: 0 5px 18px rgba(28, 32, 70, 0.04);
    min-height: 125px;
    overflow-wrap: anywhere;
}

.metric-label {
    color: #858aa4;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #20264b;
    font-size: 23px;
    font-weight: 800;
}

.metric-icon {
    font-size: 21px;
    margin-bottom: 9px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 16px;
    border-color: #e8e9f3;
}

.stDownloadButton button {
    background: linear-gradient(100deg, #5141b8, #7959dc);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 18px;
}

.stDownloadButton button:hover {
    background: #352a8c;
}

h2, h3 {
    color: #20264b !important;
}

.stButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REUSABLE FUNCTIONS
# --------------------------------------------------

def section_heading(title, subtitle=""):
    st.markdown(
        f'<div class="section-heading">{title}</div>'
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(icon, label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def normalize_optional_customer_name(df):
    """
    Recognizes optional customer name columns such as:
    CustomerName, Customer Name, customer_name.
    """
    df = df.copy()

    for col in df.columns:
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(col).lower()
        )

        if normalized in [
            "customername",
            "name",
            "customerfullname"
        ]:
            df = df.rename(columns={col: "CustomerName"})
            break

    if "CustomerName" in df.columns:
        df["CustomerName"] = (
            df["CustomerName"]
            .astype("string")
            .str.strip()
        )

    return df


def get_recommended_action(segment):
    actions = {
        "Champions": "Reward this customer with loyalty benefits and exclusive offers.",
        "Loyal Customers": "Encourage repeat purchases through membership and special offers.",
        "At-Risk Customers": "Try a win-back campaign with personalized discounts.",
        "Big Spenders": "Promote premium products and exclusive collections.",
        "Recent Customers": "Introduce related products and encourage a second purchase.",
        "Potential Customers": "Use attractive offers to build customer loyalty."
    }

    return actions.get(
        str(segment),
        "Use personalized offers to improve customer engagement."
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 20px 0;">
        <div style="font-size:34px;">🛍️</div>
        <div style="font-size:23px;font-weight:800;color:white;">
            MarketMind AI
        </div>
        <div style="font-size:11px;color:#c5c9e8;">
            CUSTOMER INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Select Dashboard",
        [
            "Dashboard Overview",
            "Customer Segmentation",
            "Customer Explorer",
            "Reports & About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📂 Data Source")

    data_choice = st.selectbox(
        "Choose Dataset",
        [
            "Real UCI Retail Dataset",
            "Synthetic Demo Dataset"
        ]
    )

    uploaded = st.file_uploader(
        "Upload your own CSV",
        type=["csv"],
        help="Upload your shop's retail transaction CSV."
    )

    st.caption(
        "Required columns: InvoiceNo, StockCode, Description, "
        "Quantity, InvoiceDate, UnitPrice, CustomerID. "
        "CustomerName and Country are optional."
    )

    st.divider()

    st.markdown("### 🤖 Model Settings")

    k = st.slider(
        "Number of customer groups",
        min_value=2,
        max_value=8,
        value=4
    )

    st.divider()

    st.markdown("""
    <div style="font-size:12px;color:#c5c9e8;">
        <b>Technology</b><br>
        Python · Streamlit · K-Means<br>
        RFM Analysis · Plotly
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

try:

    # 1. LOAD DATA

    with st.spinner("Loading customer transaction data..."):

        if uploaded is not None:

            raw = pd.read_csv(
                uploaded,
                encoding="utf-8-sig",
                encoding_errors="replace"
            )

            data_source = "Uploaded Shop Dataset"

        elif data_choice == "Real UCI Retail Dataset":

            dataset_path = Path("data/online_retail.csv")

            if not dataset_path.exists():
                st.error(
                    "Real dataset not found. "
                    "Check data/online_retail.csv."
                )
                st.stop()

            raw = pd.read_csv(dataset_path)

            data_source = "Real UCI Online Retail Dataset"

        else:

            dataset_path = Path("data/sample_transactions.csv")

            if not dataset_path.exists():
                st.error(
                    "Synthetic dataset not found. "
                    "Check data/sample_transactions.csv."
                )
                st.stop()

            raw = pd.read_csv(dataset_path)

            data_source = "Synthetic Demo Dataset"

    # 2. CLEAN DATA

    with st.spinner("Cleaning and preparing customer data..."):

        raw = normalize_optional_customer_name(raw)

        clean = load_and_clean_data(raw)

        clean = normalize_optional_customer_name(clean)

        rfm = build_rfm(clean)

    if clean.empty or rfm.empty:
        st.error(
            "No valid customer records found. "
            "Please check your CSV data."
        )
        st.stop()

    # 3. CUSTOMER NAME MAPPING

    if "CustomerName" in clean.columns:

        customer_names = (
            clean.groupby("CustomerID")["CustomerName"]
            .agg(
                lambda values: next(
                    (
                        str(v).strip()
                        for v in values
                        if pd.notna(v) and str(v).strip()
                    ),
                    None
                )
            )
            .to_dict()
        )

    else:

        customer_names = {}

    # If no customer name exists, use Customer ID as display name.

    customer_names = {
        str(cid): (
            str(name)
            if name and str(name).strip()
            else str(cid)
        )
        for cid, name in customer_names.items()
    }

    # 4. VALIDATE CLUSTER COUNT

    if len(rfm) <= k:
        st.error(
            f"Only {len(rfm)} customers are available. "
            "Please select fewer clusters."
        )
        st.stop()

    # 5. MACHINE LEARNING MODEL

    with st.spinner("Training K-Means segmentation model..."):

        segmented, model, scaler, score = run_segmentation(rfm, k)

    if segmented.empty:
        st.error("Customer segmentation failed.")
        st.stop()

    # 6. ADD CUSTOMER NAMES TO CUSTOMER REPORT

    segmented["CustomerName"] = (
        segmented["CustomerID"]
        .astype(str)
        .map(customer_names)
        .fillna(segmented["CustomerID"].astype(str))
    )

    # 7. SEGMENT SUMMARY

    summary = segmented.groupby(
        ["Cluster", "Segment"]
    ).agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Spending=("Monetary", "mean")
    ).reset_index()

    if summary.empty:
        st.error("Unable to generate segment summary.")
        st.stop()

    # --------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">✦ AI-POWERED CUSTOMER ANALYTICS</div>
        <h1>{page}</h1>
        <p>
            MarketMind AI – Intelligent Customer Segmentation
            & Recommendation System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"Data Source: {data_source} | "
        f"{len(clean):,} valid transactions | "
        f"{len(rfm):,} customers"
    )

    # --------------------------------------------------
    # PAGE 1: DASHBOARD OVERVIEW
    # --------------------------------------------------

    if page == "Dashboard Overview":

        section_heading(
            "📊 Business Overview",
            "Your retail business performance at a glance."
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "👥",
                "TOTAL CUSTOMERS",
                f"{len(rfm):,}"
            )

        with c2:
            metric_card(
                "🧾",
                "VALID TRANSACTIONS",
                f"{len(clean):,}"
            )

        with c3:
            metric_card(
                "💰",
                "TOTAL REVENUE",
                f"£{clean['Revenue'].sum():,.2f}"
            )

        with c4:
            metric_card(
                "🛒",
                "AVERAGE CUSTOMER SPEND",
                f"£{rfm['Monetary'].mean():,.2f}"
            )

        section_heading(
            "👥 Customer Distribution",
            "Number of customers in each segment."
        )

        with st.container(border=True):

            chart_data = summary.sort_values(
                "Customers",
                ascending=True
            )

            fig = px.bar(
                chart_data,
                x="Customers",
                y="Segment",
                color="Segment",
                orientation="h",
                text="Customers",
                color_discrete_sequence=[
                    "#7663D8",
                    "#38A6A5",
                    "#F2A65A",
                    "#E47791",
                    "#5986C7",
                    "#A47CC4"
                ]
            )

            fig.update_traces(textposition="outside")

            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=False,
                xaxis_title="Customers",
                yaxis_title=""
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False}
            )

        section_heading(
            "💡 Business Insights",
            "A quick look at customer segmentation."
        )

        top_segment = summary.loc[
            summary["Customers"].idxmax()
        ]

        high_value = summary.loc[
            summary["Avg_Spending"].idxmax()
        ]

        a, b = st.columns(2)

        with a:
            with st.container(border=True):
                st.markdown("#### 👑 Largest Customer Group")
                st.subheader(str(top_segment["Segment"]))
                st.write(
                    f"{int(top_segment['Customers']):,} customers "
                    "belong to this segment."
                )

        with b:
            with st.container(border=True):
                st.markdown("#### 💎 Highest Spending Group")
                st.subheader(str(high_value["Segment"]))
                st.write(
                    f"Average spending: "
                    f"£{high_value['Avg_Spending']:,.2f}"
                )

    # --------------------------------------------------
    # PAGE 2: CUSTOMER SEGMENTATION
    # --------------------------------------------------

    elif page == "Customer Segmentation":

        section_heading(
            "🤖 K-Means Model Performance",
            "Understand how the machine learning model groups customers."
        )

        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                st.metric(
                    "Silhouette Score",
                    f"{score:.3f}"
                )
                st.caption(
                    "Measures how well-separated the customer clusters are."
                )

        with c2:
            with st.container(border=True):
                st.metric("Number of Clusters", k)
                st.caption("Customer groups created by K-Means.")

        section_heading(
            "📈 Customer Analytics",
            "Explore customer purchasing patterns."
        )

        left, right = st.columns(2)

        with left:
            with st.container(border=True):

                st.markdown("#### Customers by Segment")

                fig = px.bar(
                    summary.sort_values("Customers"),
                    x="Customers",
                    y="Segment",
                    color="Segment",
                    orientation="h",
                    text="Customers",
                    color_discrete_sequence=[
                        "#7663D8",
                        "#38A6A5",
                        "#F2A65A",
                        "#E47791",
                        "#5986C7",
                        "#A47CC4"
                    ]
                )

                fig.update_traces(textposition="outside")

                fig.update_layout(
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False}
                )

        with right:
            with st.container(border=True):

                st.markdown(
                    "#### Spending vs Purchase Frequency"
                )

                fig = px.scatter(
                    segmented,
                    x="Frequency",
                    y="Monetary",
                    color="Segment",
                    hover_data=[
                        "CustomerID",
                        "CustomerName",
                        "Recency"
                    ],
                    log_y=True
                )

                fig.update_traces(
                    marker=dict(size=9, opacity=0.75)
                )

                fig.update_layout(
                    template="plotly_white",
                    height=400
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False}
                )

        section_heading(
            "👥 Customer Segment Profile",
            "Compare the average behaviour of each group."
        )

        with st.container(border=True):

            display_summary = summary[
                [
                    "Segment",
                    "Customers",
                    "Avg_Recency",
                    "Avg_Frequency",
                    "Avg_Spending"
                ]
            ].copy()

            display_summary.columns = [
                "Customer Segment",
                "Customers",
                "Average Recency (Days)",
                "Average Invoices",
                "Average Spending (£)"
            ]

            st.dataframe(
                display_summary.round(2),
                width="stretch",
                hide_index=True
            )

    # --------------------------------------------------
    # PAGE 3: CUSTOMER EXPLORER
    # --------------------------------------------------

    elif page == "Customer Explorer":

        section_heading(
            "🔍 Explore Individual Customers",
            "View customer details, purchase behaviour and recommendations."
        )

        customer_ids = sorted(
            segmented["CustomerID"]
            .astype(str)
            .unique()
            .tolist()
        )

        if not customer_ids:

            st.info("No customers are available.")

        else:

            chosen = st.selectbox(
                "Search and Select Customer",
                customer_ids,
                format_func=lambda cid: (
                    f"{customer_names.get(cid, cid)} ({cid})"
                )
            )

            selected_row = segmented[
                segmented["CustomerID"].astype(str) == chosen
            ].iloc[0]

            selected_name = customer_names.get(chosen, chosen)

            with st.container(border=True):

                st.markdown("### 👤 Customer Profile")

                st.markdown(
                    f"**Customer Name:** {selected_name}"
                )

                st.markdown(
                    f"**Customer ID:** {chosen}"
                )

                a, b, c, d = st.columns(4)

                with a:
                    metric_card(
                        "🏷️",
                        "CUSTOMER SEGMENT",
                        selected_row["Segment"]
                    )

                with b:
                    metric_card(
                        "📅",
                        "DAYS SINCE PURCHASE",
                        f"{int(selected_row['Recency']):,}"
                    )

                with c:
                    metric_card(
                        "🧾",
                        "TOTAL INVOICES",
                        f"{int(selected_row['Frequency']):,}"
                    )

                with d:
                    metric_card(
                        "💷",
                        "CUSTOMER SPENDING",
                        f"£{selected_row['Monetary']:,.2f}"
                    )

            # CUSTOMER BUSINESS ACTION

            section_heading(
                "💡 Suggested Business Action",
                "An action the shop can consider for this customer."
            )

            with st.container(border=True):

                st.info(
                    get_recommended_action(selected_row["Segment"])
                )

            # CUSTOMER PURCHASE HISTORY

            section_heading(
                "🧾 Purchase History",
                "Invoices and products purchased by this customer."
            )

            customer_transactions = clean[
                clean["CustomerID"].astype(str) == chosen
            ].copy()

            if not customer_transactions.empty:

                purchase_history = customer_transactions.groupby(
                    ["InvoiceNo", "InvoiceDate"]
                ).agg(
                    Products=("Description", lambda x: ", ".join(
                        sorted(set(x.astype(str)))
                    )),
                    Total_Quantity=("Quantity", "sum"),
                    Total_Amount=("Revenue", "sum")
                ).reset_index()

                purchase_history = purchase_history.sort_values(
                    "InvoiceDate",
                    ascending=False
                )

                st.dataframe(
                    purchase_history,
                    width="stretch",
                    hide_index=True
                )

            # PRODUCT RECOMMENDATIONS

            section_heading(
                "🎯 Product Recommendations",
                "Products suggested using purchase history and customer segment."
            )

            recs = recommend_for_customer(
                clean,
                chosen,
                segmented,
                limit=5
            )

            if recs:

                columns = st.columns(3)

                for i, product in enumerate(recs):

                    with columns[i % 3]:
                        with st.container(border=True):
                            st.caption("✨ RECOMMENDED PRODUCT")
                            st.markdown(f"**{str(product)}**")

            else:

                st.info(
                    "Not enough product history for this customer yet."
                )

            st.caption(
                "Recommendations use purchase history and popular "
                "products purchased by customers in the same K-Means cluster."
            )

    # --------------------------------------------------
    # PAGE 4: REPORTS & ABOUT
    # --------------------------------------------------

    elif page == "Reports & About":

        section_heading(
            "📥 Download Reports",
            "Export customer-level results and segment summaries."
        )

        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):

                st.markdown("### Customer Details and Segments")

                st.write(
                    "Download customer IDs, names, RFM values "
                    "and customer segments."
                )

                st.download_button(
                    "⬇️ Download Customer Report",
                    segmented.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="marketmind_customer_report.csv",
                    mime="text/csv",
                    width="stretch"
                )

        with c2:
            with st.container(border=True):

                st.markdown("### Segment Summary")

                st.write(
                    "Download the average behaviour and size of each group."
                )

                st.download_button(
                    "⬇️ Download Segment Summary",
                    summary.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="marketmind_segment_summary.csv",
                    mime="text/csv",
                    width="stretch"
                )

        section_heading(
            "ℹ️ About MarketMind AI",
            "Project information and technologies."
        )

        with st.container(border=True):

            st.markdown("### 🛍️ MarketMind AI")

            st.write(
                "MarketMind AI is an intelligent customer segmentation "
                "and product recommendation application designed "
                "to understand retail customer behaviour."
            )

            st.markdown("#### Main Features")

            st.markdown("""
            - Customer segmentation using K-Means clustering
            - RFM analysis (Recency, Frequency and Monetary)
            - Customer name and ID display when provided
            - Customer purchase history
            - Purchase-history-based product recommendations
            - Suggested business actions
            - Interactive data visualizations
            - Downloadable customer reports
            - Real and synthetic dataset support
            - Custom CSV upload
            """)

            st.markdown("#### Technologies Used")

            st.markdown("""
            - Python
            - Streamlit
            - Pandas and NumPy
            - Scikit-learn
            - K-Means Clustering
            - Plotly
            """)

            st.markdown("#### Dataset")

            st.write(
                "Public UCI Online Retail dataset and custom "
                "retail transaction CSV files."
            )

            st.markdown("#### Machine Learning Method")

            st.write(
                "K-Means is an unsupervised learning algorithm. "
                "RFM features are transformed and scaled before clustering. "
                "Silhouette Score evaluates cluster separation. "
                "Customer segment names are heuristic labels based "
                "on average RFM behaviour."
            )

            st.markdown("#### Recommendation Method")

            st.write(
                "The system first checks products in a customer's "
                "purchase history. It then suggests popular products "
                "from customers belonging to the same cluster."
            )

            st.markdown("#### Custom CSV Format")

            st.write(
                "Required columns: InvoiceNo, StockCode, Description, "
                "Quantity, InvoiceDate, UnitPrice and CustomerID. "
                "CustomerName and Country are optional."
            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.markdown("""
    <hr style="border:1px solid #e5e6f0;margin-top:35px;">

    <div style="
        text-align:center;
        color:#858aa4;
        font-size:12px;
        padding:10px;
    ">
        🛍️ <b>MarketMind AI</b><br>
        Intelligent Customer Segmentation & Recommendation System
        <br>
        Python · Streamlit · Machine Learning · Plotly
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# ERROR HANDLING
# --------------------------------------------------

except Exception as exc:

    st.error(
        "Something went wrong while processing the application."
    )

    st.exception(exc)

    st.info(
        "Please check your CSV columns, dates and numeric values."
    )
