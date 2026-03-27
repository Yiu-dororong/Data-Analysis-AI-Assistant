import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from graph.workflow import app  
import streamlit.user_info
import os
# Temporarily patch the function to raise an error and find the culprit
def raise_exception_instead_of_warning(*args, **kwargs):
    raise Exception("Deprecated user function called!")

streamlit.user_info.maybe_show_deprecated_user_warning = raise_exception_instead_of_warning

st.set_page_config(layout="wide")
init_streamlit_comm()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(SCRIPT_DIR, "test")
st.title("Data Analysis AI Assistant")
with st.expander("📖 How to Use This App "):
    st.markdown("""
        Welcome to your End-to-End AI Data Auditor. To get the most accurate insights from your data, please follow these guidelines:
        #### Properties:
        * **General Purpose Analysis**: This tool is designed for general data exploration and is not domain-specific. It generates standard visualizations (Bar, Line, Scatter, etc.) but does not currently support specialized chart types such as OHLC (Open-High-Low-Close) for financial markets or complex biological heatmaps.
        
        #### Data Requirements:
        * Minimum Volume: Please ensure your file contains at least 5 rows of valid, non-duplicate, non-empty data to allow the AI to identify patterns.
        * File Context: Use a clear, descriptive filename (e.g., q4_sales_report.csv instead of data1.csv).
        * Column Names: Ensure your columns have meaningful headers (e.g., Transaction_Date instead of Col_1). The AI uses these names to understand the context of your business metrics.
        #### Workflow:
                
        (1) Upload your CSV or select a Kaggle Test Sample.
                
        (2) Review the Data Preview to ensure everything looks correct.
                
        (3) Click Confirm & Start to trigger the multi-node AI pipeline.
                
        (4) Use the Stop button at any time to halt the process.
                
        """)
    st.warning("""
        ⚠️ Disclaimer: AI-Assisted Analysis \n
        This application utilizes Large Language Models (LLMs) to process and interpret your data.

        * Accuracy: While the AI is designed to be rigorous, it may occasionally produce incorrect insights, hallucinate trends, or suggest suboptimal visualizations.
        * Verification: Always cross-reference AI-generated findings with your original dataset. This tool is intended to assist your analysis, not replace human judgment.
        * No Professional Advice: The outputs of this app do not constitute financial, legal, or professional advice.
        """)
    
NODE_INFO = {
    "preprocessing": {
        "label": "🩺 Dataset Cleaner", 
        "desc": "Cleaned dataset and checked for data inconsistencies.",
        "next": "Working on context enrichment..."
    },
    "context_enrichment": {
        "label": "🎨 Chart Architect", 
        "desc": "Identified context and target metrics.",
        "next": "Proceeding to feature engineering..."
    },
    "feature_engineering": {
        "label": "🔨 Feature Engineer", 
        "desc": "Attempted to engineer new features to improve data quality.",
        "next": "Planning the design of charts..."
    },
    "visualisation_planning": {
        "label": "📑 Visualisation Planner", 
        "desc": "Drafted serveral charts and expected outcomes.",
        "next": "Analysing the charts..."
    },
    "visionary_analyst": {
        "label": "⚖️ Visionary Analyst",
        "desc": "Examined charts",
        "next": "Reviewing the insights..."
    },
    "executive_summary": {
        "label": "📖 Executive Summary",
        "desc": "Preparing final summary.",
        "next": "Analysis completed"
    },
    "simple_summary": {
        "label": "📋 Simple Summary",
        "desc": "No valid charts were made. Generating simple summary.",
        "next": "Analysis completed"
    },
    "error_node": {
        "label": "❌ Error",
        "desc": "Error occurred.",
        "next": "Anaylsis ended. Please check the error message."
    },
}

car2ord ={1:"1st",2:"2nd",3:"3rd",4:"4th",5:"5th"}
kaggle_dataset_info={
    "customer_subscription_churn_usage_patterns.csv":{
        "source":"https://www.kaggle.com/datasets/jayjoshi37/customer-subscription-churn-and-usage-patterns",
        "sample_work": "https://www.kaggle.com/code/lukhilaksh/customer-churn-eda-and-model-train",
        "description": "This synthetic dataset tracks usage, payments, and engagement for subscription-based services like SaaS or OTT platforms. It is designed for churn prediction and machine learning tasks, helping to identify why customers leave and how to improve retention."  
        },
    "world-happiness-report-2021.csv":{
        "source":"https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021",
        "sample_work": "https://www.kaggle.com/code/sejalkshirsagar/world-happiness-report-2021",
        "description": "The World Happiness Report uses Gallup World Poll data to analyze global well-being, ranking countries based on self-reported life evaluations and six key factors: GDP, social support, life expectancy, freedom, generosity, and corruption. "
        },
    "Depression Professional Dataset.csv":{
        "source":"https://www.kaggle.com/datasets/ikynahidwin/depression-professional-dataset",
        "sample_work": "https://www.kaggle.com/code/nourhanwael7/eda-3-models-on-depression-98",
        "description": "This dataset examines the impact of demographics, lifestyle, and workplace conditions on mental health. Featuring indicators like job satisfaction, stress levels, and history of illness, it is designed for EDA and predictive modeling to identify risk factors and understand the influence of work-life balance on mental well-being."
        }
    
}
# Initialize state flags
if "workflow_started" not in st.session_state:
    st.session_state.workflow_started = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = None
if "df" not in st.session_state:
    st.session_state.df = None
if "pyg_specs" not in st.session_state:
    st.session_state.pyg_specs = None
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0
if "raw_pngs" not in st.session_state:
    st.session_state.raw_pngs = None
if "show_advanced_viz" not in st.session_state:
    st.session_state.show_advanced_viz = False
if "has_error" not in st.session_state:
    st.session_state.has_error = False
if "outlier_infos" not in st.session_state:
    st.session_state.outlier_infos = None
if "secondary" not in st.session_state:
    st.session_state.secondary = []
if "filename" not in st.session_state:
    st.session_state.filename = None
if "null_and_dup" not in st.session_state:
    st.session_state.null_and_dup = None

# 1. Upload & Preview
selected_sample = None
uploaded_file = None

if not st.session_state.workflow_started:
    use_demo = st.checkbox("🧪 Use Local Test Samples (Kaggle)")
    if use_demo:
        # List all CSVs in the /test directory
        if os.path.exists(TEST_DATA_DIR):
            sample_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.csv')]
            
            if sample_files:
                # Let the user pick which sample to run
                selected_sample = st.selectbox(
                    " ## Select a Kaggle Dataset:", 
                    sample_files,
                    help="Source: [Browse Kaggle Datasets](https://www.kaggle.com)"
                )
                
                # Load the selected local file
                file_path = os.path.join(TEST_DATA_DIR, selected_sample)
                df = pd.read_csv(file_path)
                st.success(f"📂 Loaded: `{selected_sample}`")
                st.info(f'{kaggle_dataset_info.get(selected_sample).get("description")}')
                col1, col2 = st.columns(2)
                col1.link_button(label="Source",url=kaggle_dataset_info.get(selected_sample).get("source"))
                col2.link_button(label="Sample Work",url=kaggle_dataset_info.get(selected_sample).get("sample_work"))
            else:
                st.error("No CSV files found in /test folder.")
                df = None
        else:
            st.error(f"Directory `{TEST_DATA_DIR}` not found.")
            df = None
    else:
        # 3. Standard Upload Mode
        uploaded_file = st.file_uploader("Upload your own CSV", type="csv",max_upload_size=100)
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = None

if (selected_sample or uploaded_file) and not st.session_state.workflow_started:
    st.write("### 📋 Data Preview")
    st.dataframe(df.head()) # Show preview
    st.session_state.df = df

    if st.button("✅ Confirm & Start Analysis"):
        st.session_state.workflow_started = True
        st.session_state.stop_requested = False
        st.session_state.filename = uploaded_file.name if uploaded_file else selected_sample
        st.rerun() # Refresh to start the workflow loop

if st.session_state.workflow_started:
    if st.session_state.filename in kaggle_dataset_info:
        col1, col2 = st.columns(2)
        col1.link_button(label="Source", url=kaggle_dataset_info.get(st.session_state.filename).get("source"))
        col2.link_button(label="Sample Work", url=kaggle_dataset_info.get(st.session_state.filename).get("sample_work"))

    # 2. Add Stop Button
    stop_button_placeholder = st.empty()
    if stop_button_placeholder.button("🛑 Stop Workflow"):
        st.session_state.stop_requested = True
        st.session_state.workflow_started = False
        st.warning("Stopping workflow...")
        st.rerun()

    # 3. Controlled Streaming Loop
    with st.status("🚀 Running AI Pipeline...", expanded=True) as status:
        
        # We pass the DF directly into the initial state
        initial_state = {
            "df": st.session_state.df, 
            "filename": st.session_state.filename,
        }
       
        # Use .stream() to catch the stop flag between nodes
        for chunk in app.stream(initial_state, stream_mode="updates"):
            
            # CHECK FOR STOP SIGNAL
            if st.session_state.stop_requested:
                status.update(label="❌ Workflow Aborted", state="error")
                st.info("Process stopped by user.")
                break # Exit the loop immediately

            for node_name, output in chunk.items():
                # Update the status label to show current activity
                status.update(label=f"{NODE_INFO.get(node_name).get('desc')} {'(' + car2ord.get(st.session_state.retry_count + 1) + ' time).' if node_name == 'visionary_analyst' else ''} {NODE_INFO.get(node_name).get('next')}", state="running")

                if node_name == "preprocessing":
                    st.write("### 🧹 Data Preprocessing")
                    st.session_state.null_and_dup = output.get("null_and_dup")
                    st.write(f"Removed {output.get('null_and_dup')[0]} Nulls and {output.get('null_and_dup')[1]} Duplicates.") 
                
                elif node_name == "context_enrichment":
                    st.write("### 💭 Context Enrichment")
                    st.write(f' **Context**: {(output.get("enriched_context"))}')
                    
                    primary = output.get("primary_metric")
                    st.session_state.secondary = output.get("secondary_metrics")
                    
                    st.write(f' **Domain**: {(output.get("domain"))}')
                    st.write(f" **Targeting**: {primary}")

                elif node_name == "feature_engineering":
                    st.write("### ⚒️ Feature Engineering")
                    num_new = len(st.session_state.secondary) - len(output.get("secondary_metrics"))
                    st.write(f'{num_new} new metrics added.') 
                    if num_new > 0:
                        st.write("New Metrics: ",output.get("secondary_metrics")[-num_new:])

                elif node_name == "visualisation_planning":
                    chart_specs = output.get("chart_specs")
                    if chart_specs:
                        st.write("### 📈 Visualisation Planning")
                        st.write((f"Planned Charts: \n Univariate: {len(chart_specs['univariate'])}, Bivariate: {len(chart_specs['bivariate'])}, Multivariate: {len(chart_specs['multivariate'])}"))

                elif node_name == "visionary_analyst":
                    retry_count = st.session_state.retry_count
                    if st.session_state.retry_count == 0:
                        st.write("### 📖 Charts Analysis")
                    st.write(f"Analysing for the {car2ord.get(retry_count + 1)} time. Found {len(output.get('complete_insight'))} insights in total.")
                    st.session_state.retry_count += 1
                                    
                elif node_name == "executive_summary":
                    st.session_state.pyg_specs = output.get("pyg_specs")
                    st.session_state.df = output.get("df")
                    st.session_state.executive_summary = output.get("executive_summary")
                    st.session_state.raw_pngs = output.get("raw_pngs")
                    st.session_state.outlier_infos = output.get("outlier_infos")
                             
                elif node_name == "simple_summary":
                    st.write("No effective insights found. Simple summary will be generated automatically.")
                    st.session_state.executive_summary = output.get("executive_summary")
                
                elif node_name == "error_node":
                    st.write("### ❌ Error occured:")
                    st.write(output.get("error_message"))
                    st.session_state.has_error = True

    raw_pngs = st.session_state.raw_pngs
    
    if raw_pngs:
        st.write("### 📊 Detailed AI Chart Analysis")

        for i, item in enumerate(raw_pngs):
            with st.expander(f"{item.get('title', 'Insight')}", expanded=False):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(item['viz_png'])
                with col2:
                    st.info(f"Observation: {item['findings']}")
                    st.info(f"Outlier Information: {item['outlier_info']}")

    # Render the result outside of the chunk stream loop and status expander
    
    if st.session_state.executive_summary:
        res = st.session_state.executive_summary
        
        st.header("👔 Executive Briefing")
        tab1, tab2, tab3 = st.tabs(["Act1: Baseline", "Act2: Divergence", "Act3: Lever"])
        # Using .get() as a safety measure in case keys differ or are missing
        tab1.markdown(f"""{res.act1}""")
        tab2.markdown(f"""{res.act2}""")
        tab3.markdown(f"""{res.act3}""")
    
    if st.session_state.pyg_specs:
        @st.fragment
        def render_advanced_explorer():
            if st.button("🛠️ On/Off Advanced Data Explorer"):
                st.session_state.show_advanced_viz = not st.session_state.show_advanced_viz
            if st.session_state.show_advanced_viz:
                try:
                    walker = StreamlitRenderer(
                        st.session_state.df, 
                        spec=st.session_state.pyg_specs,
                        use_kernel_calc=True, # Better performance for larger dfs
                        spec_io_mode="json"
                    )
                    walker.explorer()
                except Exception as e:
                    st.error(f"Failed to render PyGWalker visualization. The generated spec might be invalid: {e}")
        
        render_advanced_explorer()

    if st.session_state.has_error:
        status.update(label="❌ Analysis Failed", state="error")
    else:
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
    stop_button_placeholder.empty()
