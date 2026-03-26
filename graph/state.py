from typing import Annotated, List, Dict, Any, Union, Optional, Tuple
from typing_extensions import TypedDict
import pandas as pd

def replace(a, b):
    return b

class AgentState(TypedDict):
    # --- 0. Error ---
    error: bool = False
    error_message: Optional[str]

    # --- 1. Basic Metadata ---
    df: Annotated[pd.DataFrame, replace]                 
    filename: str                
    null_and_dup: Optional[Tuple[int, int]]
    column_names: List[str]          
    sample_values: Optional[str]             
    data_profile: Optional[str]

    # --- 2. Call 1: Enriched Context ---
    enriched_context: Optional[str]
    primary_metric: Optional[str]              # The primary metric chosen by AI
    secondary_metrics: Optional[List[str]]
    domain: Optional[str]                 
    column_mapping: Optional[Dict[str, str]] # Original -> Friendly Business Names
    temporal_columns: Optional[List[str]]

    # --- 2.5 Call 2: Feature Engineering ---
    time_metadata: Optional[str]
    day_span: Optional[int]

    # --- 3. Call 3: The Viz Plan ---
    chart_specs: Optional[Dict[str, Any]] # List of chart specs (x, y, type, reasoning)
    batches: Optional[List[List[int]]]
    applied_tweaks: dict[Any, list[str]] # e.g., {"chart_spec": ["log_scale", "facet_grid"]}

    # --- 4. Execution Artifacts ---
    raw_pngs: List[bytes]        # Individual chart images (for Streamlit)
    pyg_specs: Optional[List[Any]]        # Specs for the PyGWalker interactive UI

    # --- 5. Call 3: The Visionary Analysis ---
    # The narrative descriptions written by Gemma 3 after "seeing" the PNGs
    # chart_descriptions: Dict[int, str] # Map of Chart Index -> AI Observation
    tweak_insight: Optional[List[Any]]
    complete_insight: Optional[List[Any]]
    retry_count: Optional[int]
    
    # --- 6. Call 4: Final Synthesis ---
    executive_summary: Optional[Any]      # The 3-sentence "Bottom Line" for the user
