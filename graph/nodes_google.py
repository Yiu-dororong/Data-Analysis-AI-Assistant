from enum import Enum
from typing import Dict, List, Literal, Optional
import json
import logging

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage # Keep these for multimodal calls
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # Import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from scipy.stats import kurtosis, skew

from . import prompts
from .state import AgentState
from .tools import get_batch, sanitize_column_names, get_data_profile, generate_visual_artifacts, stitch_and_label_charts, save_debug_images, get_time_metadata, get_forensic_metrics

plt.style.use('seaborn-v0_8-whitegrid')

logger = logging.getLogger(__name__)

# --- Google Generative AI LLM Setup ---
# This setup uses Google's Generative AI models.
# It securely loads the API key from a .env file in your project root.
logger.info("Setting up Google Generative AI LLM.")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0, # For deterministic structured output
)
logger.info(f"Google Generative AI LLM '{llm.model}' setup complete.")
logger.info("--- Starting new analysis run ---")

def preprocessing_node(state: AgentState):
    logger.info("--- Entering Preprocessing Node ---")
    df = state["df"]
    #Create Replace Option for null/outlier etc? like "unknown"
    logger.info("Dropping duplicates and NA values.")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        state['error'] = True
        state['error_message'] = "The dataset is empty after preprocessing"

    if len(df) < 5:
        state['error'] = True
        state['error_message'] = "The dataset has less than 5 rows after preprocessing"
    state["null_and_dup"] = (df.isna().sum().sum(), df.duplicated().sum())
    df = sanitize_column_names(df)
    state['sample_values'] = df.sample(5,ignore_index=True)
    state['data_profile'] = get_data_profile(df)
    state['df'] = df
    state["column_names"] = list(df.columns)
    logger.info("--- Exiting Preprocessing Node ---")
    return state

def create_dynamic_enrichment_model(column_names: List[str]):
    """
    Creates a Pydantic model with a dynamic Enum based on the uploaded CSV columns.
    """
    # 1. Create a dynamic Enum from the actual CSV headers
    # This forces the LLM to pick ONLY from these options
    ColumnsEnum = Enum("ColumnsEnum", {col: col for col in column_names})

    # 2. Define the nested components
    class Sufficiency(BaseModel):
        is_enough_context: bool = Field(description="Can we identify the industry/story?")
        confidence_score: float = Field(ge=0.0, le=1.0)
        missing_info_reason: Optional[str] = Field(default=None, description="Why is context lacking?")

    class Context(BaseModel):
        domain: str = Field(description="e.g., Retail, Finance, Healthcare")
        enriched_context: str = Field(description="Describe what this dataset is about")
        # Use the Enum here to force valid column selection
        primary_metric: ColumnsEnum = Field(description="The primary metric for analysis")
        secondary_metrics: List[ColumnsEnum] = Field(description="The secondary metric(s) for analysis")

    class ProcessingPlan(BaseModel):
        # List of Enums ensures AI only drops real columns
        drop_columns: List[ColumnsEnum] = Field(description="PII or Noise columns to remove")
        temporal_columns: List[ColumnsEnum] = Field(description="Columns that should be parsed as dates (pd.to_datetime)")
        # Dict mapping original keys to friendly strings
        column_mapping: Dict[ColumnsEnum, str] = Field(description="Mapping original names to human-readable business names")

    # 3. The Final Base Model
    class EnrichmentResult(BaseModel):
        sufficiency: Sufficiency
        context: Context
        processing_plan: ProcessingPlan

    return EnrichmentResult

def enrichment_node(state: AgentState):
    logger.info("--- Entering Enrichment Node ---")
    df = state["df"]
    # Get the column list from the state (populated by the file uploader)
    cols = state["column_names"] # Use the original column names for model creation
    
    # 1. Generate the custom model for THIS specific CSV
    EnrichmentModel = create_dynamic_enrichment_model(cols)
    logger.info("Created dynamic Pydantic model for enrichment.")
    
    # 2. Setup Parser and Chain
    parser = JsonOutputParser(pydantic_object=EnrichmentModel)
    prompt = PromptTemplate(
        template=prompts.enrichment_prompt_template + "\n{format_instructions}",
        input_variables=["filename", "column_names", "sample_values", "data_profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    
    # 3. Invoke
    logger.info("Invoking LLM for data enrichment and context generation.")
    result_dict = chain.invoke({
        "filename": state["filename"],
        "column_names": state["column_names"],
        "sample_values": state["sample_values"],
        "data_profile": state['data_profile']
    })
    result = EnrichmentModel(**result_dict)
    logger.info("LLM enrichment call complete.")
    
    if not result.sufficiency.is_enough_context:
        state['error'] = True
        state['error_message'] = result.sufficiency.missing_info_reason
        return state
    
    logger.info(f"Temporal columns to convert: {[c.value for c in result.processing_plan.temporal_columns]}")
    for col in result.processing_plan.temporal_columns:
        col_name = col.value
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
    
    drop_list = [result.processing_plan.column_mapping.get(drop_column.value, drop_column.value) for drop_column in result.processing_plan.drop_columns]
    if drop_list:
        logger.info(f"Dropping columns: {drop_list}")
        df.drop(columns=drop_list, inplace=True, errors='ignore')
    

    column_mapping = dict()
    for k,v in result.processing_plan.column_mapping.items():
        column_mapping[k.value] = v        
    logger.info(f"Renaming columns: {column_mapping}")
    df.rename(columns=column_mapping, inplace=True)

    if result.context.primary_metric in result.processing_plan.column_mapping:
        result.context.primary_metric = result.processing_plan.column_mapping[result.context.primary_metric]
    state['primary_metric'] = result.context.primary_metric
    logger.info(f"Primary Metric set to: {state['primary_metric']}")

    result.context.secondary_metrics = [result.processing_plan.column_mapping.get(secondary_metric, secondary_metric) for secondary_metric in result.context.secondary_metrics]
    state['secondary_metrics'] = [c for c in result.context.secondary_metrics]
    logger.info(f"Secondary Metrics set to: {state['secondary_metrics']}")

    result.processing_plan.temporal_columns = [result.processing_plan.column_mapping.get(temporal_column, temporal_column) for temporal_column in result.processing_plan.temporal_columns]
    state['temporal_columns'] = [c for c in result.processing_plan.temporal_columns]
    logger.info(f"Temporal Columns identified: {state['temporal_columns']}")

    state['enriched_context'] = result.context.enriched_context
    state['domain'] = result.context.domain
    logger.info(f"Domain identified as: {state['domain']}")

    state["column_names"]=list(df.columns)
    state["sample_values"]=df.sample(5,ignore_index=True)
    state['data_profile'] = get_data_profile(df)
    
    logger.info("--- Exiting Enrichment Node ---")
    return state

def create_feature_engineering_model(column_names: List[str]):
    """
    Creates a Pydantic model with a dynamic Enum based on the uploaded CSV columns.
    """
    # 1. Create a dynamic Enum from the actual CSV headers
    # This forces the LLM to pick ONLY from these options
    ColumnsEnum = Enum("ColumnsEnum", {col: col for col in column_names})

    # 2. Define the nested components
    class FERecipe(BaseModel):
        type: Literal["division", "subtraction", "multiplication", "addition", "temporal"]
        new_name: str
        numerator: Optional[ColumnsEnum] = None
        denominator: Optional[ColumnsEnum] = None
        
        minuend: Optional[ColumnsEnum] = None
        subtrahend: Optional[ColumnsEnum] = None

        multiplicand: Optional[ColumnsEnum] = None
        multiplier: Optional[ColumnsEnum] = None
        
        addend_1: Optional[ColumnsEnum] = None
        addend_2: Optional[ColumnsEnum] = None

        # For temporal
        time_col: Optional[ColumnsEnum] = None

        reasoning: str
    class FEPlan(BaseModel):
        needed: bool
        actions: List[FERecipe]

    return FEPlan


def feature_engineering_node(state: AgentState):
    logger.info("--- Entering Feature Engineering Node ---")
    df = state["df"]
    column_names = state["column_names"]
    FEModel = create_feature_engineering_model(column_names)
    logger.info("Created dynamic Pydantic model for feature engineering.")
    days, time_metadata=get_time_metadata(df,state['temporal_columns'])
    state['day_span'] = days
    
    parser = JsonOutputParser(pydantic_object=FEModel)
    prompt = PromptTemplate(
        template=prompts.feature_engineering_prompt_template + "\n{format_instructions}",
        input_variables=["enriched_context", "primary_metric", "secondary_metrics", "time_metadata", "data_profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser

    logger.info("Invoking LLM for feature engineering plan.")
    result_dict = chain.invoke({
        "enriched_context": state['enriched_context'],
        "primary_metric": state['primary_metric'],
        "secondary_metrics": state['secondary_metrics'],
        "time_metadata": time_metadata,
        "data_profile": state['data_profile'],
    })
    result = FEModel(**result_dict)

    if not result.needed:
        logger.info("No feature engineering needed according to the plan.")
        state['data_profile'] = get_data_profile(df) # Recalculate profile even if no changes
        return state
        
    for action in result.actions:
        logger.info(f"Applying feature engineering action: type='{action.type}', new_name='{action.new_name}'")
        # 1. Handle Math operations
        if action.type == "division" and action.numerator and action.denominator:
            df[action.new_name] = df[action.numerator.value] / df[action.denominator.value].replace(0, np.nan) # Ensure division by zero is handled
            state['secondary_metrics'].append(action.new_name)
        elif action.type == "addition" and action.addend_1 and action.addend_2:
            df[action.new_name] = df[action.addend_1.value] + df[action.addend_2.value]
            state['secondary_metrics'].append(action.new_name)
        elif action.type == "subtraction" and action.minuend and action.subtrahend:
            if pd.api.types.is_datetime64_any_dtype(df[action.minuend.value]) and pd.api.types.is_datetime64_any_dtype(df[action.subtrahend.value]):
                df[action.new_name] = (df[action.minuend.value] - df[action.subtrahend.value]).dt.days # Difference in days
            elif not pd.api.types.is_datetime64_any_dtype(df[action.minuend.value]) and not pd.api.types.is_datetime64_any_dtype(df[action.subtrahend.value]):
                df[action.new_name] = df[action.minuend.value] - df[action.subtrahend.value]
            else:
                continue
            state['secondary_metrics'].append(action.new_name)
        elif action.type == "multiplication" and action.multiplicand and action.multiplier:
            df[action.new_name] = df[action.multiplicand.value] * df[action.multiplier.value] # Simple multiplication
            state['secondary_metrics'].append(action.new_name)

        # 2. Handle Temporal
        elif action.type == "temporal":

            col = action.time_col.value
            # SAFETY CHECK: Only proceed if Pandas confirmed it is a datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # 1. Day of Week (Useful for short-term analysis)
                if days < 31:
                    df[f"{col}_day"] = df[col].dt.day_name()
                    state['secondary_metrics'].append(f"{col}_day")
                
                # 2. Day of Week + Month (Useful for mid-term analysis)
                elif days < 183:
                    df[f"{col}_day"] = df[col].dt.day_name()
                    df[f"{col}_month"] = df[col].dt.month_name()
                    state['secondary_metrics'].append(f"{col}_day")
                    state['secondary_metrics'].append(f"{col}_month")

                # 3. Month (Useful for long-term analysis)
                elif days < 365:
                    df[f"{col}_month"] = df[col].dt.month_name()
                    state['secondary_metrics'].append(f"{col}_month")

                # 4. Month + Year (Useful for annual analysis)
                else:
                    df[f"{col}_month"] = df[col].dt.month_name()
                    df[f"{col}_year"] = df[col].dt.year
                    state['secondary_metrics'].append(f"{col}_month")
                    state['secondary_metrics'].append(f"{col}_year")

        else:
            logger.warning(f"Unhandled feature engineering action type: {action.type}")

    state["column_names"]=list(df.columns)
    state["sample_values"]=df.sample(5,ignore_index=True)
    state['time_metadata'] = time_metadata
    state['data_profile'] = get_data_profile(df)
    logger.info("--- Exiting Feature Engineering Node ---")
    return state

def create_planning_model(column_names: List[str]):
    ColumnsEnum = Enum("ColumnsEnum", {col: col for col in column_names})

    class ChartSpec(BaseModel):
        title: str = Field(description="Descriptive title of the chart")
        level: Literal["univariate", "bivariate", "multivariate"]
        x: ColumnsEnum 
        y: Optional[ColumnsEnum] = None
        color_by: Optional[ColumnsEnum] = None
        #facet_by: Optional[ColumnsEnum] = Field(description="Column to split the chart into a grid (e.g., 'Category' or 'Region')") #use color_by
        strategic_score: int = Field(ge=1, le=10)
        justification: str 
        def __hash__(self):
            return hash((self.x, self.y, self.color_by))
    
    class AnalysisPlan(BaseModel):
        charts: List[ChartSpec] = Field(min_length=5, max_length=15, description="A list of 5 to 15 high-impact chart specifications.")
    
    return AnalysisPlan

def visualisation_planning_node(state: AgentState):
    logger.info("--- Entering Visualisation Planning Node ---")
    PlanningModel = create_planning_model(state['column_names'])
    logger.info("Created dynamic Pydantic model for visualisation planning.")
    
    parser = JsonOutputParser(pydantic_object=PlanningModel)
    prompt = PromptTemplate(
        template=prompts.visualisation_plan_prompt_template + "\n{format_instructions}",
        input_variables=["domain", "enriched_context", "primary_metric", "secondary_metrics", "time_metadata", "data_profile"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    
    # 3. Invoke
    logger.info("Invoking LLM for visualisation plan.")
    result_dict = chain.invoke({
        "domain": state['domain'],
        "enriched_context": state['enriched_context'],
        "primary_metric": state['primary_metric'],
        "secondary_metrics": state['secondary_metrics'],
        "time_metadata": state['time_metadata'],
        "data_profile": state['data_profile']
    })
    result = PlanningModel(**result_dict)

    logger.info(f"LLM visualisation plan received with {len(result.charts)} initial charts.")

    result.charts = [chart for chart in result.charts if chart.strategic_score > 6]
    
    logger.info(f"Filtered to {len(result.charts)} charts with strategic score > 6.")
    
    chart_specs = {"univariate" : [], 
                  "bivariate": [], 
                  "multivariate" : [],}
    chart_set = set()

    for chart in result.charts:
        x = chart.x.value # Access the string value from the Enum
        y = chart.y.value if chart.y else None # Access the string value from the Enum
        color = chart.color_by.value if chart.color_by else None

        chart_tuple = (x, y, color)
        
        if chart_tuple not in chart_set:
            chart_set.add(chart_tuple)
        else:
            continue

        if chart.y == None and chart.color_by == None :
            chart.level = "univariate"
            chart_specs["univariate"].append(chart)
        elif chart.y == None or chart.color_by == None:
            chart.level = "bivariate"
            chart_specs["bivariate"].append(chart)
        else:
            chart.level = "multivariate"
            chart_specs["multivariate"].append(chart)
    
    num_of_charts = len(chart_set)
    logger.info(f"Final unique chart count: {num_of_charts}")
    
    state['chart_specs'] = chart_specs
    state['batches'] = get_batch(num_of_charts)
    logger.info(f"Chart specs categorized: Univariate: {len(chart_specs['univariate'])}, Bivariate: {len(chart_specs['bivariate'])}, Multivariate: {len(chart_specs['multivariate'])}")
    logger.info("--- Exiting Visualisation Planning Node ---")
    return state

def create_visionary_analyst_model(column_names: List[str]):
    ColumnsEnum = Enum("ColumnsEnum", {col: col for col in column_names})
    class ChartInsight(BaseModel):
        label: str = Field(description="The label identifier from the image (A, B, C, etc.)")
        
        # --- Part 1: Technical Audit ---
        needs_tweak: bool = Field(description="True if the chart is unreadable or visually poor")
        suggested_tweak: Optional[Literal[ 
            "rolling_mean", 
            "unpin_y_zero", 
            "log_scale", 
            "facet_grid",
            "color_by_segment",
        ]] = Field(default=None, description="The specific technical fix required")
        quality_score: int = Field(ge=1, le=10, description="Visual clarity score (10 is the best)")
        facet_by: Optional[ColumnsEnum] = Field(default=None, description="Column to split the chart into a grid (e.g. 'Region')")
        color_by: Optional[ColumnsEnum] = Field(default=None, description="Column to color the chart (e.g. 'Category')")
        # --- Part 2: Strategic Analysis ---
        title: str = Field(description="The technical axis mapping. Format: 'Y-axis vs X-axis (across Facet)'. No insights here.")
        observation: str = Field(description="Direct visual observation (trends, clusters, outliers)")
        strategic_impact: str = Field(description="How this specific finding affects the North Star metric")
        strategic_score: int = Field(ge=1, le=10, description="Strategic value score (10 is the best)")
        is_spurious: bool = Field(description="True if the chart shows no logical or useful relationship")
        is_valid: bool

    class VisionAuditResult(BaseModel):
        """The root model for a batch of analyzed charts."""
        audit_results: List[ChartInsight] = Field(min_length=1, max_length=15)
        #batch_summary: str = Field(description="A 1-sentence summary of the 'vibe' of this chart batch")
    return VisionAuditResult

def visionary_analyst_node(state: AgentState):
    df = state["df"]
    if "retry_count" not in state or state["retry_count"] == 0:
        state["retry_count"] = 0
        logger.info("--- Entering Visionary Analyst Node (First Try)---")

    chart_specs = []
    #if no tweak (1st try)
    if "tweak_insight" not in state or not state["tweak_insight"]:
        chart_list = list(state['chart_specs'].values())
        for charts in chart_list:
            chart_specs.extend(charts)
    #or if it has
    else:
        chart_specs = state['tweak_insight']
        state["batches"] = get_batch(len(chart_specs))
        #pass tweak config to generate visual artifacts
    
    if not chart_specs:
        logger.warning("No chart specs found to generate visuals. Exiting node.")
        return state

    logger.info(f"Generating {len(chart_specs)} visual artifacts.")

    raw_pngs = []
    evidences = []
    pyg_configs = []

    #For retry, chart specs is a tuple that contains chart spec + tweak insight + evidence + pyg config/png
    if type(chart_specs[0]) != tuple:
        for chart_spec in chart_specs:
            png_data, pyg_config, evidence = generate_visual_artifacts(df, chart_spec, state)
            raw_pngs.append(png_data)
            evidences.append(evidence)
            pyg_configs.append(pyg_config)
    else:
        for chart_spec, tweak, e, pyg, png  in chart_specs:
            png_data, pyg_config, evidence = generate_visual_artifacts(df, chart_spec, state, tweak)
            raw_pngs.append(png_data)
            evidences.append(evidence)
            pyg_configs.append(pyg_config)
    

    state['raw_pngs'] = raw_pngs

    # Get our Batches (e.g., [[0,1,2], [3,4]])
    batches = state["batches"]
    cols = state["column_names"]

    logger.info("Creating dynamic Pydantic model for vision analysis.")
    VisionAuditModel = create_visionary_analyst_model(cols)
    parser = JsonOutputParser(pydantic_object=VisionAuditModel)
    

    logger.info(f"Processing {len(batches)} batches of charts for vision analysis.")
    all_descriptions = {}
    drop_insight = []
    complete_insight = []
    tweak_insight = []
    for i, batch_indices in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} with charts: {batch_indices}")
        try:
            # 2. Get the PNG buffers for this specific batch
            batch_pngs = [state["raw_pngs"][idx] for idx in batch_indices]
            #save_debug_images(batch_pngs, prefix="retry")
            
            logger.info("Stitching images and preparing for LLM vision call.")
            # 3. Stitch and Label them (Using the tools.py we built)
            # This returns a single Base64 string
            stitched_b64 = stitch_and_label_charts(batch_pngs)
            
            # 4. Prepare Metadata for the LLM so it knows the column names
            batch_metadata = [chart_specs[idx] for idx in batch_indices]

            # 5. Call Gemma 3 (Multimodal)
            logger.info("Invoking LLM for vision audit.")
            
            system_prompt = prompts.vision_audit_prompt_template.format(
                domain= state['domain'],
                enriched_context= state['enriched_context'],
                primary_metric= state['primary_metric'],
                secondary_metrics = state['secondary_metrics'],
                time_metadata = state['time_metadata'],
                sample_values = state['sample_values'],
                data_profile = state['data_profile'],
            )
            system_prompt += "\n" + parser.get_format_instructions()

            response_msg = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": f"Analyze this strip of {len(batch_indices)} charts."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{stitched_b64}"}}
                ])
            ])
            response_dict = parser.parse(response_msg.content)
            response = VisionAuditModel(**response_dict)
            logger.info("LLM vision audit complete for batch.")
            # 6. Parse and store results
            tweak_history = state["applied_tweaks"] if "applied_tweaks" in state else {}
            for idx, insight_obj in zip(batch_indices, response.audit_results):
            # We convert the Pydantic object to a dict to store in the State
                all_descriptions[idx] = insight_obj.dict()
                if type(chart_specs[0]) != tuple:
                    chart_spec = chart_specs[idx]
                else:
                    chart_spec = chart_specs[idx][0] # Get the original chart_spec
                evidence = evidences[idx]
                pyg_config = pyg_configs[idx]
                raw_png = raw_pngs[idx]

                # Check if the tweak has already been applied
                if insight_obj.suggested_tweak in tweak_history.get(chart_spec, []):
                    if insight_obj.strategic_score < 6:
                        drop_insight.append(insight_obj)
                        continue
                    else:
                        complete_insight.append((chart_spec, insight_obj, evidence, pyg_config, raw_png))
                        continue

                # Drop spurious charts
                if insight_obj.is_spurious:
                    drop_insight.append(insight_obj)
                    continue
                elif insight_obj.suggested_tweak:
                    if (insight_obj.suggested_tweak == "color_by_segment") and (insight_obj.color_by in (chart_spec.x, chart_spec.y)):
                        drop_insight.append(insight_obj)
                        continue
                    elif (insight_obj.suggested_tweak == "facet_grid") and (chart_spec.color_by == None) and (insight_obj.facet_by in (chart_spec.x, chart_spec.y)):
                        drop_insight.append(insight_obj)
                        continue
                    else:
                        tweak_insight.append((chart_spec, insight_obj, evidence, pyg_config, raw_png))
                else:
                    complete_insight.append((chart_spec, insight_obj, evidence, pyg_config, raw_png))
                
                if chart_spec in tweak_history:
                    tweak_history[chart_spec].append(insight_obj.suggested_tweak)
                else:
                    tweak_history[chart_spec] = [insight_obj.suggested_tweak]
            state['applied_tweaks'] = tweak_history
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}", exc_info=True)
            continue
    logger.info(f"Vision analysis complete. Dropped: {len(drop_insight)}, Complete: {len(complete_insight)}, Needs Tweak: {len(tweak_insight)}")
    state['tweak_insight'] = tweak_insight
    
    if "complete_insight" in state and state["complete_insight"]:
        state['complete_insight'].extend(complete_insight)
    else:
        state['complete_insight'] = complete_insight
    
    logger.info("--- Exiting Visionary Analyst Node ---")
    return state

class ExecutiveBriefModel(BaseModel):
    act1: str = Field(
        description="ACT 1: THE STATE OF THE UNION. Establish the current baseline and global health of the North Star."
    )
    act2: str = Field(
        description="ACT 2: THE CRITICAL DIVERGENCE. Identify why the data is moving. Connect visual trends to hidden outlier math."
    )
    act3: str = Field(
        description="ACT 3: THE DATA-DRIVEN DIRECTIVE. Provide the final 'So What' and the primary strategic move."
    )

def executive_summary_node(state: AgentState):
    logger.info("--- Entering Executive Summary Node ---")    
    # 1. Prepare the Evidence Map (Merging Vision + Outlier Math)
    logger.info("Preparing evidence map for final summary generation.")
    evidence_list = []
    pyg_configs = []
    raw_pngs = []
    
    for chart_spec, insight_obj, evidence, pyg_config, raw_png in (state["complete_insight"] if "complete_insight" in state else []):
        # Get the math facts we kept on the side during the Painter phase
        if not evidence['outlier_clipped']:
            outlier_info = "No significant outliers."
        else:
            outlier_info = evidence["group_details"]
        
        evidence_list.append({
            "chart": insight_obj.title,
            "finding": insight_obj.observation,
            "impact": insight_obj.strategic_impact,
            "outlier_context": outlier_info # This is a dict, will be stringified by json.dumps
        })
        pyg_configs.append(pyg_config)
        raw_pngs.append({"title": insight_obj.title, "findings": insight_obj.observation, "viz_png": raw_png, "outlier_info": outlier_info})

    for i, config in enumerate(pyg_configs):
        config["name"] = f"Insight {i+1}"
        config["visId"] = f"chart_{i+1}"

    state["pyg_specs"] = json.dumps(pyg_configs)
    state["raw_pngs"] = raw_pngs

    # 2. Construct the Comprehensive Prompt
    logger.info("Constructing prompt for executive summary.")
    prompt = prompts.executive_summary_prompt_template.format(
        domain=state["domain"],
        primary_metric=state["primary_metric"],
        secondary_metrics=state["secondary_metrics"],
        enriched_context=state["enriched_context"],
        data_profile=state["data_profile"],
        evidence_json=json.dumps(evidence_list, indent=2)
    )
    
    # 3. Invoke Chief Consultant
    parser = JsonOutputParser(pydantic_object=ExecutiveBriefModel)
    final_prompt = PromptTemplate(
        template="{prompt_str}\n{format_instructions}",
        input_variables=["prompt_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = final_prompt | llm | parser

    logger.info("Invoking LLM for executive summary.")
    result_dict = chain.invoke({"prompt_str": prompt})
    response = ExecutiveBriefModel(**result_dict)
    logger.info("Executive summary generated.")

    print("Final summary: ",response.act1, "\n", response.act2, "\n", response.act3, "\n")

    logger.info("--- Exiting Executive Summary Node ---")

    state["executive_summary"] = response
    logger.info("--- Analysis run finished ---")    
    return state

def simple_summary_node(state: AgentState):
    logger.info("--- Entering Simple Summary Node ---")
    df = state["df"]
    target = state["primary_metric"]

    m = get_forensic_metrics(df, target)

    if m["type"] == "numeric":
        act1 = f"BASELINE: '{target}' shows a median of {m['median']:.2f} with a CV of {m['cv']:.2f}."
        act2 = f"DIVERGENCE: We found {m['outlier_count']} extreme events representing {m['outlier_impact']:.1f}% of volume. Kurtosis is {m['kurtosis']:.2f}."
        act3 = f"LEVER: Focus on the {m['outlier_count']} records exceeding the {m['threshold']:.2f} threshold to stabilize the North Star. \n Alert if {target} > {m['threshold']:.2f}"

    else:
        act1 = f"BASELINE: '{target}' is dominated by '{m['top_class']}' ({m['top_share']:.1f}% share)."
        status = "fragmented" if m['entropy_score'] > 0.7 else "concentrated"
        act2 = f"DIVERGENCE: The structure is highly {status} (Entropy: {m['entropy_score']:.2f}) across {m['unique_count']} unique categories."
        act3 = f"LEVER: Prioritize the '{m['top_class']}' segment for immediate scale, as it represents the core driver of the Primary Metric. \n Monitor {m['top_class']} retention. Consolidate {m['unique_count']} variants"

    logger.info("--- Exiting Simple Summary Node ---")

    summary = ExecutiveBriefModel(act1=act1, act2=act2, act3=act3)
    state["executive_summary"] = summary

    logger.info("--- Analysis run finished ---")
    return state

def error_router(state: AgentState):
    if "error" in state and state["error"]:
        return "error"
    return "next"

def retry_router(state: AgentState):
    retry_count = state["retry_count"] if "retry_count" in state else 0
    if ("tweak_insight" in state and state["tweak_insight"]) and retry_count < 2:
        logger.info(f"--- Entering Visionary Analyst Retry Loop (Retry #{retry_count + 1}) ---")
        state["retry_count"] = retry_count + 1
        return "retry"
    complete_insights = state["complete_insight"] if "complete_insight" in state else []
    if not complete_insights:
        return "simple_summary"
    good_insights = []
    for insight in complete_insights:
        if insight[1].quality_score >=6 and insight[1].strategic_score >=6 :
            good_insights.append(insight)
    if good_insights:
        state['complete_insight'] = good_insights
    return "executive_summary"

def error_node(state: AgentState):
    logger.warning(f"An error occurred: {state['error_message']}. Workflow ended.")
    return state