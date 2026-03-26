from langgraph.graph import StateGraph, START, END
from .state import AgentState
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Dynamically choose the nodes module based on API key availability
if os.environ.get("GOOGLE_API_KEY"):
    logger.info("GOOGLE_API_KEY found. Using Google-specific nodes.")
    from .nodes_google import (
        preprocessing_node,
        enrichment_node, 
        feature_engineering_node,
        visualisation_planning_node,
        visionary_analyst_node,
        executive_summary_node,
        simple_summary_node,
        error_node,
        error_router,
        retry_router    
    )
else:
    logger.info("GOOGLE_API_KEY not found. Using default nodes.")
    from .nodes import (
        preprocessing_node,
        enrichment_node, 
        feature_engineering_node,
        visualisation_planning_node,
        visionary_analyst_node,
        executive_summary_node,
        simple_summary_node,
        error_node,
        error_router,
        retry_router    
    )
from .tools import setup_logging
setup_logging()

workflow = StateGraph(AgentState)

workflow.add_node("preprocessing", preprocessing_node)
workflow.add_node("context_enrichment", enrichment_node) 
workflow.add_node("feature_engineering", feature_engineering_node)        
workflow.add_node("visualisation_planning", visualisation_planning_node)
workflow.add_node("visionary_analyst", visionary_analyst_node)
workflow.add_node("executive_summary", executive_summary_node) 
workflow.add_node("simple_summary", simple_summary_node)
workflow.add_node("error_node", error_node)


workflow.add_edge(START, "preprocessing")
workflow.add_conditional_edges("preprocessing",error_router,
                                {"next": "context_enrichment",
                                "error": "error_node"})
workflow.add_conditional_edges("context_enrichment",error_router,
                                {"next": "feature_engineering",
                                "error": "error_node"})
workflow.add_edge("feature_engineering", "visualisation_planning")
workflow.add_edge("visualisation_planning", "visionary_analyst")
workflow.add_conditional_edges("visionary_analyst",retry_router,
                                {"retry": "visionary_analyst",
                                "error": "error_node",
                                "simple_summary": "simple_summary",
                                "executive_summary": "executive_summary"})
workflow.add_edge("executive_summary", END)
workflow.add_edge("error_node", END)
workflow.add_edge("simple_summary", END)

app = workflow.compile()

#app.get_graph().draw_mermaid_png(output_file_path="workflow_graph.png")
