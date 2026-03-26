import base64
import io
import os
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps
# --- Logging Setup ---
# This configures the root logger. It should be executed once when a module that needs it is imported.
def setup_logging():
    root_logger = logging.getLogger()
    
    # Add the handlers to the logger to avoid duplication
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)

        # Create a file handler which logs even info messages
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(os.path.join(log_dir, 'app_activity.log'), mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)

        # Create a console handler with a higher log level for warnings/errors
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        root_logger.addHandler(fh)
        root_logger.addHandler(ch)

# --- End Logging Setup ---

logger = logging.getLogger(__name__)

def apply_visual_tweaks(df, ax, chart_spec, state, tweak):
    """
    Applies the specific 'Elite' Tweak requested by the Vision Auditor.
    Returns: Modified DF (if clipping/grouping is needed) and Tweak Logs.
    """
    tweak_name = tweak.suggested_tweak
    logger.info(f"Applying visual tweak: '{tweak_name}'")
    # 1. UNPIN Y ZERO: Zoom in on small variance
    if tweak_name == "unpin_y_zero" and chart_spec.y:
        if not pd.api.types.is_numeric_dtype(df[chart_spec.y.value]):
            logger.warning(f"Cannot unpin Y-zero for non-numeric axis '{chart_spec.y.value}'. Skipping tweak.")
            return None
        y_min, y_max = df[chart_spec.y.value].min(), df[chart_spec.y.value].max()
        # Set limits to 5% buffer around the data range
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        logger.info("Successfully applied 'unpin_y_zero'.")

    # 2. LOG SCALE: Handle multi-order magnitude (Gold vs Silver)
    elif tweak_name == "log_scale" and chart_spec.y:
        if not pd.api.types.is_numeric_dtype(df[chart_spec.y.value]):
            logger.warning(f"Cannot apply log_scale for non-numeric axis '{chart_spec.y.value}'. Skipping tweak.")
            return None
        if (df[chart_spec.y.value] > 0).all():
            ax.set_yscale('log')
            logger.info("Successfully applied 'log_scale'.")
        else:
            # Fallback for data with zeros/negatives
            ax.set_yscale('symlog')
            logger.info("Applied 'symlog' as a fallback for 'log_scale' due to non-positive values.")

    elif tweak_name == "color_by_segment":
        if not chart_spec.color_by:
            chart_spec.color_by = tweak.color_by
            logger.info(f"Applied 'color_by_segment' with column: {tweak.color_by}")
    else:
        logger.warning(f"Unknown or unhandled tweak: '{tweak_name}'.")
        return tweak_name
    return None

def get_rolling_window(day_span: int) -> int:
    """
    Calculates the 'Scientific Best' smoothing window based on data duration.
    """
    logger.info(f"Calculating rolling window for a day span of {day_span} days.")
    # 1. Micro-span (No smoothing needed or very tiny)
    if day_span < 14:
        logger.info("Day span < 14 days, using window size 3.")
        return 3 
    
    # 2. Standard Business Span (The 'Weekly' clearing)
    if day_span <= 90:
        logger.info("Day span <= 90 days, using window size 7.")
        return 7 
    
    # 3. Mid-term Span (Bi-weekly smoothing)
    if day_span <= 365:
        logger.info("Day span <= 365 days, using window size 14.")
        return 14
    
    # 4. Long-term/Macro Span (Monthly or Quarterly)
    if day_span > 365:
        # Use a 30-day window for yearly trends to show 'Seasonality'
        logger.info("Day span > 365 days, using window size 30.")
        return 30 
    
    logger.info("Using default fallback window size 7.")
    return 7 # Default Fallback

def get_batch(num_of_charts: int):
    logger.info(f"Creating batches for {num_of_charts} charts.")
    temp = []
    res = []
    batch_plan = []
    if num_of_charts <= 4:
        batch_plan.append(num_of_charts)
    else:
        if num_of_charts % 4 == 1:
            batch_plan = [4 for _ in range((num_of_charts - 5) // 4)]
            batch_plan.extend([3,2])
        else:
            batch_plan = [4 for _ in range(num_of_charts // 4)]
            if num_of_charts % 4 != 0:
                batch_plan.append(num_of_charts % 4)
    logger.info(f"Calculated batch plan: {batch_plan}")
    cumsum = 0
    for i in range(len(batch_plan)):
        for j in range(batch_plan[i]):
            temp.append(j+cumsum)
        res.append(temp)
        temp = []
        cumsum += batch_plan[i] 
    logger.info(f"Final batches created: {res}")
    return res

def get_facet_kind(chart_type: str) -> tuple[str, str]:
    """
    Maps the Strategic Chart Type to Seaborn Figure-Level Functions.
    Returns: (Seaborn_Function_Name, Kind_Parameter)
    """
    logger.info(f"Getting facet kind for chart type: '{chart_type}'")
    # 1. Relational Plots (Numerical x Numerical)
    relational_map = {
        "line": ("relplot", "line"),
        "scatter": ("relplot", "scatter"),
    }
    
    # 2. Categorical Plots (Categorical x Numerical)
    categorical_map = {
        "bar": ("catplot", "bar"),
        "box": ("catplot", "box"),
        "violin": ("catplot", "violin"),
        "countplot": ("catplot", "count")
    }

    if chart_type in relational_map:
        result = relational_map[chart_type]
        logger.info(f"Mapped '{chart_type}' to {result}")
        return result
    elif chart_type in categorical_map:
        result = categorical_map[chart_type]
        logger.info(f"Mapped '{chart_type}' to {result}")
        return result
    
    # Default Fallback for Safety
    logger.warning(f"Chart type '{chart_type}' not found. Falling back to ('relplot', 'scatter').")
    return ("relplot", "scatter")

def profile_outliers(df, col_y, color_by=None):
    """
    Detects 'Visual Destroyers' within specific segments (e.g. Gold vs Silver).
    """
    logger.info(f"Profiling outliers for y-column '{col_y}' with grouping by '{color_by}'.")
    # 1. Type Guard: Only process numerical Y
    if not col_y or col_y not in df.columns or not pd.api.types.is_numeric_dtype(df[col_y]):
        logger.info(f"Outlier profiling skipped: Y-column '{col_y}' is non-numeric or does not exist.")
        return {"outlier_clipped": False, "data_type": "non_numeric"}

    evidence = {
        "outlier_clipped": False,
        "data_type": "numeric",
        "group_details": {}
    }

    # 2. Logic: Grouped vs Global
    groups = [color_by] if color_by and color_by in df.columns else [None]
    
    # We create a mapping of 'Group Name' -> 'Clip Threshold'
    clip_thresholds = {}

    for name, group_df in df.groupby(groups[0]) if groups[0] else [( "Global", df )]:
        series = group_df[col_y].dropna()
        if series.empty: continue
        
        p99 = series.quantile(0.99)
        max_val = series.max()
        
        # TRIGGER: If Max is 3x the 99th percentile OF THIS GROUP
        if max_val > (p99 * 3) and p99 > 0:
            logger.info(f"Outlier detected in group '{name}'. Max value {max_val:.2f} > 3 * P99 {p99:.2f}.")
            evidence["outlier_clipped"] = True
            clip_thresholds[name] = p99
            outliers = group_df[group_df[col_y] > p99]
            outlier_count = len(outliers)
            
            total_sum = series.sum()
            outlier_sum = outliers[col_y].sum()
            impact_pct = (outlier_sum / total_sum) * 100 if total_sum > 0 else 0

            # Record the 'Hidden Truth' for this specific segment
            summary = f"Stat Alert: {outlier_count} extreme outliers (Max: {max_val:.2f}) hidden. They represent {impact_pct:.1f}% of total {col_y} value."
            logger.info(summary)
            evidence["group_details"][str(name)] = {
                "original_max": round(float(max_val), 2),
                "clipped_at": round(float(p99), 2),
                "hidden_impact_pct": round(float(impact_pct), 1),
                "summary_line": summary
            }

    evidence["clip_thresholds"] = clip_thresholds
    if not evidence["outlier_clipped"]:
        logger.info("No significant outliers detected that require clipping.")
    return evidence

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures column names are unique, stripped of whitespace, 
    and formatted for smooth LLM/Pandas interaction.
    """
    logger.info("Sanitizing DataFrame column names.")
    original_cols = list(df.columns)
    # 1. Clean names first (Strip + Replace chars) to ensure we dedup the FINAL form
    df.columns = [str(col).strip().replace(" ", "_").replace(".", "_") for col in df.columns]
    
    # 2. Handle duplicates with a 'seen' set to prevent collisions
    seen = set()
    new_cols = []
    for col in df.columns:
        original_col = col
        i = 1
        while col in seen:
            col = f"{original_col}_{i}"
            i += 1
        seen.add(col)
        new_cols.append(col)
    df.columns = new_cols
    logger.info(f"Column names sanitized. Original: {original_cols}, New: {new_cols}")
    
    return df

def get_data_profile(df: pd.DataFrame) -> str:
    """
    Generates a high-density statistical profile for the LLM Planner.
    Identifies Num/Cat/Time and provides 'Visual Hints'.
    """
    logger.info(f"Generating data profile for DataFrame with {len(df.columns)} columns.")
    profile = []
    
    for col in df.columns:
        # 1. Identify Semantic Type
        dtype = df[col].dtype
        nunique = df[col].nunique()
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        
        # Determine Category
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            stype = "TEMPORAL (Time)"
            if df[col].isnull().all():
                timespan = "All values are null"
            else:
                timespan = f"{df[col].min().date()} to {df[col].max().date()}"
            
            entry = (
                f"Column: {col} | Type: {stype}\n"
                f"  - Span: {timespan}\n"
                f"  - Nulls: {null_pct:.1f}%"
            )
            profile.append(entry)
            
        elif pd.api.types.is_numeric_dtype(df[col]) and nunique > 10:
            stype = "NUMERICAL (Continuous)"
            min_v, max_v = df[col].min(), df[col].max()
            mean_v, med_v = df[col].mean(), df[col].median()
            
            # ELITE: Calculate Skewness for 'Log Scale' hint
            skew_hint = "Normal Range"
            if med_v > 0 and (max_v / (med_v + 1e-6) > 50):
                skew_hint = "High Skew (Consider Log)"
            
            entry = (
                f"Column: {col} | Type: {stype}\n"
                f"  - Range: [{min_v:.2f} to {max_v:.2f}]\n"
                f"  - Median: {med_v:.2f}\n"
                f"  - Mean: {mean_v:.2f}\n"
                f"  - Visual Hint: {skew_hint}\n"
                f"  - Nulls: {null_pct:.1f}%"
            )
            profile.append(entry)
            
        else:
            # Everything else is treated as Categorical (Strings or low-rank numbers)
            stype = "CATEGORICAL (Discrete)"
            top_values = df[col].value_counts().head(5).index.tolist()
            
            # ELITE: Cardinality Warning
            cardinality_hint = ""
            if nunique > 15:
                cardinality_hint = f"  - Warning: High Cardinality ({nunique} types). Use 'Top K' or 'Others' grouping."
            
            entry = (
                f"Column: {col} | Type: {stype}\n"
                f"  - Unique Values: {nunique}\n"
                f"  - Sample Values: {top_values}\n"
                f"  - Nulls: {null_pct:.1f}%"
            )
            if cardinality_hint:
                entry += f"\n{cardinality_hint}"
            profile.append(entry)

    logger.info("Data profile generation complete.")
    return "\n\n".join(profile)

def save_debug_images(image_bytes_list: list, prefix: str = "chart"):
    """
    Saves a list of image bytes to a local /debug folder for inspection.
    """
    debug_dir = "debug_output"
    logger.info(f"Saving {len(image_bytes_list)} debug images to '{debug_dir}' with prefix '{prefix}'.")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        logger.info(f"Created debug directory: '{debug_dir}'")
    
    timestamp = datetime.now().strftime("%H%M%S")
    saved_paths = []
    
    for i, img_bytes in enumerate(image_bytes_list):
        filename = f"{debug_dir}/{prefix}_{timestamp}_{i}.png"
        try:
            with open(filename, "wb") as f:
                f.write(img_bytes)
            saved_paths.append(filename)
        except IOError as e:
            logger.error(f"Failed to save debug image to {filename}: {e}")
        
    logger.info(f"Saved {len(saved_paths)} debug images to /{debug_dir}")
    return saved_paths

def generate_visual_artifacts(df, chart_spec, state, tweak = None):
    """
    Generates a PNG buffer and a PyGWalker-ready spec for a single chart.
    """
    # 1. Setup the Style (Optimized for AI Vision)
    logger.info(f"Generating visual artifacts for chart: {chart_spec.title}")
    fig, ax = plt.subplots(figsize=(10, 6))
    if tweak:
        logger.info(f"Applying tweak '{tweak.suggested_tweak}' before plotting.")
        ctype = apply_visual_tweaks(df, ax, chart_spec, state, tweak)
    else:
        ctype = None
    x = chart_spec.x.value
    if chart_spec.y:
        y = chart_spec.y.value
    else:
        y = None
    if chart_spec.color_by:
        color = chart_spec.color_by.value
    else:
        color = None
    if ctype == "rolling_mean" and get_chart_type(df=df, x=x, y=y, color=color, temporal_cols=state['temporal_columns']) != "line":
        ctype = None
    if not ctype:
        ctype = get_chart_type(df=df, x=x, y=y, color=color, temporal_cols=state['temporal_columns'])
    logger.info(f"Determined chart type: '{ctype}' with x='{x}', y='{y}', color='{color}'.")
    #title = chart_spec.title
    target_col = color or x
        
    # Check if the target column is categorical and high-cardinality
    if df[target_col].dtype == 'object' and df[target_col].nunique() > 12:
        # 3. Apply the Smart Top-K logic locally
        df, top_categories = apply_top_k(df, target_col, state["primary_metric"])
     
    evidence = profile_outliers(df, y, color)

    if evidence["outlier_clipped"]:
        # Surgical Clip: Apply the specific threshold to each group
        logger.info("Outliers detected and will be clipped for visualization.")
        def apply_clip(row):
            group_val = row[color] if color else "Global"
            threshold = evidence["clip_thresholds"].get(group_val)
            if threshold and row[y] > threshold:
                return False # Mark for removal
            return True

        mask = df.apply(apply_clip, axis=1)
        original_rows = len(df)
        df = df[mask]
        logger.info(f"Clipped {original_rows - len(df)} outlier rows from DataFrame.")

    g = None
    facet_kind = None
    title = f"{ctype} chart"
    logger.info(f"Painting Chart: type='{ctype}', x='{x}', y='{y}', color='{color}'")
    try:
        # 2. Dynamic Plotting Logic
        if ctype == "histogram":
            sns.histplot(data=df, x=x, hue=color, kde=True, ax=ax)
            title = f"Distribution of {x}"
        elif ctype == "bar":
            sns.barplot(data=df, x=x, y=y, hue=color, ax=ax)
            title = f"{x} vs {y}"
        elif ctype == "line":
            # Ensure X is sorted for line charts to avoid zig-zags
            temp_df = df.sort_values(by=x)
            sns.lineplot(data=temp_df, x=x, y=y, hue=color, ax=ax, alpha=0.8)
            title = f"{x} vs {y}"
        elif ctype == "twin_line":
            temp_df = df.sort_values(by=x)
            sns.lineplot(data=temp_df, x=x, y=y, ax=ax, color='royalblue', alpha=0.8)
            ax.tick_params(axis='y', labelcolor='royalblue')
            ax.grid(False)
            ax2 = ax.twinx()
            sns.lineplot(data=temp_df, x=x, y=color, ax=ax2, color='orange', alpha=0.8)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False)
            title = f"{y} and {color} across time" 
        elif ctype == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=color, size=color, ax=ax, palette="viridis_r")
            title = f"{x} vs {y}"
        elif ctype == "box":
            sns.boxplot(data=df, x=x, y=y, hue=color, ax=ax)
            title = f"Boxplot for {x} vs {y}"
        elif ctype == "countplot":
            sns.countplot(data=df, x=x, hue=color, ax=ax)
            title = f"Distribution of {x}"
        elif ctype == "pie":
            data = df[x].value_counts()
            def func(pct, allvals):
                absolute = int(np.round(pct/100.*np.sum(allvals)))
                return f"{pct:.1f}%\n(Count: {absolute:d})"
            # 2. Draw Donut (Modern Pie)
            ax.pie(data, labels=data.index, autopct=lambda pct: func(pct, data), 
                wedgeprops={'width': 0.6}) # The 'width' makes it a Donut
            plt.xticks(rotation=45)
            plt.tight_layout()
            title = f"Distribution of {x}"
        # --- 1. STACKED BAR (Categorical Intersection) ---
        elif ctype == "stacked_bar":
            # We pivot the data to create a 'Compositional' view
            segment_col = color or y
            
            # Crosstab creates the "Count" matrix for the AI
            ct = pd.crosstab(df[x], df[segment_col])
            # Normalize to 100% so Gemma 3 Vision can see the 'Mix'
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

            for i, container in enumerate(ax.containers):
            # Get the segment name from the legend/columns
                col_name = ct_pct.columns[i]
                
                # Create labels: "Count (Pct%)"
                # We zip the bars in the container with raw values from the 'ct' matrix
                labels = [
                    f"{count}\n({pct:.1f}%)" if pct > 5 else "" # Hide labels for tiny segments
                    for count, pct in zip(ct[col_name], ct_pct[col_name])
                ]
        
                # Add labels to the center of each segment
                ax.bar_label(container, labels=labels, label_type='center', 
                            fontsize=9, fontweight='bold', color='white')
            
            ct_pct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylabel("Percentage (%) of Total")
            title = f"Composition: {x} by {segment_col}"
            plt.xticks(rotation=45)

        # --- 2. HEATMAP (2 Categorical + 1 Numerical) ---
        elif ctype == "heatmap":
            # AI picks X, Y (Categories) and Color (Numeric)
            pivot_table = df.pivot_table(index=y, columns=x, 
                                         values=color, aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        
        # --- 3. VIOLIN / RIDGE (Density Distribution) ---
        elif ctype == "violin":
            # Violin is better for Bivariate (Cat vs Num)
            sns.violinplot(data=df, x=x, y=y, hue=color, 
                           split=True, inner="quart", ax=ax, bw_adjust=0.5)
            title = f"Violin chart for {x} vs {y}"
        elif ctype == "rolling_mean":
            # 1. Calculate the 'Elite' window based on the state's day_span
            window_size = get_rolling_window(state["day_span"])
            
            # 2. Sort and Calculate
            if color:
                df_sorted = df.sort_values(by=[color, x])
                df_sorted['rolling_y'] = df_sorted.groupby(color)[y].transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
                )
                sns.lineplot(
                    data=df_sorted, x=x, y='rolling_y', hue=color,
                    ax=ax, alpha=0.8, palette=state.get("global_palette")
                )
            else:
                df_sorted = df.sort_values(by=x)
                # 'min_periods=1' ensures the line starts at the first point (no gap)
                df_sorted['rolling_y'] = df_sorted[y].rolling(window=window_size, min_periods=1).mean()
                sns.lineplot(
                    data=df_sorted, x=x, y='rolling_y',
                    ax=ax, 
                )
            title = f"{x} vs {y}, with {window_size}-Day rolling mean"
            # 3. Plot the Overlay
            ax.set_xlabel(x)
            ax.set_ylabel(y)
                    
        elif ctype == "facet_grid":
            if not color:
                if tweak.facet_by:
                    color = tweak.facet_by.value
                else:
                    color = y 
            if (pd.api.types.is_numeric_dtype(df[color]) and df[color].nunique() > 10) or color == y: #route back for continuous numeric data type
                ctype = "scatter"
                sns.scatterplot(data=df, x=x, y=y, hue=color,size=color, ax=ax)
                title = f"{x} vs {y}"
            else:
                facet_kind = get_chart_type(df=df, x=x, y=y, color=None, temporal_cols=state['temporal_columns'])  #color is seperated out for facet_grid

                # 1. Initialize the Grid (Deterministic & Safe)
                g = sns.FacetGrid(
                    data=df, 
                    col=color, 
                    hue=color, 
                    col_wrap=3,
                    sharex=False, 
                    sharey=False, 
                    height=3, 
                    aspect=1.2
                )
                # 2. Map the Plot (The Line/Bar/Box NEVER sees 'sharey')
                if facet_kind == "line":
                    g.map_dataframe(sns.lineplot, x=x, y=y)
                elif facet_kind == "scatter":
                    g.map_dataframe(sns.scatterplot, x=x, y=y)
                elif facet_kind == "box":
                    g.map_dataframe(sns.boxplot, x=x, y=y)
                elif facet_kind == "bar":
                    g.map_dataframe(sns.barplot, x=x, y=y)
                elif facet_kind == "countplot":
                    g.map_dataframe(sns.countplot, x=x, y=y)
                elif facet_kind == "violin":
                    g.map_dataframe(sns.violinplot, x=x, y=y)
                else:
                    pass
                # 3. Finalize Layout
                g.set_titles(col_template="{col_name}", fontweight='bold')
                g.set_xticklabels(rotation=45, ha='right')
                title = f"Facet Grid chart of {x} vs {y} among {color}"
                g.figure.suptitle(title)
                g.figure.subplots_adjust(top=0.85)
                fig = g.figure
        else:
            pass
        if color and ctype != "twin_line" and ctype != "facet_grid": #twin line plot does not view color as color
            # Get the legend object
            ax.legend(
                title=color,
                bbox_to_anchor=(1.02, 1), # 1.02 moves it just outside the right border
                loc='upper left',         # Anchors the top-left of the legend to that spot
                borderaxespad=0,          # Removes extra padding
                frameon=True              # Keeps a clean box around it for AI OCR
            )
        ax.set_title(title, fontsize=14, pad=15)
        plt.tight_layout()
        if ctype == "facet_grid":
                fig.suptitle(title)
                fig.subplots_adjust(top=0.85)
        logger.info(f"Successfully generated '{ctype}' chart titled '{title}'.")
        # 3. Save to PNG Buffer (For Call 4 Vision)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120) # 120 DPI is clear for OCR
        buf.seek(0)
        png_data = buf.getvalue()
        plt.close(fig)
        logger.info("Saved chart to in-memory PNG buffer.")
        mapping = {
            "histogram": "histogram",
            "bar": "bar",
            "countplot": "bar",
            "line": "line",
            "scatter": "point",
            "box": "boxplot",
            "heatmap": "rect",
            "violin": "boxplot",
            "twin_line": "twin_line",
            "stacked_bar": "stacked_bar",
            "rolling_mean": "line",
            "facet_grid": "facet_grid",
            "pie": "arc"
        }

        # 4. Create PyGWalker Spec (For Streamlit UI)
        def generate_pyg_spec(df, x, y=None, color=None, facet=None, ctype="bar",width=800, height=600):
            # Helper to construct valid GraphicWalker field specifications
            facet = color if ctype == "facet_grid" else facet
            ctype = mapping.get(facet_kind, "bar") if facet else mapping.get(ctype, "bar")
            def get_field(col_name):
                if not col_name or col_name not in df.columns: return []
                is_num = pd.api.types.is_numeric_dtype(df[col_name])
                is_time = pd.api.types.is_datetime64_any_dtype(df[col_name])
                
                if is_num:
                    sem_type, ana_type = "quantitative", "measure"
                elif is_time:
                    sem_type, ana_type = "temporal", "dimension"
                else:
                    sem_type, ana_type = "nominal", "dimension"
                    
                field = {
                    "fid": col_name,
                    "name": col_name,
                    "semanticType": sem_type,
                    "analyticType": ana_type
                }
                if is_num and ctype not in ["boxplot", "point", "line", "histogram"]:
                    field["aggName"] = "sum"
                return [field]

            # Populate base dimension and measure lists for the dataset
            dimensions = [get_field(c)[0] for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            measures = [get_field(c)[0] for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if ctype == "arc":
                theta_field =  [{"fid": x,
                                "name": f"Count of {x}",
                                "semanticType": "quantitative",
                                "analyticType": "measure",
                                "aggName": "count"
                                }]
                color_field = get_field(x) 
                col_field = get_field(facet)
                row_field = []
            elif ctype == "rect":
                theta_field = []
                color_field = get_field(color)
                if color_field and "aggName" in color_field[0]:
                    color_field[0]["aggName"] = "mean"
                col_field = get_field(facet) + get_field(x) 
                row_field = get_field(y)

            else:
                theta_field = []
                color_field = get_field(color)
                col_field = get_field(facet) + get_field(x)
                if y:
                    if (get_field(y)[0]["analyticType"] == "dimension") and (ctype in ["bar", "stacked_bar"]):
                        row_field = [{
                                "fid": y,
                                "name": f"Count of {y}",
                                "semanticType": "quantitative",
                                "analyticType": "measure",
                                "aggName": "count"
                            }]
                        if ctype == "stacked_bar":
                            color_field = get_field(y)
                    else:
                        row_field = get_field(y)
                else:
                    row_field = []
                
            if ctype == "boxplot" and color:
                col_field = get_field(color) + get_field(x)
            
            if ctype == "stacked_bar":
                ctype = "bar"

            pyg_config ={
                "visId": "gw_1",
                "name": "Chart 1",
                "config": {"defaultAggregated": False if ctype in ["boxplot", "point", "line", "histogram"] else True, 
                           "geoms": [ctype], 
                           "coordSystem": "generic",
                           "size": {"mode": "fixed", "width": width, "height": height}
                           },
                "encodings": {
                    "dimensions": dimensions,
                    "measures": measures,
                    "columns": col_field,
                    "rows": row_field, 
                    "color": color_field,
                    "opacity": [],
                    "size": [],
                    "shape": [],
                    "radius": [],
                    "theta": theta_field,
                    "details": [],
                    "filters": [],
                    "text": []
                }
            }
            if ctype == "twin_line":
                pyg_config["config"]["resolve"] = {"x": "shared", "y": "independent"}
                pyg_config["config"]["stack"] = "none"
                pyg_config["config"]["geoms"] = ["line", "line"]
                pyg_config["encodings"]["color"] = []
                pyg_config["encodings"]["rows"] += get_field(color)
            if ctype == "histogram":
                pyg_config["config"]["geoms"] = ["bar"]

            return pyg_config

        pyg_config = generate_pyg_spec(df, x, y, color, facet_kind, ctype)

        logger.info("Created PyGWalker spec for interactive UI.")

        return png_data, pyg_config, evidence

    except Exception as e:
        logger.error(f"Failed to generate visual artifact for chart '{chart_spec.title}'. Error: {e}", exc_info=True)
        plt.close(fig)
        raise e

def stitch_and_label_charts(image_buffers):
    """
    image_buffers: List of bytes (from plt.savefig)
    Returns: base64 string of the final stitched 'strip'
    """
    num_images = len(image_buffers)
    logger.info(f"Stitching and labeling {num_images} chart images.")
    if not image_buffers:
        logger.warning("No image buffers provided to stitch.")
        return None
    images = [Image.open(io.BytesIO(b)) for b in image_buffers]
    
    # 1. Add Labels (A, B, C...) to each image
    labels = ["A", "B", "C", "D", "E"]
    labeled_images = []
    max_width = max(img.width for img in images)
    
    for i, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        if i >= len(labels):
            logger.warning(f"More than {len(labels)} images to stitch, label for image {i+1} will be missing.")
        else:
            # Draw a black box with white text in the top-left corner
            draw.rectangle([10, 10, 70, 70], fill="black")
            # Use default font if custom isn't loaded
            draw.text((25, 15), labels[i], fill="white", size=40)
        # 2. Pad the narrower images with white to match the max_width
        # This keeps the 'A, B, C' labels perfectly aligned on the left
        padding = max_width - img.width
        # (left, top, right, bottom)
        img = ImageOps.expand(img, border=(0, 0, padding, 0), fill='white')

        labeled_images.append(img)
    logger.info("Added labels and padding to images.")

    # 2. Stack Vertically
    total_width = max(img.width for img in labeled_images)
    total_height = sum(img.height for img in labeled_images)
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    y_offset = 0
    for img in labeled_images:
        canvas.paste(img, (0, y_offset))
        y_offset += img.height
    logger.info("Vertically stacked all images onto a single canvas.")

    # 3. Convert to Base64 for Gemma 3
    buffered = io.BytesIO()
    canvas.save(buffered, format="PNG")
    b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    logger.info("Converted final stitched image to Base64 string.")
    return b64_string

def get_chart_type(df, x, y, color, temporal_cols):
    """
    Scientific Decision Tree for Chart Selection.
    Logic Flow: 1. Time Check -> 2. Dimensionality -> 3. Type Matching
    """
    logger.info(f"Determining chart type for x='{x}', y='{y}', color='{color}'.")
    temporal_cols = temporal_cols or []
    
    # 1. Helper: Identify types early
    is_x_time = x in temporal_cols
    is_x_num = pd.api.types.is_numeric_dtype(df[x])
    is_y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
    is_color_num = pd.api.types.is_numeric_dtype(df[color]) if color else False

    # --- PHASE 1: TEMPORAL ANCHOR (The Story of Time) ---
    if is_x_time:
        if not y: 
            chart_type = "line"      # Frequency of events over time
        if is_y_num:
            if not is_color_num:
                chart_type = "line"      # Trend of a value over time
            else:
                chart_type = "twin_line"
        else:
            chart_type = "stacked_bar"   # Change in composition over time (Cat Y)
        logger.info(f"Temporal anchor. Selected chart type: '{chart_type}'.")
        return chart_type
    # --- PHASE 2: UNIVARIATE (The Baseline) ---
    if not y:
        if is_x_num:
            chart_type = "histogram" # Statistical distribution
        else:
            # Low cardinality (e.g. 2-5 items) = Pie; High = Bar
            chart_type = "pie" if df[x].nunique() <= 10 else "countplot" # Frequency/Volume per category
        logger.info(f"Univariate analysis. Selected chart type: '{chart_type}'.")
        return chart_type

    # --- PHASE 3: BIVARIATE & MULTIVARIATE (The Drivers & Nuance) ---
    
    # CASE A: NUMERICAL X (Scatter Logic)
    if is_x_num:
        if is_y_num:
            # 2 Numbers = Scatter. If 3rd is Num = Bubble; if Cat = Colored Scatter.
            chart_type = "scatter" 
        else:
            # Numeric X, Categorical Y
            chart_type = "box"
        logger.info(f"Numerical X. Selected chart type: '{chart_type}'.")
        return chart_type

    # CASE B: CATEGORICAL X (Comparison Logic)
    else:
        if is_y_num:
            # Categorical X, Numerical Y
            if color:
                # return "facet_grid" if df[x].nunique() > 10 else "box"
                chart_type = "box"
            else:
                chart_type = "violin" if len(df) > 1000 else "box"
        
        else:
            # Categorical X, Categorical Y (The "Mix")
            # If 3rd column is Num, show Intensity; else show Frequency.
            chart_type = "heatmap" if (color and is_color_num) else "stacked_bar"
        logger.info(f"Categorical X. Selected chart type: '{chart_type}'.")
        return chart_type

    logger.warning("No specific chart type matched. Using fallback 'countplot'.")
    return "countplot" # The ultimate "Safe" fallback for any data

def get_time_metadata(df: pd.DataFrame, temporal_columns: list):
    """
    Returns: (days_int, metadata_string)
    """
    logger.info(f"Getting time metadata for temporal columns: {temporal_columns}")
    # 1. Handle cases with no temporal columns
    if not temporal_columns:
        msg = "No temporal data identified. Skip temporal features."
        logger.info(msg)
        return 0, msg

    # 2. Use the primary temporal column (first one identified by Call 1)
    primary_col = temporal_columns[0]
    logger.info(f"Using primary temporal column: '{primary_col}'")
    
    # 3. Safety check: ensure it's converted (it should be from the casting step)
    if not pd.api.types.is_datetime64_any_dtype(df[primary_col]):
        msg = f"Column {primary_col} is not in datetime format."
        logger.error(msg)
        return 0, msg

    # 4. Calculate Span
    min_date = df[primary_col].min()
    max_date = df[primary_col].max()

    # Handle empty or all-null columns
    if pd.isnull(min_date) or pd.isnull(max_date):
        msg = "Temporal column contains only null values."
        logger.warning(msg)
        return 0, msg

    delta = max_date - min_date
    days = delta.days
    
    metadata_string = f"Data spans {days} days (From {min_date.date()} to {max_date.date()})."
    logger.info(f"Calculated time metadata: {metadata_string}")
    
    return days, metadata_string

def apply_top_k(df, category_col, value_col, p_limit=0.9, max_k=12, gap_threshold=5.0):
    """
    Groups high-cardinality columns into Top-K and 'Others'.
    """
    # 1. Aggregate and Sort by descending value
    stats = df.groupby(category_col)[value_col].sum().sort_values(ascending=False).reset_index()
    total = stats[value_col].sum()
    
    # 2. Calculate Cumulative % and Gap Ratios
    stats['cum_p'] = stats[value_col].cumsum() / total
    # Gap ratio = (Current Value) / (Next Value)
    stats['gap_ratio'] = stats[value_col] / stats[value_col].shift(-1)
    
    selected_k = len(stats)
    
    for i in range(len(stats)):
        k = i + 1
        current_p = stats.iloc[i]['cum_p']
        #current_gap = stats.iloc[i]['gap_ratio']
        
        # Rule 1: Visual Limit (Hard Cap)
        if k >= max_k:
            selected_k = k
            break
        
        # Rule 2: Coverage Limit (90% captured)
        if current_p >= p_limit:
            selected_k = k
            break
                       
        # Rule 3: The Gap Rule (Numerical 'Cliff') - Off for now
        # Only check after top 2 to ensure we have a baseline
        # if k >= 2 and current_gap >= gap_threshold:
        #     selected_k = k
        #     break
            

    # 3. Final Mapping
    top_categories = stats.head(selected_k)[category_col].tolist()
    
    df_styled = df.copy()
    df_styled[category_col] = df_styled[category_col].apply(
        lambda x: x if x in top_categories else 'Others'
    )
    logger.info("Applied Top K Others")
    return df_styled, top_categories

def get_forensic_metrics(df: pd.DataFrame, target_col: str):
    series = df[target_col].dropna()
    
    # --- CASE A: NUMERIC (The Volatility Story) ---
    if np.issubdtype(series.dtype, np.number):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_fence = q3 + (1.5 * iqr)
        outliers = series[series > upper_fence]
        
        return {
            "type": "numeric",
            "median": series.median(),
            "mean": series.mean(),
            "cv": series.std() / series.mean() if series.mean() != 0 else 0,
            "kurtosis": kurtosis(series),
            "outlier_count": len(outliers),
            "outlier_impact": (outliers.sum() / series.sum() * 100) if series.sum() != 0 else 0,
            "threshold": upper_fence
        }

    # --- CASE B: CATEGORICAL (The Concentration Story) ---
    else:
        counts = series.value_counts(normalize=True) * 100
        top_class = counts.index[0]
        top_share = counts.iloc[0]
        
        # Entropy (0 = Uniform/Clean, 1 = Chaotic/Fragmented)
        probs = series.value_counts(normalize=True)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
        
        return {
            "type": "categorical",
            "top_class": top_class,
            "top_share": top_share,
            "unique_count": len(counts),
            "entropy_score": entropy / max_entropy,
            "is_imbalanced": top_share > 50
        }