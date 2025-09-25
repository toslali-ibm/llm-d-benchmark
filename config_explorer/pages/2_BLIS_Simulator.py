"""
Benchmarking sweep visualization page
"""
from matplotlib import pyplot as plt
import streamlit as st
import db
import pandas as pd
import util
from src.config_explorer.capacity_planner import *


# --- Helpers ---
def parse_numeric_list(text: str, cast=int, fallback: List[int] | None = None) -> List[int]:
    """
    Parse a string like "50, 60 70;90" into a list of numbers (ints by default).
    Returns fallback if no valid values are found.
    """
    import re
    if not isinstance(text, str):
        return fallback or []
    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    values = []
    for t in tokens:
        try:
            # Support int-only by default but allow float cast if needed
            if cast is int:
                # Allow e.g. "50.0" to become 50
                values.append(int(float(t)))
            else:
                values.append(cast(t))
        except Exception:
            pass
    if not values:
        return fallback or []
    return values

def first_or_fallback(values: List[int], fallback: int) -> List[int]:
    """
    Return a single-value list: the first value if present, else [fallback].
    """
    if values:
        return [values[0]]
    return [fallback]


def check_input():
    """
    Check all required input is there
    """
    scenario = st.session_state[util.USER_SCENARIO_KEY]
    if not scenario.model_name or not scenario.gpu_name or not scenario.gpu_count_avail or not scenario.max_model_len:
        return False
    return True

def display_capacity_planner_data():
    """
    Display info about data
    """
    user_scenario = st.session_state[util.USER_SCENARIO_KEY]
    st.info(f"""Benchmarking results will be filtered based on the following inputs:

- Model: `{user_scenario.model_name}`
- GPU Type: `{user_scenario.gpu_name}`
- GPU Available: `{user_scenario.get_gpu_count()}`
- Max model length: `{user_scenario.max_model_len}` (max context length = `{max_context_len(user_scenario.model_config)}`)
""")

def table(benchmark_data):
    """
    Display table of benchmark data
    """
    st.subheader("Optimal configurations")

    # Data cleaning
    df = benchmark_data[[
        'Name',
        'Concurrency',
        'Request_Throughput',
        'Output_Token_Throughput',
        'Total_Token_Throughput',
        'Mean_TTFT_ms',
        'Mean_TPOT_ms',
        'Mean_ITL_ms',
        'Mean_E2EL_ms',
        'Thpt_per_GPU',
        'Thpt_per_User',
    ]].rename(columns={
        'Name': 'Replicas/Parallelism',
        'Concurrency': 'Batch Size',
        'Request_Throughput': 'Request Thpt',
        'Output_Token_Throughput': 'Output Token Thpt',
        'Total_Token_Throughput': 'Total Token Thpt',
        'Mean_TTFT_ms': 'TTFT (ms)',
        'Mean_TPOT_ms': 'TPOT (ms)',
        'Mean_ITL_ms': 'ITL (ms)',
        'Mean_E2EL_ms': 'E2EL (ms)',
        'Thpt_per_GPU': 'Thpt/GPU (tok/s)',
        'Thpt_per_User': 'Thpt/User (tok/s)',
    })

    # Add a selection checkbox column
    df.insert(0, "Select", False)

    # Build column configuration dynamically
    column_config = {}
    for col in df.columns:
        if col == "Select":
            column_config[col] = st.column_config.CheckboxColumn(
                "Select", help="Check to select this row"
            )
        else:
            # Disable editing for all other columns automatically
            column_config[col] = st.column_config.TextColumn(col, disabled=True)

    # Render the data editor
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        use_container_width=True,
        num_rows="fixed",  # Prevent row adding/deleting
        hide_index=True    # Optional, makes UI cleaner
    )


    # Get selected rows
    selected_rows_df = edited_df[edited_df["Select"]]

    if not selected_rows_df.empty:
        # Get the indices of selected rows
        selected_indices = selected_rows_df.index.tolist()

        st.markdown("##### llm-d configuration:")

        # Iterate over selected rows and fetch data from the original df
        for idx in selected_indices:
            row_data = benchmark_data.loc[idx]  # Drop checkbox col for clean data
            model_name = row_data['Model']
            dp = row_data['DP']
            tp = row_data['TP']

            col1, col2 = st.columns(2)
            col1.write("vLLM arguments")
            col1.code(f"""vllm serve {model_name} \\
--data-parallel-size {str(dp)} \\
--tensor-parallel-size {str(tp)} \\
"""
            )

            col2.write("Inference Scheduler config")
            col2.code("""apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
    parameters:
    hashBlockSize: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
schedulingProfiles:
- name: default
    plugins:
    - pluginRef: decode-filter
    - pluginRef: max-score-picker
    - pluginRef: prefix-cache-scorer
    weight: 50
""", language='yaml')

            st.write("---")
    else:
        st.info("Select one or more rows to see details below.")

def select_slo(benchmark_data):
    """
    Display widgets to select SLO requirements
    """

    user_scenario = st.session_state['scenario']

    st.subheader("Select SLO requirements")
    col1, col2, col3 = st.columns(3)
    user_scenario.ttft = col1.number_input("Max TTFT (ms)", min_value=0, value=5000, step=10)
    user_scenario.tpot = col2.number_input("Max TPOT (ms)", min_value=0, value=30, step=1)
    user_scenario.throughput = col3.number_input("Min total token throughput (tokens/s)",
                                                 min_value=0,
                                                 step=1,
                                                 value=5000,
                                                 max_value=1000000,
                                                 )

    # TODO: what else?

# def misc(benchmark_data):



def get_pareto_front(df: pd.DataFrame) -> set[int]:
    """Get indices of rows on Pareto front.

    Args:
        df (pandas.DataFrame): DataFrame to get Pareto front for.

    Returns:
        set[int]: Indices of DataFrame that are on Pareto front.
    """
    pareto_set = set(df.index.tolist())
    for ii, rowa in df.iterrows():
        is_pareto_front = df.index.isin(pareto_set)
        for jj, rowb in df[is_pareto_front].iterrows():
            if ii == jj:
                continue
            if rowa.Thpt_per_User > rowb.Thpt_per_User and rowa.Thpt_per_GPU > rowb.Thpt_per_GPU:
                # Index jj worse in all ways to index ii
                pareto_set.remove(jj)
    return pareto_set

def pareto_plots(runs_selected):
    """
    Pareto plots
    """

    user_scenario = st.session_state['scenario']

    runs_filtered = runs_selected[
        (runs_selected.Mean_TTFT_ms <= user_scenario.ttft) &
        (runs_selected.Mean_TPOT_ms <= user_scenario.tpot) &
        (runs_selected.Total_Token_Throughput >= user_scenario.throughput)
    ]
    pareto_set = get_pareto_front(runs_selected)

    # Runs that meet scenario selection, but fail SLOs
    runs_fails_slo = runs_selected[~runs_selected.index.isin(runs_filtered.index.tolist())]

    # Runs that meet SLOs, but are not on the Pareto front
    runs_filtered_not_front = runs_filtered[~runs_filtered.index.isin(pareto_set)]

    # Runs on the Pareto front
    runs_pareto_front = runs_filtered[runs_filtered.index.isin(pareto_set)]

    # Plot
    # Create a figure and plot all three lines on the SAME graph
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(runs_pareto_front.Thpt_per_User, runs_pareto_front.Thpt_per_GPU,
         marker='o', markersize=4,
         color='#FF00FF',
         linestyle='',
         label='Pareto front (optimal)'
        )

    ax.plot(runs_filtered_not_front.Thpt_per_User, runs_filtered_not_front.Thpt_per_GPU,
         marker='o', markersize=4,
         color='#000000',
         linestyle='',
         label='Meets SLOs but non-optimal'
        )

    ax.plot(runs_fails_slo.Thpt_per_User, runs_fails_slo.Thpt_per_GPU,
         marker='o', markersize=4,
         color='#CCCCCC',
         linestyle='',
         label='Fails SLOs'
        )

    ax.set_xlabel('Tok/s/User', fontsize='16')
    ax.set_ylabel('Tok/s/GPU', fontsize='16')
    ax.grid(True, linewidth=1, ls='--', color='gray')
    ax.axis([0, None, 0, None])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    _, col, _ = st.columns([.5, 1, .5])
    with col:
        st.pyplot(fig)
    plt.show()

    return runs_pareto_front

def inputs():
    """
    Inputs to the BLIS simulator
    """

    st.header("Inputs")
    st.caption("Inputs to the BLIS simulator. Note that this page provides the regression model for estimating a selective set of models. Training a new model is straightforward.")

    # Model
    models = [
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-72B",
        "Qwen/Qwen2.5-14B",
        ]

    selected_model = st.selectbox("Select a model",
                 options=models
                 )


    if selected_model and selected_model != "":
        # Fetch model info
        model_info = get_model_info_from_hf(selected_model)
        model_config = get_model_config_from_hf(selected_model)
        model_gpu_memory_req = round(model_memory_req(model_info), 2)

        # Display first precision
        col1, col2 = st.columns(2)

        col1.info(f"Size of model in memory: ~{model_gpu_memory_req} GB")
        with col2.expander("See how model size is calculated below"):
            st.write("""Below shows how model memory is estimated. The number of parameters and precision are fetched from Hugging Face. Common data types include `BF16` (floating point 16-bit) and `F8_E4M3` (floating point 8-bit, 4 for exponents and 3 for mantissa). The total is then summed.""")

            data_types = []
            bytes_list = []
            params = []
            memory_req = []

            for d_type, param in model_info.safetensors.parameters.items():
                data_types.append(d_type)
                params.append(param)

                try:
                    bytes_list.append(precision_to_byte(d_type))
                except Exception as e:
                    st.warning(e)
                    pass

                memory_req.append(parameter_memory_req(param, d_type))

            data = {
                "Data type": data_types,
                "Size in bytes": bytes_list,
                "Number of parameters": params,
                "Memory in GB (params x bytes)": memory_req,
            }
            st.dataframe(data, hide_index=True)

            st.write("In addition, vLLM [profiles memory](https://github.com/vllm-project/vllm/blob/dcf2f3ec067711ff69e5ab7478fca6ffb4f11daf/vllm/worker/worker.py#L229) by doing a forward pass with `--max-model-len` with dummy data to estimate the non-torch and torch activation peak memory consumption. This means the estimation of the model memory is actually an underestimation. Estimating intermediate memory footprint is currently work in progress.")

    # Scenario
    with st.container(border=True):
        st.write("**Workload/Application**")
        st.caption("Define the type of workload for the LLM. Select from a set of pre-defined workloads or tune each parameter based on your need.")

        preset_scenarios = {
            "Chatbot": {
                "dataset": "shareGPT",
                "request_rate": 25,
                "input_len": 512,
                "output_len": 128,
                "prefix_hit_ratio": 30,
                "latency_p50": 10,
                "latency_p90": 100,
                "throughput": 100,
                "ttft": 200,
                "itl": 50,
                },
            "Summarization": {
                "dataset": "Summary Dataset",
                "request_rate": 5,
                "input_len": 2048,
                "output_len": 256,
                "prefix_hit_ratio": 5,
                "latency_p50": 100,
                "latency_p90": 1000,
                "throughput": 1000,
                "ttft": 2000,
                "itl": 500,
                },
            "Classification": {
                "dataset": "Text Dataset",
                "request_rate": 5,
                "input_len": 3096,
                "output_len": 32,
                "prefix_hit_ratio": 5,
                "latency_p50": 1,
                "latency_p90": 10,
                "throughput": 10,
                "ttft": 20,
                "itl": 5,
                },
            "Custom": {
                "dataset": "Synthetic",
                "request_rate": 1,
                "input_len": 1,
                "output_len": 1,
                "prefix_hit_ratio": 1,
                "latency_p50": 10,
                "latency_p90": 100,
                "throughput": 100,
                "ttft": 200,
                "itl": 50,
                }
        }

        datasets = [
            "shareGPT",
            "Summary Dataset",
            "Text Dataset",
            "Synthetic",
            "Dataset 5",
            "Dataset 6",
        ]

        col1, col2 = st.columns([0.3, 0.7])

        selected_workload = col1.radio("Select workload",
                     options=preset_scenarios.keys())

        if selected_workload:
            scenario = preset_scenarios[selected_workload]
            with col2:
                disabled = selected_workload != "Custom"

                st.selectbox("Dataset",
                             options=datasets,
                             index=datasets.index(scenario['dataset']),
                             disabled=disabled,
                             )

                st.number_input("Request rate per second",
                          step=1,
                          min_value=1,
                          value=scenario['request_rate'],
                          disabled=disabled,
                          )

                st.number_input("Input sequence length",
                        step=1,
                        min_value=1,
                        value=scenario['input_len'],
                        disabled=disabled,
                            )

                st.number_input("Output sequence length",
                        step=1,
                        min_value=1,
                        value=scenario['output_len'],
                        disabled=disabled,
                            )

                st.slider("Prefix hit ratio (%)",
                        step=1,
                        min_value=1,
                        max_value=100,
                        value=scenario['prefix_hit_ratio'],
                        disabled=disabled,
                            )

    # SLOs
    with st.container(border=True):
        st.write("**Goals / SLOs**")
        st.caption("Define the desire constraints to reach for your application.")

        if selected_workload:
            scenario = preset_scenarios[selected_workload]
            disabled = selected_workload != "Custom"

            latency_p50 = st.number_input("Latency p50 (ms)",
                                          value=scenario['latency_p50'],
                                          )
            latency_p95 = st.number_input("Latency p95 (ms)",
                                          value=scenario['latency_p90'],
                                          )
            throughput = st.number_input("Throughput (token/s)",
                                         value=scenario['throughput'],
                                         )
            ttft = st.number_input("TTFT (ms)",
                                   value=scenario['ttft'],
                                   )
            itl = st.number_input("ITL (ms)",
                                  value=scenario['itl'],
                                  )




    # vLLM parameters
    with st.container(border=True):
        st.write("**vLLM parameters**")
        st.caption("Select what vLLM engine arguments to sweep over and simulate.")

        # Keep consistent column proportions and vertical alignment to ensure clean rows.
        COL_SPEC = [1.2, 3.8]

        # -----------------------------
        # GPU Memory Utilization (%)
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_gpu_mem = st.checkbox(
                "GPU Memory Utilization (%)",
                value=True,
                help="Comma/space/semicolon-separated integers (1–100)."
            )
        default_gpu_mem_text = "50, 60, 70, 80, 90"
        with c2:
            gpu_mem_text = st.text_input(
                "Values to sweep (comma/space separated)",
                value=default_gpu_mem_text,
                key="gpu_mem_text",
                disabled=not sweep_gpu_mem,
                label_visibility="collapsed",
                help="Example: 50,60,70,80,90"
            )
            gpu_memory_utilization_all = parse_numeric_list(
                gpu_mem_text, cast=int, fallback=[50, 60, 70, 80, 90]
            )
            # Constrain to 1..100
            gpu_memory_utilization_all = [v for v in gpu_memory_utilization_all if 1 <= v <= 100]
            if sweep_gpu_mem and not gpu_memory_utilization_all:
                st.warning("Please provide at least one integer between 1 and 100.")
            # If not sweeping, select a single value (first or fallback)
            gpu_memory_utilization = (
                gpu_memory_utilization_all if sweep_gpu_mem
                else first_or_fallback(gpu_memory_utilization_all, 50)
            )

        # -----------------------------
        # Block Size
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_block_size = st.checkbox(
                "Block Size",
                value=True,
                help="Select one or more vLLM-acceptable block sizes (tokens per KV block)."
            )
        ACCEPTABLE_BLOCK_SIZES = [8, 16, 32, 64]  # Adjust if your vLLM build supports others
        default_block_sizes = ACCEPTABLE_BLOCK_SIZES[:]  # all selected by default
        with c2:
            block_size_selected = st.multiselect(
                "Block sizes",
                options=ACCEPTABLE_BLOCK_SIZES,
                default=default_block_sizes,
                key="block_sizes",
                disabled=not sweep_block_size,
                label_visibility="collapsed"
            )
            # If the user somehow clears selection while sweeping, restore defaults
            if sweep_block_size and not block_size_selected:
                st.info("No block sizes selected. Disable this parameter to sweep if that is what you intended, otherwise, use all acceptable block sizes by default.")
                block_size_selected = ACCEPTABLE_BLOCK_SIZES[:]
            block_size = (
                block_size_selected if sweep_block_size
                else first_or_fallback(block_size_selected, 16)
            )

        # -----------------------------
        # Max Num Batched Tokens
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_max_tokens = st.checkbox(
                "Max Num Batched Tokens",
                value=True,
                help="Comma/space/semicolon-separated integers."
            )
        default_max_tokens_text = "256, 512, 1024, 2048"
        with c2:
            max_tokens_text = st.text_input(
                "Values to sweep (comma/space separated)",
                value=default_max_tokens_text,
                key="max_tokens_text",
                disabled=not sweep_max_tokens,
                label_visibility="collapsed",
                help="Example: 256,512,1024,2048"
            )
            max_num_batched_tokens_all = parse_numeric_list(
                max_tokens_text, cast=int, fallback=[256, 512, 1024, 2048]
            )
            if sweep_max_tokens and not max_num_batched_tokens_all:
                st.warning("Please provide at least one integer for Max Num Batched Tokens.")
            max_num_batched_tokens = (
                max_num_batched_tokens_all if sweep_max_tokens
                else first_or_fallback(max_num_batched_tokens_all, 512)
            )

        # -----------------------------
        # Long Prefill Token Threshold
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_long_prefill = st.checkbox(
                "Long Prefill Token Threshold",
                value=True,
                help="Comma/space/semicolon-separated integers."
            )
        default_long_prefill_text = "256, 512, 1024, 2048"
        with c2:
            long_prefill_text = st.text_input(
                "Values to sweep (comma/space separated)",
                value=default_long_prefill_text,
                key="long_prefill_text",
                disabled=not sweep_long_prefill,
                label_visibility="collapsed",
                help="Example: 256,512,1024,2048"
            )
            long_prefill_token_threshold_all = parse_numeric_list(
                long_prefill_text, cast=int, fallback=[256, 512, 1024, 2048]
            )
            if sweep_long_prefill and not long_prefill_token_threshold_all:
                st.warning("Please provide at least one integer for Long Prefill Token Threshold.")
            long_prefill_token_threshold = (
                long_prefill_token_threshold_all if sweep_long_prefill
                else first_or_fallback(long_prefill_token_threshold_all, 512)
            )

        # -----------------------------
        # Enable Prefix Caching
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_prefix_cache = st.checkbox(
                "Enable Prefix Caching",
                value=True,
                help="Sweep over True/False to evaluate effect of prefix caching."
            )
        with c2:
            # Two multiselect values True/False, both selected by default
            prefix_options = [True, False]
            prefix_selected = st.multiselect(
                "Enable Prefix Caching values",
                options=prefix_options,
                default=prefix_options,  # both selected by default
                key="prefix_caching_values",
                disabled=not sweep_prefix_cache,
                label_visibility="collapsed"
            )
            if sweep_prefix_cache and not prefix_selected:
                st.info("No selection made. Using both True and False by default.")
                prefix_selected = prefix_options[:]
            enable_prefix_caching = (
                prefix_selected if sweep_prefix_cache
                else [True]  # default fixed value when not sweeping
            )


        # -----------------------------
        # Tensor Parallelism (tp)
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_tp = st.checkbox(
                "Tensor Parallelism (constrained by number of attention heads)",
                value=True,
                help="Comma/space/semicolon-separated integers ≥ 1. Example: 1, 2, 4"
            )
        default_tp_text = "1, 2, 4"
        with c2:
            tp_text = st.multiselect(
                "TP Values to sweep (comma/space separated)",
                options=find_possible_tp(model_config),
                default=find_possible_tp(model_config),
                key="tp_text",
                disabled=not sweep_tp,
                label_visibility="collapsed",
                help="Example: 1,2,4"
            )
            tp_all = [v for v in parse_numeric_list(tp_text, cast=int, fallback=[1, 2, 4]) if v >= 1]
            if sweep_tp and not tp_all:
                st.warning("Please provide at least one integer ≥ 1 for Tensor Parallelism (tp).")
            tp = tp_all if sweep_tp else first_or_fallback(tp_all, 1)

        # -----------------------------
        # Data Parallelism (dp)
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_dp = st.checkbox(
                "Data Parallelism (constrained by number of hidden layers)",
                value=True,
                help="Comma/space/semicolon-separated integers ≥ 1. Example: 1, 2"
            )
        with c2:
            dp_text = st.multiselect(
                "DP Values to sweep (comma/separated)",
                options=[i for i in range(1, model_config.num_hidden_layers)],
                default=[i for i in range(1, model_config.num_hidden_layers)],
                key="dp_text",
                disabled=not sweep_dp,
                label_visibility="collapsed",
                help="Example: 1,2"
            )
            dp_all = [v for v in parse_numeric_list(dp_text, cast=int, fallback=[1, 2]) if v >= 1]
            if sweep_dp and not dp_all:
                st.warning("Please provide at least one integer ≥ 1 for Data Parallelism (dp).")
            dp = dp_all if sweep_dp else first_or_fallback(dp_all, 1)

        # -----------------------------
        # Pipeline Parallelism (pp)
        # -----------------------------
        c1, c2 = st.columns(COL_SPEC, vertical_alignment="center")
        with c1:
            sweep_pp = st.checkbox(
                "Pipeline Parallelism (replicas of model)",
                value=True,
                help="Comma/space/semicolon-separated integers ≥ 1. Example: 1, 2"
            )
        default_pp_text = "1, 2"
        with c2:
            pp_text = st.text_input(
                "Values to sweep (comma/space separated)",
                value=default_pp_text,
                key="pp_text",
                disabled=not sweep_pp,
                label_visibility="collapsed",
                help="Example: 1,2"
            )
            pp_all = [v for v in parse_numeric_list(pp_text, cast=int, fallback=[1, 2]) if v >= 1]
            if sweep_pp and not pp_all:
                st.warning("Please provide at least one integer ≥ 1 for Pipeline Parallelism (pp).")
            pp = pp_all if sweep_pp else first_or_fallback(pp_all, 1)




    # Environment & Hardware Section
    with st.container(border=True):
        st.write("**Environment & Hardware**")

        # GPU Configuration
        gpu_type = st.selectbox("Accelerator Type", db.gpu_specs.keys())
        total = st.number_input("Total number of GPUs (this will filter out parallelism options that are invalid)", min_value=1)

def output():
    st.button("Simulate performance results")

if __name__ == "__main__":
    st.title("Performance Estimation")
    st.caption("For unseen configurations, suggest optimal configuration by estimating performance using the BLIS simulator.")

    util.init_session_state()

    inputs()
    output()