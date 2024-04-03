import dataclasses
import datetime
import time
from typing import List, Tuple
import altair as alt
import gpuhunt
import gpuhunt.providers.vastai
import gpuhunt.providers.tensordock
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="KeenSight - GPU finder",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def fetch_gpu_catalog() -> gpuhunt.Catalog:
    catalog = gpuhunt.Catalog(balance_resources=True, auto_reload=True)
    catalog.load()
    catalog.add_provider(gpuhunt.providers.vastai.VastAIProvider())
    catalog.add_provider(gpuhunt.providers.tensordock.TensorDockProvider())
    return catalog


@st.cache_data(show_spinner=False)
def fetch_all_gpu_offers() -> List[gpuhunt.CatalogItem]:
    return fetch_gpu_catalog().query(provider=AVAILABLE_PROVIDERS)


AVAILABLE_PROVIDERS = ["aws", "azure", "datacrunch",
                       "gcp", "lambdalabs", "tensordock", "vastai"]
ALL_GPU_OFFERS = fetch_all_gpu_offers()
ALL_GPU_NAMES = sorted(
    set(offer.gpu_name for offer in ALL_GPU_OFFERS if offer.gpu_count > 0))
ALL_GPU_MEMORIES = [
    0.0] + sorted(set(offer.gpu_memory for offer in ALL_GPU_OFFERS if offer.gpu_count > 0))
ALL_GPU_COUNTS = sorted(set(offer.gpu_count for offer in ALL_GPU_OFFERS))
ALL_CPUS = sorted(set(offer.cpu for offer in ALL_GPU_OFFERS))
ALL_MEMORIES = sorted(set(offer.memory for offer in ALL_GPU_OFFERS))
CACHE_TTL = 5 * 60

DEFAULT_PROVIDERS = [
    provider for provider in AVAILABLE_PROVIDERS if provider not in ["vastai"]]
DEFAULT_GPUS = ["H100", "A100", "A6000", "A10",
                "A10G", "L40", "L4", "T4", "V100", "P100"]
DEFAULT_GPUS = [gpu for gpu in DEFAULT_GPUS if gpu in ALL_GPU_NAMES]


def format_gpu_version(version: str) -> str:
    return f"{version[:4]}/{version[4:6]}/{version[6:]}"


with st.sidebar:
    st.markdown("## Configuration")

    with st.expander("Providers", expanded=True):
        providers = st.multiselect(
            "Providers", options=AVAILABLE_PROVIDERS, default=DEFAULT_PROVIDERS)

    with st.expander("GPU", expanded=False):
        gpu_count = st.select_slider("GPU Count", options=ALL_GPU_COUNTS, value=(
            ALL_GPU_COUNTS[0], ALL_GPU_COUNTS[-1]))
        gpu_memory = st.select_slider("GPU Memory", options=ALL_GPU_MEMORIES, value=(
            ALL_GPU_MEMORIES[0], ALL_GPU_MEMORIES[-1]))
        gpu_names = st.multiselect(
            "GPU Name", options=ALL_GPU_NAMES, default=DEFAULT_GPUS)

    with st.expander("Instance", expanded=False):
        spot_type = st.radio(
            "Spot", options=["on-demand", "interruptable", "any"], index=0)
        cpu_range = st.select_slider(
            "CPUs", options=ALL_CPUS, value=(ALL_CPUS[0], ALL_CPUS[-1]))
        memory_range = st.select_slider(
            "RAM", options=ALL_MEMORIES, value=(ALL_MEMORIES[0], ALL_MEMORIES[-1]))


@st.cache_data(show_spinner=False)
def retrieve_gpu_offers(providers: List[str], gpu_count: Tuple[int, int], gpu_memory: List[float],
                        gpu_names: List[str], spot_type: str, cpu_range: Tuple[int], memory_range: Tuple[float], ttl: int) -> Tuple[datetime.datetime, pd.DataFrame]:
    # Handle the case where gpu_memory is None
    if gpu_memory is None:
        gpu_memory = [0.0, max(ALL_GPU_MEMORIES)]

    offers = fetch_gpu_catalog().query(
        provider=providers or AVAILABLE_PROVIDERS,
        min_gpu_count=gpu_count[0], max_gpu_count=gpu_count[1],
        min_gpu_memory=gpu_memory[0], max_gpu_memory=gpu_memory[1],
        min_cpu=cpu_range[0], max_cpu=cpu_range[-1],
        min_memory=memory_range[0], max_memory=memory_range[1],
        gpu_name=gpu_names or None,
        spot={"interruptable": True,
              "on-demand": False, "any": None}[spot_type],
    )
    updated_at = datetime.datetime.utcnow()
    df = pd.DataFrame([dataclasses.asdict(offer) for offer in offers])
    if not df.empty:
        df["gpu"] = df.apply(lambda row: "" if row.gpu_count ==
                             0 else f"{row.gpu_name} ({row.gpu_memory:g} GB)", axis=1)
        df = df[["provider", "price", "gpu_count", "gpu", "cpu",
                 "memory", "spot", "location", "instance_name"]]
    return updated_at, df


updated_at, df = retrieve_gpu_offers(providers, gpu_count, gpu_memory, gpu_names,
                                     spot_type, cpu_range, memory_range, round(time.time() / CACHE_TTL))

df_filtered = df[df['gpu_count'] > 0]

st.image('logo.png', width=200)
st.title("Discover the Best Cloud GPU Deals")
# Create an Altair Chart
chart = alt.Chart(df_filtered).mark_bar().encode(
    x='provider:N',
    y='price:Q',
    color='gpu:N',
    tooltip=['provider', 'price', 'gpu_count', 'gpu', 'cpu',
             'memory', 'spot', 'location', 'instance_name']
).properties(
    title='Price Comparison by Provider and GPU',
    width=800,
    height=400
)

# Display the chart
st.altair_chart(chart, use_container_width=True)

st.dataframe(
    df,
    column_config={
        "price": st.column_config.NumberColumn(format="$%.3f"),
    },
)
st.write(f"{len(df)} offers queried at",
         updated_at.strftime('`%Y-%m-%d %H:%M:%S UTC`'))
