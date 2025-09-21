import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
import re
from datetime import datetime
import torch
from typing import TypedDict, List, Optional, Dict
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, Document
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
import json
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# ---------------- Device ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on:", device)

# ---------------- Models - Load once and cache ----------------
@st.cache_resource
def load_models():
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={"device": device}
    )
    print("Embedding model loaded.")
    
    groq_key = os.getenv('GROQ_KEY')
    # llm = ChatGroq(model='qwen/qwen3-32b', api_key=groq_key)
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key)
    print("LLM initialized.")
    
    print("Loading FAISS vector store...")
    VECTOR_PATH='./src/vector_store/faiss_index'
    vector_store = FAISS.load_local(VECTOR_PATH, embed_model, allow_dangerous_deserialization=True)
    print("Vector store loaded.")
    
    return embed_model, llm, vector_store

embed_model, llm, vector_store = load_models()

# ---------- System prompt ----------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are FloatChat, an expert oceanographer AI assistant specialized in ARGO float data analysis. 
Use retrieved metadata and float data to answer queries about temperature, salinity, DOXY, nitrate, chlorophyll, pH, backscatter.

Instructions:
1. Filter data by location, time, and depth if mentioned.
2. Provide numerical summaries, trends, comparisons where possible.
3. Include guidance on visualizations if relevant.
4. Avoid hallucination.
5. Format clearly with tables/lists if needed.
6. If you dont know the answer or not sure about it, be frank and respond honestly.
"""),
    ("user", "Question: {question}\n\nContext: {context}")
])

# General prompt for non-RAG queries
general_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are FloatChat, an expert oceanographer AI assistant specialized in ARGO float data analysis.
You help users query and analyze oceanographic data from ARGO floats, providing insights on temperature, salinity, and other parameters.
For general greetings or introductions, introduce yourself as FloatChat and explain your capabilities in ocean data analysis.
Be helpful, professional, and engaging.
     
     Note :Dont generate any code ,If you dont know the answer just say , you dont know in professional way,
     also.Only use your Vector store data.
"""),
    ("user", "{query}")
])

# ---------- Metadata extraction ----------
def extract_metadata(query: str):
    """Extract approximate latitude, longitude, year, month, depth from user query."""
    metadata_filter = {}

    lat_match = re.search(r'lat(?:itude)?\s*([-\d\.]+)', query, re.I)
    lon_match = re.search(r'lon(?:gitude)?\s*([-\d\.]+)', query, re.I)
    if lat_match:
        lat = float(lat_match.group(1))
        metadata_filter["latitude"] = lambda x: abs(x - lat) <= 5
    if lon_match:
        lon = float(lon_match.group(1))
        metadata_filter["longitude"] = lambda x: abs(x - lon) <= 5

    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        year = int(year_match.group(1))
        metadata_filter["year"] = lambda x: x == year

    months = {
        "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
        "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
    }
    for mname, mval in months.items():
        if mname in query.lower():
            metadata_filter["month"] = lambda x, mv=mval: x == mv
            break
    month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', query)
    if month_match:
        metadata_filter["month"] = lambda x, mv=int(month_match.group(1)): x == mv

    depth_match = re.search(r'depth\s*([\d\.]+)\s*(m|meters)?', query, re.I)
    if depth_match:
        depth = float(depth_match.group(1))
        metadata_filter["depth"] = lambda x: abs(x - depth) <= 10

    return metadata_filter

def hybrid_rag(query_text: str, vector_store: FAISS, llm, top_k: int = 5):
    """
    Hybrid RAG query with FAISS, metadata filtering, deduplication, and structured output.
    Returns:
        dict: {
            'query': str,
            'result': str,
            'source_documents': list of dicts with metadata
        }
    """
    # Extract metadata filters from query
    metadata_filter_dict = extract_metadata(query_text)

    # Create filter function
    def metadata_filter_func(metadata):
        for key, filt in metadata_filter_dict.items():
            value = metadata.get(key)
            if value is None or not filt(value):
                return False
        return True

    # Build FAISS retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k},
        filter=metadata_filter_func if metadata_filter_dict else None
    )

    # RetrievalQA chain with custom prompt
    chain_type_kwargs = {"prompt": chat_prompt}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    # Run query
    qa_result = qa_chain({"query": query_text})

    # Deduplicate source documents by (file_name, page_content)
    seen = set()
    unique_sources = []
    for doc in qa_result["source_documents"]:
        key = (doc.metadata.get("file_name"), doc.page_content)
        if key not in seen:
            seen.add(key)
            metadata = doc.metadata.copy()
            metadata.update({
                "latitude": metadata.get("latitude"),
                "longitude": metadata.get("longitude"),
                "depth": metadata.get("depth"),
                "date": metadata.get("date")
            })
            unique_sources.append({
                "metadata": metadata,
                "page_content": doc.page_content
            })

    return {
        "query": query_text,
        "result": qa_result["result"],
        "source_documents": unique_sources
    }

def parse_rag_for_analysis(rag_response: str, llm) -> dict:
    """
    LLM parses the RAG output and returns:
    1. structured dataframe-ready JSON
    2. suggested plots to create
    """
    prompt = f"""
You are an expert oceanographer and data visualization agent.

Input RAG response:

{rag_response}

Tasks:
   
1. Extract all tabular data into JSON format (list of objects).
   Include depth, latitude, longitude, float_file, date_time, salinity, temperature, nitrate, chlorophyll, pH, backscatter (if present). Use 'null' for missing values.
   Ensure all numerical fields like depth, temperature, salinity, etc., are extracted as pure numbers without symbols like '~' or approximations. Use null if not an exact number.
2. Analyze which variables make sense to visualize. Suggest plots only if the query involves analysis, trends, or explicitly requests visualizations. Possible plot types:
   - "depth_temperature": If depth and temperature data available for profiles.
   - "depth_salinity": If depth and salinity.
   - "depth_nitrate": Similarly for other variables like chlorophyll, pH, etc.
   - "latlon_temperature": If multiple locations, map with color for temperature.
   - "compare_salinity_temperature": Pairplot for comparing two or more variables.
   Suggest relevant plots based on data availability and query context.
   
Return valid JSON ONLY with two keys:
- "data": []  # array of dicts, each a data row
- "plots": []  # array of strings like ["depth_temperature", "latlon_salinity", "compare_salinity_temperature"]
Use null for None, true/false lowercase. No extra text.
"""
    response = llm([HumanMessage(content=prompt)])
    
    try:
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]
        
        parsed_json = json.loads(json_text)
        return parsed_json
    except Exception as e:
        print("Failed to parse JSON from LLM:", e)
        return {"data": [], "plots": []}

def agentic_plot(parsed: dict) -> List[str]:
    """
    Generates plots based on LLM-suggested 'plots' list from parsed RAG response.
    Handles depth, lat/lon, and variable comparisons.
    Returns list of base64 encoded PNG images.
    """
    data = parsed.get("data", [])
    plots = parsed.get("plots", [])
    
    if not data:
        return []
    
    df = pd.DataFrame(data)
    base64_list = []
    
    for plot_type in plots:
        if plot_type.startswith("depth_") and "depth" in df.columns:
            var = plot_type.replace("depth_", "")
            if var in df.columns:
                plot_df = df.copy()
                plot_df[var] = pd.to_numeric(plot_df[var], errors='coerce')
                plot_df["depth"] = pd.to_numeric(plot_df["depth"], errors='coerce')
                plot_df = plot_df.dropna(subset=[var, "depth"])
                if not plot_df.empty and len(plot_df) > 1:  # Ensure at least 2 points for plot
                    fig = plt.figure(figsize=(8, 6))
                    plt.plot(plot_df[var], plot_df["depth"], marker='o', linestyle='-')
                    plt.gca().invert_yaxis()
                    plt.xlabel(var.capitalize())
                    plt.ylabel("Depth (m)")
                    plt.title(f"{var.capitalize()} vs Depth")
                    plt.grid(True)
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    base64_list.append(base64.b64encode(buf.read()).decode('utf-8'))
                    plt.close(fig)
        
        elif plot_type.startswith("latlon_") and "latitude" in df.columns and "longitude" in df.columns:
            var = plot_type.replace("latlon_", "")
            if var in df.columns:
                plot_df = df.copy()
                plot_df[var] = pd.to_numeric(plot_df[var], errors='coerce')
                plot_df["latitude"] = pd.to_numeric(plot_df["latitude"], errors='coerce')
                plot_df["longitude"] = pd.to_numeric(plot_df["longitude"], errors='coerce')
                plot_df = plot_df.dropna(subset=[var, "latitude", "longitude"])
                if not plot_df.empty and len(plot_df) > 1:
                    fig = plt.figure(figsize=(8, 6))
                    sc = plt.scatter(plot_df["longitude"], plot_df["latitude"], c=plot_df[var], cmap="viridis", s=100)
                    plt.colorbar(sc, label=var.capitalize())
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.title(f"Map of {var.capitalize()}")
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    base64_list.append(base64.b64encode(buf.read()).decode('utf-8'))
                    plt.close(fig)
        
        elif plot_type.startswith("compare_"):
            vars_to_compare = plot_type.replace("compare_", "").split("_")
            existing_vars = [v for v in vars_to_compare if v in df.columns]
            if len(existing_vars) >= 2:
                plot_df = df.copy()
                for v in existing_vars:
                    plot_df[v] = pd.to_numeric(plot_df[v], errors='coerce')
                plot_df = plot_df.dropna(subset=existing_vars)
                if not plot_df.empty and len(plot_df) > 1:
                    pairplot = sns.pairplot(plot_df[existing_vars], height=3)
                    fig = pairplot.fig
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    base64_list.append(base64.b64encode(buf.read()).decode('utf-8'))
                    plt.close(fig)
    
    return base64_list

# Response prompt for final summary
response_prompt = ChatPromptTemplate.from_template(
    "You are FloatChat, an expert on Argo float data. Respond to the user query: {query}\n"
    "If relevant data is provided, summarize it professionally. Data: {data}"
)

# LangGraph State
class AgentState(TypedDict):
    query: str
    rag_result: Optional[Dict]
    parsed: Optional[Dict]
    plot_base64s: List[str]
    response: str

# Nodes
def rag_node(state: AgentState) -> AgentState:
    state["rag_result"] = hybrid_rag(state["query"], vector_store, llm, top_k=5)
    return state

def parse_node(state: AgentState) -> AgentState:
    rag_response = state["rag_result"]["result"]
    state["parsed"] = parse_rag_for_analysis(rag_response, llm)
    return state

def plot_node(state: AgentState) -> AgentState:
    if state.get("parsed"):
        state["plot_base64s"] = agentic_plot(state["parsed"])
    return state

def response_node(state: AgentState) -> AgentState:
    if state.get("rag_result"):
        data = state.get("parsed", {}).get("data", []) or state["rag_result"]["result"]
        data_str = pd.DataFrame(data).to_string() if isinstance(data, list) else str(data)
        chain = response_prompt | llm | StrOutputParser()
        state["response"] = chain.invoke({"query": state["query"], "data": data_str})
    else:
        chain = general_prompt | llm | StrOutputParser()
        state["response"] = chain.invoke({"query": state["query"]})
    return state

# Router
def router_node(state: AgentState) -> str:
    query_lower = state["query"].lower()
    data_keywords = ["argo", "float", "data", "temperature", "salinity", "depth", "retrieve", "chlorophyll", "ph", "nitrate"]
    if any(word in query_lower for word in data_keywords):
        return "rag"
    return "respond"

# Build the graph - Cache the compiled app
@st.cache_resource
def build_workflow():
    workflow = StateGraph(state_schema=AgentState)
    
    workflow.add_node("rag", rag_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("plot", plot_node)
    workflow.add_node("respond", response_node)

    workflow.add_edge("rag", "parse")
    workflow.add_edge("parse", "plot")
    workflow.add_edge("plot", "respond")
    workflow.add_edge("respond", END)

    workflow.set_conditional_entry_point(router_node, {"rag": "rag", "respond": "respond"})
    
    return workflow.compile()

app = build_workflow()

# Streamlit UI - Professional Interface
st.set_page_config(
    page_title="FloatChat - ARGO Ocean AI", 
    page_icon="🌊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Initialize session state for query processing
if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar for input and controls
with st.sidebar:
    st.header("FloatChat")
    # st.markdown("**ARGO Ocean Data Analysis**")
    st.warning("This is a prototype contains only 2025 Argo Data.")

    st.markdown("by WaveLens")
    
    st.subheader("Query")
    query = st.text_area(
        "Ask about ARGO data:", 
        height=150, 
        placeholder="e.g., What is the temperature profile at latitude 40N in 2023?"
    )
    
    if st.button("Analyze", type="primary", disabled=st.session_state.processing):
        if query.strip() and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.current_query = query
            st.rerun()
        elif not query.strip():
            st.warning("Please enter a query.")

# Main area for results
st.header("FloatChat: ARGO Ocean Data Assistant")
st.markdown("Query ARGO float data in natural language for insights, summaries, and visualizations.")

# --- Process query if there's a new one ---
if st.session_state.get("current_query") and st.session_state.processing:
    query = st.session_state.current_query
    col1, col2 = st.columns([2, 1])
    
    # --- Show user query ---
    # with col1:
        # st.subheader("Your Query")
        # st.markdown(f"<div class='user-message'>{query}</div>", unsafe_allow_html=True)


    
    # --- Thinking / Progress ---

    progress_placeholder = st.empty()
    thinking_steps = ["Initializing ARGO analysis..."]
    progress_placeholder.markdown("  \n".join(thinking_steps))
        
    inputs = {"query": query}
    current_state = inputs.copy()
        
    for event in app.stream(inputs):
            for node_name, update in event.items():
                if node_name == "rag":
                    thinking_steps.append("↘ Retrieving relevant float data...")
                elif node_name == "parse":
                    thinking_steps.append("↘ Parsing oceanographic data...")
                elif node_name == "plot":
                    thinking_steps.append("↘ Generating visualizations...")
                elif node_name == "respond":
                    thinking_steps.append("↘ Formulating response...")
                
                # Update thinking display
                progress_placeholder.markdown("  \n".join(thinking_steps))
                current_state = {**current_state, **update}
        
    output = current_state
    progress_placeholder.empty()
    
    # --- Assistant Response Streaming ---
    
    st.subheader("FloatChat Response :-")
    response_text = ""
    response_placeholder = st.empty()
        
        # Use RAG result if available
    if output.get("rag_result"):
            data = output.get("parsed", {}).get("data", []) or output["rag_result"]["result"]
            data_str = pd.DataFrame(data).to_string() if isinstance(data, list) else str(data)
            chain = response_prompt | llm | StrOutputParser()
            for chunk in chain.stream({"query": query, "data": data_str}):
                if isinstance(chunk, str):
                    response_text += chunk
                    content_html = response_text.replace("\n", "  \n")
                    response_placeholder.markdown(
                        f"<div class='assistant-message'>{content_html}</div>",
                        unsafe_allow_html=True
                    )
    else:
            chain = general_prompt | llm | StrOutputParser()
            for chunk in chain.stream({"query": query}):
                if isinstance(chunk, str):
                    response_text += chunk
                    content_html = response_text.replace("\n", "  \n")
                    response_placeholder.markdown(
                        f"<div class='assistant-message'>{content_html}</div>",
                        unsafe_allow_html=True
                    )
    
    output["response"] = response_text
    
    # --- Display Data Table ---
    if output.get("parsed") and output["parsed"].get("data"):
        df = pd.DataFrame(output["parsed"]["data"])
        csv = df.to_csv(index=False).encode("utf-8")
        
        st.subheader("📊 Data Table")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", csv, "argo_data.csv", "text/csv")
    
    # --- Display Plots ---
    if output.get("plot_base64s"):
        
            st.subheader("📈 Visualizations")
            for img_b64 in output["plot_base64s"]:
                st.image(base64.b64decode(img_b64), use_container_width=True)
    
    # --- Reset state ---
    st.session_state.processing = False
    st.session_state.current_query = None
