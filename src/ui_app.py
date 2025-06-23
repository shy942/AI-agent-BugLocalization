# ui_app.py
import streamlit as st
import os
import asyncio
from pipeline_runner import process_selected_projects, get_available_projects
import zipfile
import io

st.set_page_config(page_title="Bug Localization UI", layout="wide")
st.title("🧠 Bug Localization Agent Pipeline")

# --- Project Selection ---
#selected = st.multiselect("Select Projects to Process", PROJECTS)
selected = st.multiselect("Select Projects", get_available_projects())
if st.button("🚀 Run Pipeline on Selected Projects"):
    if not selected:
        st.warning("Please select at least one project.")
    else:
        st.info(f"Processing projects: {', '.join(selected)}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_selected_projects(selected))
        st.success("✅ Done processing selected projects.")

# --- View Log Output ---
if st.button("📄 Show Log"):
    log_path = "./logs/parallel_logs/reason_log.txt"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            st.text_area("Log", f.read(), height=300)
    else:
        st.error("Log file not found.")

# --- Browse Outputs ---
st.subheader("📂 Browse Output Files")

output_section = st.radio("Choose output type", ["ConstructedQueries", "SearchResults"])
#project_to_view = st.selectbox("Choose a project", PROJECTS, key="output_project")
project_to_view = st.selectbox("Choose a project", get_available_projects(), key="output_project")

base_dir = {
    "ConstructedQueries": f"./AgentProjectData/ConstructedQueries/BaselineVsReason/{project_to_view}_no_stem",
    "SearchResults": f"./AgentProjectData/SearchResults/{project_to_view}"
}[output_section]

if os.path.exists(base_dir):
    files = os.listdir(base_dir)
    if files:
        selected_file = st.selectbox("Choose a file to view", files)
        if st.button("📖 View File"):
            with open(os.path.join(base_dir, selected_file), "r", encoding="utf-8") as f:
                st.text_area("File Content", f.read(), height=400)

        if st.button("⬇️ Download File"):
            with open(os.path.join(base_dir, selected_file), "rb") as f:
                st.download_button(label="Download", data=f, file_name=selected_file)
    else:
        st.info("No output files found yet.")
else:
    st.warning("Output directory doesn't exist yet.")
