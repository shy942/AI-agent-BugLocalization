# Agent class for simplicity
from tools import (
    readFile, processBugReportContent, preprocess_text, load_stopwords, processBugReportQueryKeyBERT, 
    index_source_code, bug_localization_BM25_and_FAISS, load_index_bm25_and_faiss
)
from litellm import completion
from typing import Callable

class Agent:
    def __init__(self, model: Callable, name: str, instruction: str, tools: list, output_key: str):
        self.model = model
        self.name = name
        self.instruction = instruction
        self.tools = {tool.__name__: tool for tool in tools}
        self.output_key = output_key

    def run(self, *args):
        try:
            # Construct prompt
            full_prompt = f"""{self.instruction}

User Input:
Arguments: {args}

Now decide which tool to use.
"""
            print(f"Sending prompt to LLM:\n{full_prompt}\n")

            # Simulate choosing a tool based on the instruction
            if self.name == "readBugReportContent_agent":
                tool_output = self.tools["readFile"](*args)
            elif self.name == "process_bug_report_content_agent":
                tool_output = self.tools["processBugReportContent"](*args)
            elif self.name == "process_bug_report_query_keybert_agent":
                tool_output = self.tools["processBugReportQueryKeyBERT"](*args)
            elif self.name == "index_source_code_agent":
                tool_output = self.tools["index_source_code"](*args)
            elif self.name == "load_index_bm25_and_faiss_agent":
                tool_output = self.tools["load_index_bm25_and_faiss"](*args)
            elif self.name == "bug_localization_BM25_and_FAISS_agent":
                tool_output = self.tools["bug_localization_BM25_and_FAISS"](*args)
            #elif self.name == "hybrid_retrieve_agent":
            #    tool_output = self.tools["hybrid_retrieve"](*args)
            else:
                tool_output = "Unknown agent."

            return {self.output_key: tool_output}

        except Exception as e:
            return {self.output_key: f"Error: {e}"}
        


# Initialize the agents
try:
    MY_MODEL = completion  # liteLLM wrapper (can be your own callable model)
    #need to update this block coz now readFile only takes the directory path
    readBugReportContent_agent = Agent(
        model=MY_MODEL,
        name="readBugReportContent_agent",
        instruction="""You are the ReadBugReportContent Agent.
        The user will provide a folder path. Your task is to read the content of the files inside the folder.
        Use the tool 'readFile' to do this and return the file content as a string.""",
        tools=[readFile],
        output_key="file_content"
    )

    processBugReportContent_agent = Agent(
        model=MY_MODEL,
        name="process_bug_report_content_agent",
        instruction="You are the ProcessBugReportContent Agent."
                    "You will receive the output ('result') of the 'readBugReportContent_agent'."
                    "Your ONLY task is to process that content and return it as a string."
                    "Use the 'processBugReportContent' tool to perform this action. ",
        tools=[processBugReportContent, preprocess_text, load_stopwords],
        output_key="file_content"
    )

    processBugReportQueryKeyBERT_agent = Agent(       
        model=MY_MODEL,
        name="process_bug_report_query_keybert_agent",          
        instruction="You are the ProcessBugReportQueryKeyBERT Agent."
                    "You will receive the output ('result') of the 'processBugReportContent_agent'."
                    "You will also receive a number ('top_n') which is the number of keywords to extract."
                    "Your ONLY task is to process that content and return it as a string."
                    "Use the 'processBugReportQueryKeyBERT' tool to perform this action. ",    
        tools=[processBugReportQueryKeyBERT], # List of tools the agent can use
        output_key="file_content" # Specify the output key for the tool's result
        )
    
    index_source_code_agent = Agent(
        model=MY_MODEL,
        name="index_source_code_agent",
        instruction="You are the IndexSourceCode Agent."
                    "You will receive a folder path ('source_code_dir') which is the path to the source code."
                    "Your ONLY task is to index the source code using the 'index_source_code' tool. ",
        tools=[index_source_code],
        output_key="file_content"
    )
    
    load_index_bm25_and_faiss_agent = Agent(
        model=MY_MODEL,
        name="load_index_bm25_and_faiss_agent",
        instruction="You are the LoadIndexBM25AndFAISS Agent."
                    "You will receive a path to the BM25 index ('bm25_index_path') and a path to the FAISS index ('faiss_index_path')."
                    "Your ONLY task is to load the BM25 and FAISS indexes using the 'load_index_bm25_and_faiss' tool. ",    
        tools=[load_index_bm25_and_faiss],
        output_key="file_content"
    )

    bug_localization_BM25_and_FAISS_agent = Agent(
        model=MY_MODEL,
        name="bug_localization_BM25_and_FAISS_agent",
        instruction="You are the BugLocalizationBM25AndFAISS Agent."
                    "You will receive the output ('result') of the 'processBugReportContent_agent'."
                    "You will also receive a number ('top_n') which is the number of keywords to extract."
                    "You will also receive a BM25 index ('bm25_index') and a FAISS index ('faiss_index') from the 'load_index_bm25_and_faiss_agent'."
                    "Your ONLY task is to localize the bug report using the 'bug_localization_BM25_and_FAISS' tool. ",
        tools=[bug_localization_BM25_and_FAISS],
        output_key="file_content"
    )
except Exception as e:
    print(f"Self-test failed for agent setup. Error: {e}")
