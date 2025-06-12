# Agent class for simplicity
from tools import (
    readFile, processBugReportContent, preprocess_text, load_stopwords, processBugReportQueryKeyBERT, processBugReportQueryReasoning,
    processBugReportContentPostReasoning, processBugReportQueryReasoningReflectOnResults,
    index_source_code, bug_localization_BM25_and_FAISS, get_short_filename
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
            elif self.name == "process_bug_report_query_reasoning_agent":
                tool_output = self.tools["processBugReportQueryReasoning"](*args)
            elif self.name == "process_bug_report_content_agent_post_reasoning":
                tool_output = self.tools["processBugReportContentPostReasoning"](*args)
            elif self.name == "process_bug_report_query_reasoning_reflects_on_results_agent":
                tool_output = self.tools["processBugReportQueryReasoningReflectOnResults"] (*args)
            elif self.name == "index_source_code_agent":
                tool_output = self.tools["index_source_code"](*args)
            elif self.name == "bug_localization_BM25_and_FAISS_agent":
                tool_output = self.tools["bug_localization_BM25_and_FAISS"](*args)
            else:
                tool_output = "Unknown agent."

            return {self.output_key: tool_output}

        except Exception as e:
            return {self.output_key: f"Error: {e}"}
        


# Initialize the agents
try:
    MY_MODEL = completion  # liteLLM wrapper (can be your own callable model)
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

    processBugReportContentPostReasoning_agent = Agent(
        model=MY_MODEL,
        name="process_bug_report_content_agent_post_reasoning",
        instruction="You are the processBugReportContentPostReasoning Agent."
                    "You will receive the output ('result') of the 'processBugReportQueryReasoning_agent'."
                    #"Your task is to discard 'Main issue:' from the first sentence and 'Functionality:' from the second sentece, and return the rest of the content as a string."
                    "Your task is to process that content and return it as a string."
                    "Use the 'processBugReportContentPostReasoning' tool to perform this process. ",  
        tools=[processBugReportContentPostReasoning, preprocess_text, load_stopwords],
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
    
    processBugReportQueryReasoning_agent = Agent(
        model=MY_MODEL,     
        name="process_bug_report_query_reasoning_agent",
        instruction="You are the ProcessBugReportQueryReasoning Agent."
                    "You will receive the output ('result') of the 'readBugReportContent_agent'."
                    "Your ONLY task is to process that content using LLM and return it as a string."
                    "Use the 'processBugReportQueryReasoning' tool to perform this action. ",
        tools=[processBugReportQueryReasoning], # List of tools the agent can use     
        output_key="file_content" # Specify the output key for the tool's result
        )

    processBugReportQueryReasoningReflectOnResults_agent = Agent(
        model=MY_MODEL,     
        name="process_bug_report_query_reasoning_reflects_on_results_agent",
        instruction="You are the processBugReportQueryReasoningReflectOnResults Agent."
                    "You will receive the output ('bug_report_content') of the 'readBugReportContent_agent'."
                    "You will recive the output ('search_query') of the processBugReportQueryReasoning_agent"
                    "Your ONLY task is to process that content using LLM and return (yes or no) as a string."
                    "Use the 'processBugReportQueryReasoningReflectOnResults' tool to perform this action. ",
        tools=[processBugReportQueryReasoningReflectOnResults], # List of tools the agent can use     
        output_key="file_content"
        )

    index_source_code_agent = Agent(
        model=MY_MODEL,
        name="index_source_code_agent",
        instruction="You are the IndexSourceCode Agent. "
                    "You will receive a folder path ('source_code_dir') which is the path to the source code. "
                    "You will also receive a folder path ('source_code_index_path') which is the path to the source code index. "
                    "Your ONLY task is to index the source code using the 'index_source_code' tool. "
                    "Your output will be the a BM25 index ('bm25_index') and a FAISS index ('faiss_index') and the  processed documents ('processed_documents') from the 'index_source_code' tool.",
        tools=[index_source_code],
        output_key="file_content"
    )
    
    bug_localization_BM25_and_FAISS_agent = Agent(
        model=MY_MODEL,
        name="bug_localization_BM25_and_FAISS_agent",
        instruction="You are the BugLocalizationBM25AndFAISS Agent. "
                    "You will receive the output ('result') of the 'processBugReportContent_agent'. "
                    "You will also receive a number ('top_n') which is the number of keywords to extract. "
                    "You will also receive a BM25 index ('bm25_index') and a FAISS index ('faiss_index') from the 'load_index_bm25_and_faiss_agent'. "
                    "Your ONLY task is to localize the bug report using the 'bug_localization_BM25_and_FAISS' tool. "
                    "Format the output as a string with each result on a new line.",
        tools=[bug_localization_BM25_and_FAISS, get_short_filename],
        output_key="file_content"
    )
except Exception as e:
    print(f"Self-test failed for agent setup. Error: {e}")
