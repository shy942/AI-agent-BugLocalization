# Example run
from agents import readBugReportContent_agent, processBugReportContent_agent, processBugRepotQueryKeyBERT_agent

if __name__ == "__main__":
    folder_path = "./example_bug_reports"
    filename = "103-226.txt"

    # Step 1: Read the bug report content
    read_result = readBugReportContent_agent.run(folder_path, filename)
    file_content = read_result.get("file_content", "")
    print("\n Bug Report Content:\n", file_content)

    # Step 2: Process the bug report content
    processed_result = processBugReportContent_agent.run(file_content)
    file_content = processed_result.get("file_content", "")
    print("\n Processed Bug Report Content:\n", processed_result["file_content"])
    #print("\n Processed Bug Report Content:\n", file_content)

    #step 3: Process the bug report content using KeyBERT
    keywords_result = processBugRepotQueryKeyBERT_agent.run(file_content)    
    file_content = keywords_result.get("file_content", "")  
    print("\n Processed Bug Report Content:\n", file_content) 
    #query=file_content
    #print("\n Keywords from Bug Report Content:\n", keywords_result["file_content"])