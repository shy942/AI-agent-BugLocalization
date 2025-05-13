from query_constructions import construct_queries_for_project
import os

if __name__ == "__main__":
    project_id = "103"
    BugReportPath = os.path.expanduser("./ExampleProjectData/ProjectBugReports/")
    SearchQueryPath = os.path.expanduser("./ExampleProjectData/ConstructedQueries/")

    construct_queries_for_project(project_id, BugReportPath, SearchQueryPath)
