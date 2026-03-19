from typing import Dict, Any

import tool_cvxpy
from load_case import load_case_data
import memory_tool
import visualize


class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

        search_description = (
            "Use this tool ONLY ONCE at the beginning to fetch raw parameters if the user specifies a standard case "
            "(e.g., 'IEEE14'). Once you receive the 'System Data' table in the Observation, DO NOT call this tool again. "
            "Instead, proceed immediately to parsing the data and calling 'cvxpy'. "
            "Input should be a string in the format 'IEEE' followed by the number of buses (e.g., 'IEEE14' or 'IEEE118')."
        )
        self.registerTool("load_case", search_description, load_case_data)

        cvxpy_description = (
            "Invoke this tool to perform the final numerical optimization after you have gathered or updated the "
            "structured JSON parameters. This tool interfaces with a professional mathematical solver to calculate "
            "the optimal power dispatch values and the minimum total cost. "
            "The input must be a valid JSON object strictly matching the required schema."
        )
        self.registerTool("cvxpy", cvxpy_description, tool_cvxpy.cvxpy)

        memory_description = """Memory tool for storing and retrieving conversation history and knowledge.
Supported actions and their parameters:
- add         : Add a memory. Requires memory_content(str), memory_type(working, default), importance(0.0-1.0, default 0.5)
- search      : Search relevant memories. Requires query(str), optional limit(int, default 5), memory_type(str)
- summary     : Get memory summary and statistics. Optional limit(int, default 10)
- stats       : Get memory system statistics. No extra parameters needed
- update      : Update an existing memory. Requires memory_id(str), optional memory_content(str), importance(float)
- remove      : Delete a specific memory. Requires memory_id(str)
- forget      : Bulk-forget low-value memories. Optional strategy(importance_based/time_based, default importance_based), threshold(float, default 0.1)
- clear_all   : Clear all memories (destructive operation)

Usage tips:
- Use 'search' before answering to retrieve relevant context
- Use 'add' with importance >= 0.7 to save important dispatch parameters or results"""

        self.registerTool("memory", memory_description, memory_tool.memory)

        vis_description = """Generate Economic Dispatch result visualizations (dispatch bar chart + load distribution pie chart).
Call this tool after cvxpy solves successfully. Call it ONLY ONCE per solve.
Input JSON format:
{
    "cvxpy_result": <full return value from the cvxpy tool>,
    "generators":   <list of generator parameters extracted from load_case>,
    "total_load":   <total system load in MW>,
    "case_name":    <case name, e.g. "IEEE118">
}"""

        self.registerTool("visualization", vis_description, visualize.visualization)

    def registerTool(self, name: str, description: str, func: callable):
        """Register a tool with its name, description, and callable function."""
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool '{name}' registered.")

    def getTool(self, name: str) -> callable:
        """Get the callable function of a registered tool by name."""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """Return a formatted string listing all available tools and their descriptions."""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
