import json
import re

import ToolExecutor


from EDAgentLLM import EDAgentLLM

REACT_PROMPT_TEMPLATE = """
Role: You are a Power Systems Engineering Expert and Mathematical Optimization Assistant.
You specialize in converting unstructured requirements into structured Economic Dispatch (ED) model parameters and solving them using external tools.

Task:

0. Memory Retrieval: At the start of EVERY conversation, call memory[{{"action": "search", "query": "<user question>"}}]
   to check if there are relevant parameters or previous dispatch results stored.
1. Data Loading: Use tools to load power system case data. Supported cases: IEEE14, IEEE30, IEEE57, IEEE118, IEEE200, IEEE300.
2. Data Extraction: Identify generator limits (P_min, P_max), cost coefficients (a, b, c), and total load (P_d).
3. Intent Parsing & Update: Identify user modifications (e.g., "shut down unit 3", "increase load by 10%") and update the parameters.
   - After finalizing parameters, call memory[{{"action": "add", "memory_content": "<case + final params summary>", "memory_type": "working", "importance": 0.8}}] to save them.
4. Mathematical Solving: Call the 'cvxpy' tool by passing the finalized JSON parameters to get the optimal power dispatch.
5. After solving, call memory[{{"action": "add", "memory_content": "<result summary>", "memory_type": "working", "importance": 0.9}}] to save the result.
6. If you have arrived at the final answer, summarize it with Finish[] and produce visualization using corespond tool.

Constraints:

- Power Balance: Total generation must equal P_d.
- Status Management: If a unit is offline, set its status to 0.
- Physical Limits: Ensure P_min <= Pg <= P_max for all active units.

Available Tools:
{tools}

Format:
Thought: Your reasoning about the current state and what parameters need to be updated.
Action: The action you decide to take must follow one of the following formats:
- `{{tool_name}}[{{tool_input}}]`: Invoke an available tool.
- `Finish[Final Answer]`: When you believe you have obtained the final answer.

Question:
{question}

History:
{history}

JSON Schema for 'cvxpy' tool:
{{
  "target_load_mw": float,
  "generators": [
    {{
      "id": int,
      "status": 1|0,
      "p_min": float,
      "p_max": float,
      "cost": {{"a": float, "b": float, "c": float}}
    }}
  ]
}}
"""


class ReActAgent:
    def __init__(self, llm_client: EDAgentLLM,  tool_executor: ToolExecutor, max_steps: int = 30):
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.history = []
        self.tool_executor = tool_executor

    def _parse_output(self, text: str):
        """extract Thought and Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):

        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("XXX : no response_text")
                break

            thought, action = self._parse_output(response_text)

            if thought:
                print(f"thought: {thought}")

            if not action:
                print("no valid action")
                break

            # Finish 检查
            if action.startswith("Finish"):
                match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
                    print(f"🎉 final answer: {final_answer}")
                    return final_answer
                return action

            # ---- 解析所有 Action 并逐个执行 ----
            all_actions = self._parse_all_actions(response_text)  # 返回 [(tool_name, tool_input), ...]

            if not all_actions:
                continue

            for tool_name, tool_input in all_actions:
                print(f"🎬 action: {tool_name}[{tool_input}]")

                tool_function = self.tool_executor.getTool(tool_name)

                if not tool_function:
                    observation = f"❌ there is no tool named '{tool_name}'"
                elif tool_name == "memory":
                    try:
                        import json
                        params = json.loads(tool_input)
                        mem_action = params.pop("action")  # 用新变量，不覆盖 action
                        observation = tool_function(mem_action, **params)
                    except Exception as e:
                        observation = f"❌ memory 工具调用失败: {e}"
                else:
                    try:
                        observation = tool_function(tool_input)
                    except Exception as e:
                        observation = f"❌ {tool_name} 执行失败: {e}"

                print(f"👀 observation: {observation}")

                # history 记录完整的原始 action
                self.history.append(f"Action: {tool_name}[{tool_input}]")
                self.history.append(f"Observation: {observation}")

        print("max steps")
        return None

    def _parse_all_actions(self, llm_output: str):
        results = []
        decoder = json.JSONDecoder()

        for m in re.finditer(r'Action:\s*(\w+)\[', llm_output):
            tool_name = m.group(1).strip()
            if tool_name == "Finish":
                continue

            start = m.end()
            # 找到对应的 ] 结束位置
            raw = llm_output[start:].strip()

            # ✅ 先尝试 JSON 解析
            try:
                obj, _ = decoder.raw_decode(raw)
                results.append((tool_name, json.dumps(obj)))
                continue
            except Exception:
                pass

            # ✅ 不是 JSON，直接取 [ ] 之间的纯字符串
            end_match = re.search(r'^(.*?)\]', raw, re.DOTALL)
            if end_match:
                plain_input = end_match.group(1).strip()
                results.append((tool_name, plain_input))

        return results
