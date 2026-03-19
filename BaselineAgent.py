from EDAgentLLM import EDAgentLLM


class BaselineAgent:
    def __init__(self, llm_client: EDAgentLLM, temperature: float = 0):
        self.llm_client = llm_client
        self.temperature = temperature

    def run(self, prompt: str):
        print("🚀 Running Baseline (single-shot)...")

        messages = [
            {"role": "user", "content": prompt}
        ]

        response_text = self.llm_client.think(
            messages=messages,
            temperature=self.temperature
        )

        return response_text
    
    def build_non_evo_prompt(self, fewshot_data, target_pd):
        examples_text = ""

        for sample in fewshot_data:
            examples_text += f"PD = {sample['pd']} MW, Cost = {sample['cost']}\n"
            examples_text += f"PG = {sample['pg']}\n\n"

        prompt = f"""
    You are given optimal generation dispatch solutions.

    Goal:
    Generate a new dispatch vector such that:
    - Sum(PG) equals PD exactly.
    - Follow trend in examples.
    - Maintain proportional scaling.
    - Respect implicit limits.

    Examples:

    {examples_text}

    Now solve:

    PD = {target_pd} MW

    Output only the PG vector.
    """

        return prompt
