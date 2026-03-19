from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback
import base64
import os
import re

from EDAgentLLM import EDAgentLLM
from ToolExecutor import ToolExecutor
from EDAgent import ReActAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = EDAgentLLM()
tool_executor = ToolExecutor()

SKIP_KEYWORDS = ["generate a title", "generate tags", "follow_ups", "follow-up"]


def image_to_base64_markdown(image_path: str) -> str:
    """Convert a local image file to an inline Markdown base64 string."""
    if not os.path.exists(image_path):
        return f"(Image not found: {image_path})"

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    filename = os.path.basename(image_path)
    return f"![{filename}](data:{mime};base64,{data})"


def embed_images_in_result(result_str: str) -> str:
    """Scan result string for image paths, embed them as base64 Markdown, then clean up files."""
    image_paths = re.findall(r'[\w\-/]+\.(?:png|jpg|jpeg)', result_str)

    appended = []
    for path in image_paths:
        if os.path.exists(path):
            appended.append(image_to_base64_markdown(path))

    if appended:
        result_str += "\n\n" + "\n\n".join(appended)

    chart_dir = "output_charts"
    if os.path.exists(chart_dir):
        for fname in os.listdir(chart_dir):
            if fname.endswith(".png"):
                try:
                    os.remove(os.path.join(chart_dir, fname))
                except Exception:
                    pass

    return result_str


def make_response(content: str) -> dict:
    """Build a standard OpenAI-compatible chat completion response."""
    return {
        "id": "chatcmpl-eda",
        "object": "chat.completion",
        "model": "ed-agent-react",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ed-agent-react", "object": "model", "owned_by": "lukun_he"}]
    }


@app.api_route("/v1/openapi.json", methods=["GET", "OPTIONS"])
async def openapi_proxy():
    return {}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            return {"error": {"message": "No messages provided", "type": "invalid_request_error"}}

        last_content = messages[-1].get("content", "")
        system_content = messages[0].get("content", "") if messages else ""
        combined = (system_content + last_content).lower()
        if any(kw in combined for kw in SKIP_KEYWORDS):
            return make_response("")

        conversation_history = ""
        for msg in messages[:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_history += f"{role}: {content}\n"

        user_question = last_content
        if conversation_history:
            full_input = f"[Conversation History]\n{conversation_history}\n[Current Question]\n{user_question}"
        else:
            full_input = user_question

        # Run the Agent
        agent = ReActAgent(llm_client, tool_executor, max_steps=10)
        result = agent.run(full_input)

        # Embed any generated charts as base64 images
        result_str = embed_images_in_result(str(result))

        return make_response(result_str)

    except Exception as e:
        traceback.print_exc()
        return make_response(f"Agent error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
