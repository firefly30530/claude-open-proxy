from openai import OpenAI
import json

client = OpenAI(
    api_key="cc87a039b8ce11616ae8943e3c22fe3d",
    base_url="https://api.bytez.com/v1",
)


def calculator(a, b, op):
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b
    else:
        raise ValueError("unknown op")


for model_id in [
    "anthropic/claude-opus-4-5",
]:

    # 1. First request → model should CALL the tool
    first = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "Use the calculator tool to add 42 and 19."}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Add/sub/mul/div two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                            "op": {
                                "type": "string",
                                "enum": ["add", "sub", "mul", "div"],
                            },
                        },
                        "required": ["a", "b", "op"],
                    },
                },
            }
        ],
        extra_body={
            "reasoning_effort": "high",  # OpenAI-style (if supported)
            # or
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2048,
            },  # Anthropic-style (if supported)
        },
    )

    msg = first.choices[0].message
    tool_call = msg.tool_calls[0]
    print(first)

    print("🔧 Tool call:", tool_call)

    # 2. Execute the tool in Python
    args = json.loads(tool_call.function.arguments)
    result = calculator(**args)

    print("🧮 Tool result:", result)

    # 3. Send tool result back to the model
    final = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "Use the calculator tool to add 42 and 19."},
            {
                "role": "assistant",
                "content": "",  # ← REQUIRED for Bytez
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"result": result}),
            },
        ],
    )

    print(f"\n🤖 Final answer for {model_id}:")
    print(final.choices[0].message.content)