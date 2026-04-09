from src.services.search import extract_search_query, is_web_search_request


def test_extract_search_query_from_string_content():
    messages = [
        {
            "role": "user",
            "content": "Perform a web search for the query: best M.2 NVMe SSD 2026",
        }
    ]

    assert extract_search_query(messages) == "best M.2 NVMe SSD 2026"


def test_extract_search_query_from_block_content():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Perform a web search for the query: best M.2 NVMe SSD 2026",
                }
            ],
        }
    ]

    assert extract_search_query(messages) == "best M.2 NVMe SSD 2026"


def test_is_web_search_request_accepts_builtin_and_function_tool_shapes():
    assert is_web_search_request(
        [{"type": "web_search_20250305"}],
        "You are an assistant for performing a web search tool use",
    )
    assert is_web_search_request(
        [{"name": "web_search"}],
        "You are an assistant for performing a web search tool use",
    )
