# TODO: Implement llm_handler.py module
import requests
import logging

def ask_model(prompt: str, model_name: str, endpoint_url: str) -> str:
    """
    Sends a prompt to the specified model server and retrieves the generated response.
    """
    url = f"{endpoint_url}/v1/completions"
    payload = {
        "prompt": prompt,
        "model": model_name,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 25,
        "max_tokens": 500  # Limit the response length
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with {model_name}: {e}")
        return f"Error communicating with {model_name}: {e}"

def generate_final_answer(filtered_chunks, user_query, model_name, endpoint_url):
    """
    Generates the final answer using the filtered chunks and user query.
    """
    first_chunk = filtered_chunks[0]["text"]
    query_text = (
        "You are an assistant who answers questions strictly based on the provided Document Text. "
        "Your primary response should be based on the information given in the Document Text. "
        "Ensure that the final inference result is included in the <answer> tag. When making the inference, treat all information in the document as true.\n\n"
        f"Document Text:\n{first_chunk}\n\n"
        f"Question: {user_query}\n\n"
        "If the answer is not found in the Document Text, respond with 'No information available.' Only say 'No information available' "
        "if there is genuinely no relevant information in the Document Text.\n\n"
        "Please respond using the following XML format and provide the answer in key terms only (e.g., 'New Delhi' instead of 'The capital of India is New Delhi'):\n\n"
        "<response>\n"
        "    <answer>\n"
        "        [Your concise answer in key terms based strictly on the Document Text, or 'No information available']\n"
        "    </answer>\n\n"
        "    <comments>\n"
        "        [Any factual corrections, additional context, or your interpretation, if applicable]\n"
        "    </comments>\n"
        "</response>"
    )
    return ask_model(query_text, model_name, endpoint_url)