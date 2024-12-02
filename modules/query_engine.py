# TODO: Implement query_engine.py module
def extract_final_answers(results):
    """
    Extracts and formats the final answers from query results.
    """
    final_answers = []
    if results and "documents" in results:
        for idx, document in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][idx] if "metadatas" in results else {}
            distance = results["distances"][0][idx] if "distances" in results else None

            answer = {
                "text": document.strip(),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "distance": distance,
            }
            final_answers.append(answer)
    return final_answers