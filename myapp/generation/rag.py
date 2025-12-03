import json
import os
from typing import List, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


FALLBACK_PHRASE = "There are no good products that fit the request based on the retrieved results."


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

## Task
Use only the structured product facts below. Pick the best match for the user, quoting PID and exact product title. Justify with concrete attributes (price, discount, rating, notable features). All monetary values are in Indian Rupees (₹); never use the dollar sign. If nothing fits, respond with the fallback sentence exactly.

## Retrieved Products (JSON)
{retrieved_results}

## User Request
{user_query}

## Output JSON (return ONLY JSON, no extra text)
{{
    "best_product": {{"title": "Product Title"}},
    "why": "Short justification referencing provided attributes",
    "alternative": {{"title": "Product Title", "why": "Optional justification"}} | null,
    "fallback": "{fallback}"
}}
Where "fallback" must be "{fallback}" if no suitable item exists, otherwise an empty string.
"""

    DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
    NO_MATCH = FALLBACK_PHRASE

    def __init__(self, max_docs: int = 6, max_description_chars: int = 220):
        self.max_docs = max_docs
        self.max_description_chars = max_description_chars
        self.model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def _format_document(self, doc) -> dict:
        description = (doc.description or "").strip()
        if description and len(description) > self.max_description_chars:
            description = description[: self.max_description_chars].rstrip() + "…"

        return {
            "pid": getattr(doc, "pid", None),
            "title": getattr(doc, "title", ""),
            "brand": getattr(doc, "brand", None),
            "category": getattr(doc, "category", None),
            "price_inr": getattr(doc, "selling_price", None),
            "discount": getattr(doc, "discount", None),
            "average_rating": getattr(doc, "average_rating", None),
            "description": description,
            "url": getattr(doc, "url", None),
        }

    def _parse_structured_response(self, content: str) -> Optional[dict]:
        if not content:
            return None
        snippet = content.strip()
        if snippet.startswith("```"):
            snippet = snippet.strip("`")
            if "\n" in snippet:
                snippet = snippet.split("\n", 1)[1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    def _format_structured_answer(self, data: dict) -> str:
        fallback = data.get("fallback") or ""
        if fallback.strip():
            return self.NO_MATCH

        best = data.get("best_product") or {}
        why = data.get("why") or ""
        alt = data.get("alternative")

        lines = []
        if best:
            title = best.get("title", "").strip() or "Unknown product"
            lines.append(f"Best Product: <strong>{title}</strong>")
        if why:
            lines.append(f"Why: {why.strip()}")
        if alt:
            alt_title = alt.get("title", "").strip()
            alt_why = alt.get("why", "")
            alt_title_fmt = alt_title or "Alternative product"
            lines.append(f"Alternative: <strong>{alt_title_fmt}</strong> — {alt_why.strip()}")
        return "\n".join(lines) if lines else self.NO_MATCH

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> str:
        if not retrieved_results:
            return self.NO_MATCH

        if not self.client:
            return self.DEFAULT_ANSWER

        try:
            context_docs = []
            for doc in retrieved_results[: min(top_N, self.max_docs)]:
                context_docs.append(self._format_document(doc))
            if not context_docs:
                return self.NO_MATCH

            formatted_results = json.dumps(context_docs, indent=2)
            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query.strip(),
                fallback=FALLBACK_PHRASE,
            )

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )

            generation = chat_completion.choices[0].message.content
            structured = self._parse_structured_response(generation)
            if structured:
                return self._format_structured_answer(structured)
            return generation or self.NO_MATCH
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return self.DEFAULT_ANSWER
