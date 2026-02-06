import logging
import pandas as pd
from ollama import Client

logger = logging.getLogger(__name__)


class TopicExplainer:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-oss:120b",
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.timeout = timeout

        self.client = Client(
            host="https://ollama.com",
            headers={
                "Authorization": "Bearer " + api_key
            },
            timeout=self.timeout,
        )

    def explain_topics(
        self,
        topic_terms: pd.DataFrame,
        search_query: str,
        source: str,
        created_at: str,
    ) -> pd.DataFrame:
        logger.info("Generating topic explanations...")

        rows = []

        for topic_id, group in topic_terms.groupby("topic_id"):
            terms = (
                group.sort_values("weight", ascending=False)["term"]
                .astype(str)
                .tolist()
            )

            prompt = self._build_prompt(
                topic_id=topic_id,
                terms=terms,
                search_query=search_query,
                source=source,
                created_at=created_at,
            )

            description = self._call_ollama(prompt)

            rows.append(
                {
                    "topic_id": topic_id,
                    "description": description,
                }
            )

        df = pd.DataFrame(rows)

        logger.info(f"Generated explanations for {len(df)} topics")
        return df

    def _build_prompt(
        self,
        topic_id: int,
        terms: list[str],
        search_query: str,
        source: str,
        created_at: str,
    ) -> str:
        terms_str = ", ".join(terms)
        date_info = f"Creation date: {created_at}"

        return f"""
        Context:
        The documents come from {source}

        Search Query:
        "{search_query}"
        {date_info}

        Topic ID: {topic_id}

        Top keywords for this topic:
        {terms_str}

        Task:
        Describe in 1–2 concise sentences what this topic represents.
        Focus on the research subfield or thematic area.
        Do not list the keywords explicitly.
        """.strip()

    def _call_ollama(self, prompt: str) -> str:
        logger.debug("Calling Ollama Cloud...")

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
        )

        return response["message"]["content"].strip()
