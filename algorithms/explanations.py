import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class TopicExplainer:
    def __init__(
        self,
        model_name: str = "llama3.1",
        ollama_url: str = "http://localhost:1234",
        timeout: str = 60
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout

    def explain_topics(
        self,
        topic_terms: pd.DataFrame,
        search_query: str,
        source: str,
        created_at: str
    ):
        logger.info("Generating topic explainations...")

        rows = []

        for topic_id, group in topic_terms.groupby("topic_id"):
            terms = (
                group
                .sort_values("weight", ascending=False)
                .tolist()
            )

            prompt = self._build_promt(
                topic_id=topic_id,
                terms=terms,
                search_query=search_query,
                source=source,
                created_at=created_at
            )

            description = self._call_ollama(prompt)

            rows.append({
                "topic_id": topic_id,
                "description": description
            })

        df = pd.DataFrame(rows)
        logger.info(
            f"Generated explainations for {len(df)} topics"
        )

    def _build_prompt(
        self,
        topic_id: int,
        terms: list[str],
        search_query: str,
        source: str,
        created_at: str
    ):
        terms_str = ", ".join(terms)

        date_info = f"Creation date: {created_at}"

        return f"""
        Context:
        The documents come from {source}

        Search Query:
        \"{search_query}\"{date_info}

        Topic ID: {topic_id}

        Top keywords for this topic:
        {terms_str}

        Task:
        Describe in 1-2 concise sentences what this topic represents.
        Focus on the research subfield or thematic area.
        Do not list the keywords explicitly.
        """

    def _call_ollama(self, prompt: str) -> str:
        logger.debug("Calling Ollama...")

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()["response"].strip()
