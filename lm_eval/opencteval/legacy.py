"""Unfinished legacy prototype retained solely for import compatibility."""

from collections import Counter

from lm_eval.api.task import Task


class CrossConsistencyCoT(Task):
    """Retain the historical prototype API; use reasoning_modes for evaluation."""

    OUTPUT_TYPE = "generate_until"  # assuming generation-based prompts

    def process_results(self, doc, results):
        for model_id in self.config.model_list:
            print(model_id)

        return 0

    def extract_chain(self, text: str) -> str:
        # Customize (e.g., split from "Answer:" or markers)
        parts = text.split("Answer:")
        return parts[0].strip() if len(parts) > 1 else text.strip()

    def extract_final_answer(self, chain: str) -> str:
        # Simplest heuristic: last non-empty line
        lines = [l for l in chain.strip().splitlines() if l]
        return lines[-1] if lines else ""

    def integrate_chains(self, chains: list[str]) -> str:
        # Example: voting-based consensus + detailed log
        answers = [self.extract_final_answer(c) for c in chains]
        most_common = Counter(answers).most_common(1)[0][0]
        header = f"Consensus answer: {most_common}"
        body = "\n\n".join(f"Chain {i + 1}:\n{c}" for i, c in enumerate(chains))
        return f"{header}\n\n{body}"
