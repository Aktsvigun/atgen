from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    SummarizationMetric,
    PromptAlignmentMetric,
)
from openai import OpenAI, AsyncOpenAI


class EvaluationLLM(DeepEvalBaseLLM):
    """
    Custom Evaluation LLM implementation for DeepEval.

    This class implements the DeepEvalBaseLLM interface to allow using
    custom models with DeepEval metrics.
    """

    def __init__(
        self,
        api_key=None,
        model="openai/gpt-4o-2024-11-20",
        base_url="https://openrouter.ai/api/v1",
    ):
        """
        Initialize the Evaluation LLM.

        Args:
            api_key: Evaluation API key
            model: Model identifier (e.g., "openai/gpt-4o-2024-11-20")
            base_url: Evaluation API base URL
        """
        self.api_key = api_key

        self.model_name = model
        self.base_url = base_url
        self.client = None
        self.async_client = None
        self.OpenAI = OpenAI
        self.AsyncOpenAI = AsyncOpenAI

    def load_model(self):
        """Load and return the client."""
        if self.client is None:
            self.client = self.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self.client

    def load_async_model(self):
        """Load and return the async client."""
        if self.async_client is None:
            self.async_client = self.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self.async_client

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the evaluation model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response as a string
        """
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the evaluation model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response as a string
        """
        # Use the async client for async operations
        client = self.load_async_model()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_model_name(self):
        """Return the name of the model."""
        return f"EvaluationLLM: {self.model_name}"
