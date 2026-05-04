from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


model = OpenAIChatModel(
    model_name="llama3.2", provider=OpenAIProvider(base_url="http://localhost")
)
