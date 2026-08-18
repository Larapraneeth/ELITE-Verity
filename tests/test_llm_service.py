from unittest.mock import Mock, patch

from app.config import GROQ_MODEL
from app.services import llm_service


def test_generate_answer_uses_configured_groq_model():
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="configured model reply"))]
    )

    with patch.object(llm_service, "client", client):
        result = llm_service.generate_answer("Explain the configuration.")

    assert result == "configured model reply"
    client.chat.completions.create.assert_called_once()
    # The model sent to Groq must be the configured GROQ_MODEL.
    assert client.chat.completions.create.call_args.kwargs["model"] == GROQ_MODEL