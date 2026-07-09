"""LLM API client with cost tracking and retry logic.

Supports: Anthropic, DeepSeek (OpenAI-compatible), and mock mode."""

import os
import time
import logging
import yaml
from dataclasses import dataclass
from typing import Optional

from src.core.exceptions import LLMError

logger = logging.getLogger(__name__)

# Auto-load .env file if present
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        # Look for .env in project root
        for _path in [".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]:
            _abs = os.path.abspath(_path)
            if os.path.exists(_abs):
                load_dotenv(_abs)
                break
    except ImportError:
        pass

_load_dotenv()


@dataclass
class LLMCallResult:
    """Raw result from a single LLM API call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    cost_usd: float


class LLMClient:
    """
    Wrapper around the Anthropic API with:
    - Model configuration from YAML
    - Automatic retry with backoff
    - Token usage and cost tracking
    - Optional mock mode for testing

    Usage:
        client = LLMClient.from_yaml("config/model.yaml")
        result = client.send(prompt="...")
        client.print_usage_summary()
    """

    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        input_price_per_mtok: float = 0.80,
        output_price_per_mtok: float = 4.00,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        mock_mode: bool = False,
    ):
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.mock_mode = mock_mode

        # Usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_calls = 0
        self.total_latency = 0.0

        # Lazily initialized client
        self._client = None

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: str = "config/model.yaml",
                  profile: Optional[str] = None,
                  mock_mode: bool = False) -> "LLMClient":
        """Create LLMClient from a YAML config file."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if profile is None:
            profile = data.get("active", "haiku")

        if profile not in data:
            raise ValueError(f"Unknown model profile: {profile}. Available: {list(data.keys())}")

        cfg = data[profile]
        api = data.get("api", {})

        return cls(
            model=cfg["model"],
            provider=cfg.get("provider", "anthropic"),
            max_tokens=cfg.get("max_tokens", 1024),
            temperature=cfg.get("temperature", 0.0),
            input_price_per_mtok=cfg.get("input_price_per_mtok", 0.80),
            output_price_per_mtok=cfg.get("output_price_per_mtok", 4.00),
            timeout_seconds=api.get("timeout_seconds", 60.0),
            max_retries=api.get("max_retries", 3),
            retry_delay_seconds=api.get("retry_delay_seconds", 2.0),
            mock_mode=mock_mode,
        )

    # ── Client Initialization ─────────────────────────────────────────────

    def _get_client(self):
        """Lazy-initialize the API client (Anthropic or DeepSeek/OpenAI)."""
        if self._client is not None:
            return self._client

        # OpenAI-compatible providers (deepseek, qwen, etc.)
        _openai_providers = {
            "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "qwen":     ("QWEN_API_KEY",     "QWEN_BASE_URL",     "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        }

        if self.provider in _openai_providers:
            key_env, url_env, default_url = _openai_providers[self.provider]
            try:
                import openai
            except ImportError:
                raise LLMError("openai package is required. Install with: pip install openai")

            api_key = os.environ.get(key_env)
            if not api_key and not self.mock_mode:
                raise LLMError(
                    f"{key_env} environment variable is not set.\n"
                    f"Set it in .env file: {key_env}=your-key-here"
                )
            base_url = os.environ.get(url_env, default_url)
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise LLMError(
                    "anthropic package is required. Install with: pip install anthropic"
                )

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key and not self.mock_mode:
                raise LLMError(
                    "ANTHROPIC_API_KEY environment variable is not set.\n"
                    "Set it with: export ANTHROPIC_API_KEY='your-key-here'\n"
                    "Or use mock_mode=True for testing without API calls."
                )

            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise LLMError(f"Unknown provider: {self.provider}")

        return self._client

    # ── Core API ──────────────────────────────────────────────────────────

    def send(self, prompt: str, system: Optional[str] = None) -> LLMCallResult:
        """
        Send a prompt to the LLM and return the result.

        Args:
            prompt: The user message content.
            system: Optional system prompt.

        Returns:
            LLMCallResult with content, tokens, latency, cost.

        Raises:
            LLMError: On API failure after all retries.
        """
        if self.mock_mode:
            return self._mock_response(prompt)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                t0 = time.time()

                if self.provider in ("deepseek", "qwen"):
                    # OpenAI-compatible chat API
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})

                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    content = response.choices[0].message.content
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens

                elif self.provider == "anthropic":
                    # Anthropic Messages API
                    messages = [{"role": "user", "content": prompt}]
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system or "",
                        messages=messages,
                    )
                    content = response.content[0].text
                    prompt_tokens = response.usage.input_tokens
                    completion_tokens = response.usage.output_tokens

                else:
                    raise LLMError(f"Unknown provider: {self.provider}")

                latency = time.time() - t0
                total_tokens = prompt_tokens + completion_tokens
                cost = self._calc_cost(prompt_tokens, completion_tokens)

                # Update tracking
                self.total_input_tokens += prompt_tokens
                self.total_output_tokens += completion_tokens
                self.total_cost_usd += cost
                self.total_calls += 1
                self.total_latency += latency

                logger.info(
                    f"API call: {total_tokens} tokens, "
                    f"{latency:.2f}s, ${cost:.4f}"
                )

                return LLMCallResult(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_seconds=latency,
                    model=self.model,
                    cost_usd=cost,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt+1}/{self.max_retries+1}): "
                        f"{e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"API call failed after {self.max_retries+1} attempts")

        raise LLMError(f"LLM API error: {last_error}")

    def send_with_system(
        self, system: str, user: str
    ) -> LLMCallResult:
        """Send with separate system and user prompts."""
        return self.send(prompt=user, system=system)

    # ── Cost Tracking ─────────────────────────────────────────────────────

    def _calc_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for a single call."""
        input_cost = (prompt_tokens / 1_000_000) * self.input_price_per_mtok
        output_cost = (completion_tokens / 1_000_000) * self.output_price_per_mtok
        return input_cost + output_cost

    @property
    def avg_latency(self) -> float:
        """Average latency across all calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency / self.total_calls

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def usage_summary(self) -> dict:
        """Return a usage summary dictionary."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_latency": round(self.avg_latency, 2),
            "model": self.model,
        }

    def print_usage_summary(self) -> None:
        """Print a human-readable usage summary."""
        s = self.usage_summary()
        print("─" * 50)
        print("LLM Usage Summary")
        print("─" * 50)
        print(f"  Model:        {s['model']}")
        print(f"  Total calls:  {s['total_calls']}")
        print(f"  Input tokens: {s['total_input_tokens']:,}")
        print(f"  Output tokens:{s['total_output_tokens']:,}")
        print(f"  Total tokens: {s['total_tokens']:,}")
        print(f"  Total cost:   ${s['total_cost_usd']:.4f}")
        print(f"  Avg latency:  {s['avg_latency']}s")
        print("─" * 50)

    # ── Mock Mode ─────────────────────────────────────────────────────────

    def _mock_response(self, prompt: str) -> LLMCallResult:
        """Return a deterministic mock response for testing."""
        import hashlib
        import random

        # Simple keyword-based mock that returns YES/NO based on prompt content
        # Count 'member_of' vs 'not_member_of' to simulate majority voting
        member_count = prompt.count("is a member of") + prompt.count("member of")
        not_member_count = prompt.count("is not a member of") + prompt.count("not a member")

        # Force deterministic but varied output using a hash of the prompt
        h = hashlib.md5(prompt.encode()).hexdigest()
        rng = random.Random(int(h[:8], 16))

        if member_count > not_member_count:
            answer = "YES"
        elif not_member_count > member_count:
            answer = "NO"
        else:
            answer = rng.choice(["YES", "NO"])

        # Detect conflict keywords
        has_negation = "not a member" in prompt.lower()
        has_transitive_chain = prompt.count("is a member of") >= 2
        conflict_detected = "YES" if (has_negation and has_transitive_chain) else "NO"

        mock_content = (
            f"Let me reason through this step by step.\n\n"
            f"Looking at the facts, I can see the relationships between entities.\n"
            f"Based on my analysis, the answer is clear.\n\n"
            f"ANSWER: {answer}\n"
            f"CONFLICT_DETECTED: {conflict_detected}\n"
        )

        tokens = len(mock_content.split())
        latency = rng.uniform(0.1, 0.3)

        return LLMCallResult(
            content=mock_content,
            prompt_tokens=len(prompt.split()),
            completion_tokens=tokens,
            total_tokens=len(prompt.split()) + tokens,
            latency_seconds=latency,
            model=f"mock:{self.model}",
            cost_usd=0.0,
        )
