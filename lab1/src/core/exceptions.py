"""Custom exceptions for the experiment framework."""


class ExperimentError(Exception):
    """Base exception for experiment framework."""
    pass


class GenerationError(ExperimentError):
    """Error during sample generation."""
    pass


class AgentError(ExperimentError):
    """Error during agent execution."""
    pass


class LLMError(AgentError):
    """Error from the LLM API."""
    pass


class AnswerExtractionError(AgentError):
    """Failed to extract a boolean answer from agent output."""
    pass


class ConfigError(ExperimentError):
    """Error in experiment configuration."""
    pass
