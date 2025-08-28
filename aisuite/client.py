from .provider import ProviderFactory
import os
import asyncio
import inspect
from .utils.tools import Tools
from .framework.message_normalizer import MessageNormalizer


class Client:
    def __init__(self, provider_configs: dict = {}):
        """
        Initialize the client with provider configurations.
        Use the ProviderFactory to create provider instances.

        Args:
            provider_configs (dict): A dictionary containing provider configurations.
                Each key should be a provider string (e.g., "google" or "aws-bedrock"),
                and the value should be a dictionary of configuration options for that provider.
                For example:
                {
                    "openai": {"api_key": "your_openai_api_key"},
                    "aws-bedrock": {
                        "aws_access_key": "your_aws_access_key",
                        "aws_secret_key": "your_aws_secret_key",
                        "aws_region": "us-west-2"
                    }
                }
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self._chat = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

    def _validate_provider_key(self, provider_key):
        """
        Validate if the provider key corresponds to a supported provider.
        """
        supported_providers = ProviderFactory.get_supported_providers()

        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        return provider_key

    def configure(self, provider_configs: dict = None):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers()  # NOTE: This will override existing provider instances.

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


class Chat:
    def __init__(self, client: "Client"):
        self.client = client
        self._completions = Completions(self.client)

    @property
    def completions(self):
        """Return the completions interface."""
        return self._completions


class Completions:
    def __init__(self, client: "Client"):
        self.client = client
        self._loop = None
    
    def _get_event_loop(self):
        """Get or create an event loop for running async code"""
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            return None
        except RuntimeError:
            # No running loop, we can create one
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously"""
        loop = self._get_event_loop()
        if loop is None:
            # We're already in an async context, can't run synchronously
            # This should be handled by making the entire create method async
            raise RuntimeError("Cannot run async provider in existing event loop. Use async client instead.")
        return loop.run_until_complete(coro)
    
    def _wrap_async_generator(self, async_gen):
        """Wrap an async generator to make it synchronously iterable"""
        import asyncio
        
        # Create a new event loop for this generator if needed
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context already, can't wrap
            raise RuntimeError("Cannot wrap async generator in existing event loop")
        except RuntimeError:
            # No running loop, we can create one
            loop = asyncio.new_event_loop()
            
        class AsyncGenWrapper:
            def __init__(self, agen, loop):
                self.agen = agen
                self.loop = loop
                
            def __iter__(self):
                return self
                
            def __next__(self):
                try:
                    # Run the async generator's __anext__ method synchronously
                    coro = self.agen.__anext__()
                    chunk = self.loop.run_until_complete(coro)
                    return chunk
                except StopAsyncIteration:
                    raise StopIteration
                    
        return AsyncGenWrapper(async_gen, loop)
    
    def _wrap_streaming_coroutine(self, coro):
        """Wrap a coroutine that returns an async generator for streaming"""
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError("Cannot wrap streaming coroutine in existing event loop")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            
        class StreamingCoroutineWrapper:
            def __init__(self, coro, loop):
                self.coro = coro
                self.loop = loop
                self.async_gen = None
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.async_gen is None:
                    # First call - await the coroutine to get the async generator
                    async def get_generator():
                        return await self.coro
                    self.async_gen = self.loop.run_until_complete(get_generator())
                    
                try:
                    # Get next item from the async generator
                    coro = self.async_gen.__anext__()
                    chunk = self.loop.run_until_complete(coro)
                    return chunk
                except StopAsyncIteration:
                    raise StopIteration
                    
        return StreamingCoroutineWrapper(coro, loop)

    def _extract_thinking_content(self, response):
        """
        Extract content between <think> tags if present and store it in reasoning_content.

        Args:
            response: The response object from the provider

        Returns:
            Modified response object
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                content = message.content.strip()
                if content.startswith("<think>") and "</think>" in content:
                    # Extract content between think tags
                    start_idx = len("<think>")
                    end_idx = content.find("</think>")
                    thinking_content = content[start_idx:end_idx].strip()

                    # Store the thinking content
                    message.reasoning_content = thinking_content

                    # Remove the think tags from the original content
                    message.content = content[end_idx + len("</think>") :].strip()

        return response

    def _tool_runner(
        self,
        provider,
        model_name: str,
        messages: list,
        tools: any,
        max_turns: int,
        **kwargs,
    ):
        """
        Handle tool execution loop for max_turns iterations.

        Args:
            provider: The provider instance to use for completions
            model_name: Name of the model to use
            messages: List of conversation messages
            tools: Tools instance or list of callable tools
            max_turns: Maximum number of tool execution turns
            **kwargs: Additional arguments to pass to the provider

        Returns:
            The final response from the model with intermediate responses and messages
        """
        # Handle tools validation and conversion
        if isinstance(tools, Tools):
            tools_instance = tools
            kwargs["tools"] = tools_instance.tools()
        else:
            # Check if passed tools are callable
            if not all(callable(tool) for tool in tools):
                raise ValueError("One or more tools is not callable")
            tools_instance = Tools(tools)
            kwargs["tools"] = tools_instance.tools()

        turns = 0
        intermediate_responses = []  # Store intermediate responses
        intermediate_messages = []  # Store all messages including tool interactions

        while turns < max_turns:
            # Make the API call
            response = provider.chat_completions_create(model_name, messages, **kwargs)
            
            # Handle async providers
            if inspect.iscoroutine(response):
                response = self._run_async(response)
            
            response = self._extract_thinking_content(response)

            # Store intermediate response
            intermediate_responses.append(response)

            # Check if there are tool calls in the response
            tool_calls = (
                getattr(response.choices[0].message, "tool_calls", None)
                if hasattr(response, "choices")
                else None
            )

            # Store the model's message
            intermediate_messages.append(response.choices[0].message)

            if not tool_calls:
                # Set the intermediate data in the final response
                response.intermediate_responses = intermediate_responses[
                    :-1
                ]  # Exclude final response
                response.choices[0].intermediate_messages = intermediate_messages
                return response

            # Execute tools and get results
            results, tool_messages = tools_instance.execute_tool(tool_calls)

            # Add tool messages to intermediate messages
            intermediate_messages.extend(tool_messages)

            # Add the assistant's response and tool results to messages
            messages.extend([response.choices[0].message, *tool_messages])

            turns += 1

        # Set the intermediate data in the final response
        response.intermediate_responses = intermediate_responses[
            :-1
        ]  # Exclude final response
        response.choices[0].intermediate_messages = intermediate_messages
        return response

    def create(self, model: str, messages: list, stream: bool = False, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.
        Supports automatic tool execution when max_turns is specified.
        """
        # Check that correct format is used
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'"
            )

        # Extract the provider key from the model identifier, e.g., "google:gemini-xx"
        provider_key, model_name = model.split(":", 1)

        # Validate if the provider is supported
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        # Initialize provider if not already initialized
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Normalize messages for the target provider/model (use full model string for better detection)
        normalized_messages = MessageNormalizer.normalize_messages(messages, model)

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.get("tools", None)

        # Check environment variable before allowing multi-turn tool execution
        if max_turns is not None and tools is not None:
            return self._tool_runner(
                provider,
                model_name,
                normalized_messages.copy(),
                tools,
                max_turns,
            )

        # Default behavior without tool execution
        # Delegate the chat completion to the correct provider's implementation
        result = provider.chat_completions_create(model_name, normalized_messages, stream=stream, **kwargs)
        
        
        # Check if result is a coroutine (async function)
        if inspect.iscoroutine(result):
            if stream:
                # For streaming, we need special handling
                # The coroutine, when awaited, returns an async generator
                return self._wrap_streaming_coroutine(result)
            else:
                # Non-streaming, run the coroutine normally
                result = self._run_async(result)
        # Check if result is already an async generator
        elif inspect.isasyncgen(result):
            # Create a wrapper that converts async generator to sync
            return self._wrap_async_generator(result)
        
        return result
        #return self._extract_thinking_content(response)

