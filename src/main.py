# src/main.py
import asyncio
import dspy
from agents.coordinator import MultiAgentCoordinator
from utils.config import Config
from utils.error_handling import setup_logging


async def main() -> None:
    """Main function demonstrating the multi-agent system."""
    # Setup
    config = Config()
    logger = setup_logging(config.log_level)

    # Configure DSPy
    lm = dspy.LM(
        model=f"openai/{config.model_name}",
        api_key=config.openai_api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    dspy.configure(lm=lm, async_max_workers=config.async_workers)

    # Initialize multi-agent system
    coordinator = MultiAgentCoordinator(config.search_api_key)

    # Example request
    request = "Write a comprehensive article about the impact of artificial intelligence on modern healthcare"

    logger.info(f"Processing request: {request}")

    try:
        # Process the request
        result = await coordinator.forward(
            request=request, target_audience="healthcare professionals", max_sources=10
        )

        # Display results
        print(f"\n{'='*60}")
        print(f"TOPIC: {result['topic']}")
        print(f"{'='*60}")
        print(f"\nSUMMARY:")
        print(result["summary"])
        print(f"\n{'='*60}")
        print(f"FULL ARTICLE:")
        print(result["final_article"])
        print(f"\n{'='*60}")
        print(f"SOURCES:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source}")

        logger.info("Request processed successfully")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise
    finally:
        # Clean up resources
        await coordinator.close()


if __name__ == "__main__":
    asyncio.run(main())
