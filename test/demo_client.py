#!/usr/bin/env python3
"""Client script to run the multi-agent demo via FastAPI."""

import httpx
import asyncio
import json
from typing import Dict, Any


async def run_demo(base_url: str = "http://localhost:8000") -> None:
    """Run the demo by calling the FastAPI endpoint."""
    
    print("🚀 Starting multi-agent demo via FastAPI...")
    print(f"API Server: {base_url}")
    print("-" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Call the demo endpoint
            print("📡 Calling /agent/demo endpoint...")
            response = await client.post(f"{base_url}/agent/demo")
            
            if response.status_code == 200:
                result = response.json()
                display_results(result)
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except httpx.RequestError as e:
            print(f"❌ Request failed: {e}")
            print("Make sure the FastAPI server is running at http://localhost:8000")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def display_results(result: Dict[str, Any]) -> None:
    """Display results in the same format as main.py."""
    
    print(f"\n{'='*60}")
    print(f"TOPIC: {result['topic']}")
    print(f"{'='*60}")
    print(f"\nSUMMARY:")
    print(result['summary'])
    print(f"\n{'='*60}")
    print(f"FULL ARTICLE:")
    print(result['article'])
    print(f"\n{'='*60}")
    print(f"SOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. {source}")
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING INFO:")
    print(f"Processing Time: {result['processing_time']:.2f} seconds")
    print(f"Agent ID: {result['agent_id']}")
    print(f"Status: {result['status']}")
    print("✅ Demo completed successfully!")


async def check_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if the FastAPI server is running and healthy."""
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy: {health_data['status']}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return False


async def main() -> None:
    """Main function."""
    base_url = "http://localhost:8000"
    
    print("🔍 Checking server health...")
    if await check_server_health(base_url):
        print()
        await run_demo(base_url)
    else:
        print("\n💡 To start the server, run:")
        print("   python start_api.py")
        print("   # or")
        print("   uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    asyncio.run(main()) 