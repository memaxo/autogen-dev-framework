import asyncio
from src.chat import DevelopmentChat

async def test_development_chat():
    chat = DevelopmentChat()
    
    # Test a single task
    result = await chat.process_single_task(
        "Create a simple function to calculate fibonacci numbers"
    )
    
    print("Task Results:", result)

if __name__ == "__main__":
    asyncio.run(test_development_chat()) 