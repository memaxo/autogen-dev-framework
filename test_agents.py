import asyncio
from src.agents.planner import PlanningAgent
from src.agents.coder import CodingAgent
from src.agents.debugger import DebuggingAgent

async def test_planning_agent():
    planner = PlanningAgent()
    plan = await planner.plan_next_steps(
        task="Create a REST API endpoint",
        current_state={
            'status': 'in_progress',
            'current_phase': 'planning',
            'context': {}
        }
    )
    print("Planning Result:", plan)

async def test_coding_agent():
    coder = CodingAgent()
    result = await coder.execute_task(
        task="Write a function to sort a list",
        context={},
        reply_to=None
    )
    print("Coding Result:", result)

if __name__ == "__main__":
    asyncio.run(test_planning_agent())
    asyncio.run(test_coding_agent()) 