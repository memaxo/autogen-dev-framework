from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models import OpenAIChatCompletionClient

class CoderAgent:
    """
    CoderAgent is responsible for implementing code based on specifications
    provided by the PlannerAgent. It follows best practices, adheres to coding
    standards, and returns the generated code back to the planner.
    """

    def __init__(self, name, api_key, tools=None):
        self.agent = AssistantAgent(
            name=name,
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-2024-08-06",
                api_key=api_key,
            ),
            tools=tools or [],
        )

    async def implement_code(self, specifications):
        """
        Generate code based on provided specifications.
        
        Args:
            specifications (str): Text description of code specifications.
        
        Returns:
            str: The generated code to be returned to the PlannerAgent.
        """
        task_description = f"Implement code based on the following specifications: {specifications}"
        response = await self.agent.complete_task(task=task_description)
        return response  # Return the code back to the PlannerAgent
