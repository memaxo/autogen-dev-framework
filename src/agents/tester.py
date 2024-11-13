from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models import OpenAIChatCompletionClient
from typing import Dict, Any

class TestingAgent(AssistantAgent):
    """
    Agent responsible for testing code produced by the CoderAgent.
    Interacts with ExecutorAgent to run tests and DebuggerAgent to handle failures.
    """

    def __init__(self):
        super().__init__(
            name="tester",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4-0125-preview",
            ),
            system_message=self._get_system_message()
        )

    def _get_system_message(self) -> str:
        """Define the tester's core capabilities and responsibilities"""
        return """
        You are the testing agent responsible for ensuring code quality.
        Your responsibilities include:

        1. Test Execution:
           - Run unit and integration tests on the provided code
           - Validate test results and report issues

        2. Debugging Coordination:
           - Collaborate with the DebuggerAgent to resolve test failures
           - Provide detailed error reports and logs

        3. Communication:
           - Report test results to the PlannerAgent and CoderAgent
           - Suggest improvements or additional tests if necessary
        """

    async def execute_tests(self, code: str) -> Dict[str, Any]:
        """
        Execute tests on the given code and return results.
        
        Args:
            code: The code to be tested
            
        Returns:
            A dictionary with test results and status
        """
        # Simulate test execution
        test_results = await self._run_tests(code)
        if not test_results['success']:
            await self._coordinate_with_debugger(test_results['errors'])
        return test_results

    async def _run_tests(self, code: str) -> Dict[str, Any]:
        """Simulate running tests on the code"""
        # Placeholder for actual test execution logic
        return {
            'success': True,
            'errors': None,
            'message': "All tests passed successfully."
        }

    async def _coordinate_with_debugger(self, errors: Any):
        """Coordinate with the DebuggerAgent to resolve test failures"""
        # Placeholder for interaction with DebuggerAgent
        pass
