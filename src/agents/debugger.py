import asyncio
from typing import Dict, List, Optional
from autogen import AssistantAgent, GroupChat, GroupChatManager
from autogen_ext.models import OpenAIChatCompletionClient

class DebuggingAgent(AssistantAgent):
    """An agent specialized in debugging code and analyzing errors."""
    
    def __init__(self, name="debugging_agent", **kwargs):
        system_message = """You are an expert debugging agent that works with a planning agent to solve coding issues.
        When receiving tasks:
        1. Analyze the problem thoroughly
        2. Report findings back to the planner
        3. Execute debugging tasks as directed
        4. Provide detailed feedback on results
        
        Your core capabilities include:
        - Error message analysis
        - Stack trace interpretation
        - Code fix suggestions
        - Bug pattern recognition"""
        
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs
        )

    async def analyze_error(self, error_message: str, stack_trace: Optional[str] = None) -> Dict:
        """Analyzes error messages and stack traces to identify issues."""
        
        system_prompt = """You are an expert debugging agent. Analyze the provided error and stack trace 
        to identify the root cause and suggest fixes. Focus on:
        1. Error type and location
        2. Potential causes
        3. Recommended fixes
        4. Prevention strategies"""

        user_message = f"""Error Message: {error_message}
        Stack Trace: {stack_trace if stack_trace else 'Not provided'}
        
        Please provide a structured analysis."""

        response = await self.model_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        return response

    async def suggest_fixes(self, code: str, issues: List[str]) -> Dict:
        """Generates potential fixes for identified issues."""
        
        system_prompt = """You are an expert code fixer. Review the code and reported issues
        to suggest specific fixes. Provide:
        1. Code modifications
        2. Explanation of changes
        3. Testing recommendations"""

        user_message = f"""Code: {code}
        Reported Issues: {', '.join(issues)}
        
        Please suggest fixes."""

        response = await self.model_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        return response

# Example usage
async def main():
    debugging_agent = DebuggingAgent(
        name="debugging_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4",
            # api_key="YOUR_API_KEY"
        )
    )

    # Example error analysis
    error_result = await debugging_agent.analyze_error(
        error_message="IndexError: list index out of range",
        stack_trace="File 'main.py', line 25, in process_data\n    result = data[index]"
    )

    # Example fix suggestion
    fix_result = await debugging_agent.suggest_fixes(
        code="def process_data(data, index):\n    result = data[index]\n    return result",
        issues=["Index out of bounds error", "No input validation"]
    )

if __name__ == "__main__":
    asyncio.run(main())
