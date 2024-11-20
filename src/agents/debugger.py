import asyncio
from typing import Dict, List, Optional, Any
from autogen import AssistantAgent, GroupChat, GroupChatManager
from autogen_ext.models import OpenAIChatCompletionClient
import logging
from src.config import Config
from src.monitor import measure_time

logger = logging.getLogger(__name__)

class DebuggingAgent(AssistantAgent):
    """An agent specialized in debugging code and analyzing errors."""
    
    def __init__(
        self,
        name: str = "debugging_agent",
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the debugging agent with AutoGen's native configuration.
        
        Args:
            name: Agent identifier
            llm_config: Language model configuration
            tools: List of available tools for the agent
            **kwargs: Additional configuration options
        """
        system_message = """
        You are an expert debugging agent that works with a planning agent to solve coding issues.
        
        When receiving tasks:
        1. Analyze the problem thoroughly
        2. Report findings back to the planner
        3. Execute debugging tasks as directed
        4. Provide detailed feedback on results
        
        Your core capabilities include:
        - Error message analysis
        - Stack trace interpretation
        - Code fix suggestions
        - Bug pattern recognition
        
        Use TERMINATE when debugging is complete.
        """
        
        # Default debugging tools
        default_tools = [{
            "name": "analyze_error",
            "description": "Analyzes error messages and stack traces",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_type": {"type": "string"},
                    "location": {"type": "string"},
                    "suggested_fixes": {"type": "array"}
                }
            }
        }]
        
        # Merge provided tools with defaults
        all_tools = (tools or []) + default_tools
        
        llm_config = llm_config or Config.get_agent_config("debugger")
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            tools=all_tools,
            **kwargs
        )

    async def analyze_error(
        self,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes error messages and stack traces to identify issues.
        
        Args:
            error_message: The error message to analyze
            stack_trace: Optional stack trace information
            context: Additional context about the error
            
        Returns:
            Dict containing analysis results and suggestions
        """
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Analyze the following error:
                
                Error Message: {error_message}
                Stack Trace: {stack_trace or 'Not provided'}
                Context: {context or {}}
                
                Provide structured analysis focusing on:
                1. Error type and location
                2. Potential causes
                3. Recommended fixes
                4. Prevention strategies
                """
            }]
            
            response = await self.generate_reply(messages)
            
            return {
                'success': True,
                'analysis': response.content,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error in error analysis: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    async def suggest_fixes(
        self,
        code: str,
        issues: List[str],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates potential fixes for identified issues.
        
        Args:
            code: The problematic code
            issues: List of identified issues
            requirements: Optional requirements and constraints
            
        Returns:
            Dict containing suggested fixes and explanations
        """
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Review and suggest fixes for the following code:
                
                Code:                ```
                {code}                ```
                
                Identified Issues:
                {', '.join(issues)}
                
                Requirements:
                {requirements or {}}
                
                Provide:
                1. Specific code modifications
                2. Explanation of changes
                3. Testing recommendations
                """
            }]
            
            response = await self.generate_reply(messages)
            
            return {
                'success': True,
                'fixes': response.content,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error in suggesting fixes: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }

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
