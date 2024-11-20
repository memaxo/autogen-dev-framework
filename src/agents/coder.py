from autogen import AssistantAgent
from typing import Optional, Dict, Any, List
import logging
from src.config import Config
from src.monitor import measure_time

logger = logging.getLogger(__name__)

class CoderAgent(AssistantAgent):
    """
    CoderAgent is responsible for implementing code based on specifications.
    Inherits from AutoGen's AssistantAgent for native integration.
    """

    def __init__(
        self,
        name: str = "coder",
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the coder agent with AutoGen's native configuration.

        Args:
            name: Agent identifier
            llm_config: Language model configuration
            tools: List of available tools for the agent
        """
        llm_config = llm_config or Config.get_agent_config("coder")
        
        system_message = """
        You are an expert coding assistant focused on writing clean, efficient code.
        
        Responsibilities:
        1. Write code based on requirements
        2. Follow best practices and patterns
        3. Implement error handling
        4. Document code appropriately
        
        Guidelines:
        - Write modular, maintainable code
        - Include proper type hints
        - Add docstrings and comments
        - Consider edge cases
        - Follow project coding standards
        
        Use TERMINATE when implementation is complete.
        """
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            tools=tools or []
        )

    async def execute_coding_task(
        self,
        specifications: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code based on provided specifications.
        
        Args:
            specifications: Detailed code requirements
            context: Additional context and requirements
        
        Returns:
            Dict containing generated code and metadata
        """
        try:
            # Use AutoGen's native message handling
            response = await self.generate_reply(
                messages=[{
                    "role": "user",
                    "content": f"Implement code based on: {specifications}\n\nContext: {context}"
                }]
            )
            
            return {
                'success': True,
                'code': response.content,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    async def review_code(
        self,
        code: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review generated code against requirements.
        
        Args:
            code: The code to review
            requirements: Original requirements and constraints
            
        Returns:
            Dict containing review results
        """
        try:
            review_prompt = f"""
            Review the following code against requirements:
            
            Code:            ```
            {code}            ```
            
            Requirements:
            {requirements}
            
            Provide detailed review focusing on:
            1. Correctness
            2. Best practices
            3. Performance
            4. Security
            """
            
            response = await self.generate_reply(
                messages=[{
                    "role": "user",
                    "content": review_prompt
                }]
            )
            
            return {
                'success': True,
                'review': response.content,
                'approved': 'TERMINATE' in response.content,
                'suggestions': response.suggested_actions
            }
            
        except Exception as e:
            logger.error(f"Error in code review: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
