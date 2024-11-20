from autogen import AssistantAgent
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from src.config import Config
from src.monitor import measure_time

logger = logging.getLogger(__name__)

class TestingAgent(AssistantAgent):
    """
    Agent specialized in writing and executing test cases for code validation.
    Inherits from AutoGen's AssistantAgent for native integration.
    """
    
    def __init__(
        self,
        name: str = "tester",
        llm_config: Optional[Dict[str, Any]] = None,
        test_frameworks: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the testing agent with AutoGen's native configuration.

        Args:
            name: Agent identifier
            llm_config: Language model configuration
            test_frameworks: List of supported testing frameworks
            **kwargs: Additional configuration options
        """
        system_message = """
        You are responsible for comprehensive testing of code.
        
        Responsibilities:
        1. Write unit tests
        2. Perform integration testing
        3. Validate edge cases
        4. Ensure code coverage
        
        Guidelines:
        - Follow testing best practices
        - Write clear test cases
        - Include positive and negative tests
        - Document test scenarios
        
        Use TERMINATE when testing is complete.
        """
        
        # Default testing tools
        test_tools = [{
            "name": "write_unit_test",
            "description": "Generates unit tests for given code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "framework": {"type": "string"},
                    "requirements": {"type": "array"}
                }
            }
        }, {
            "name": "run_tests",
            "description": "Executes test suite and returns results",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_files": {"type": "array"},
                    "coverage": {"type": "boolean"}
                }
            }
        }]
        
        llm_config = llm_config or Config.get_agent_config("tester")
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            tools=test_tools,
            **kwargs
        )
        
        self.test_frameworks = test_frameworks or ["pytest", "unittest"]

    async def generate_test_suite(
        self,
        code: str,
        requirements: Dict[str, Any],
        framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive test suite for the provided code.
        
        Args:
            code: Source code to test
            requirements: Testing requirements and constraints
            framework: Testing framework to use
            
        Returns:
            Dict containing test suite and metadata
        """
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Generate a complete test suite for:
                                ```python
                {code}                ```
                
                Requirements:
                {requirements}
                
                Use {framework} framework.
                Include:
                1. Unit tests
                2. Edge cases
                3. Error scenarios
                4. Documentation
                """
            }]
            
            response = await self.generate_reply(messages)
            
            return {
                'success': True,
                'test_suite': response.content,
                'framework': framework,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating test suite: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    async def validate_implementation(
        self,
        code: str,
        tests: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates code implementation against test suite and requirements.
        
        Args:
            code: Implementation to validate
            tests: Test suite to run
            requirements: Validation requirements
            
        Returns:
            Dict containing validation results
        """
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Validate implementation against tests:
                
                Code:                ```python
                {code}                ```
                
                Tests:                ```python
                {tests}                ```
                
                Requirements:
                {requirements}
                
                Provide:
                1. Test coverage analysis
                2. Requirements compliance
                3. Edge case handling
                4. Performance considerations
                """
            }]
            
            response = await self.generate_reply(messages)
            
            return {
                'success': True,
                'validation': response.content,
                'passed': 'TERMINATE' in response.content,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    async def generate_test_report(
        self,
        test_results: Dict[str, Any],
        coverage_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive test execution report.
        
        Args:
            test_results: Results from test execution
            coverage_data: Optional coverage metrics
            
        Returns:
            Dict containing test report
        """
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Generate test report for:
                
                Results:
                {test_results}
                
                Coverage:
                {coverage_data or 'No coverage data available'}
                
                Include:
                1. Test execution summary
                2. Coverage analysis
                3. Failed test details
                4. Recommendations
                """
            }]
            
            response = await self.generate_reply(messages)
            
            return {
                'success': True,
                'report': response.content,
                'metadata': {
                    'suggestions': response.suggested_actions,
                    'key_points': response.key_points
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {}
            }
