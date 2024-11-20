from autogen import UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from typing import Optional, Dict, Any, List
import tempfile
import logging
import os
from src.config import Config
from src.monitor import measure_time

logger = logging.getLogger(__name__)

class ExecutorAgent(UserProxyAgent):
    """
    ExecutorAgent is responsible for safely executing code and providing execution results.
    Inherits from AutoGen's UserProxyAgent for code execution capabilities.
    """

    def __init__(
        self,
        name: str = "executor",
        llm_config: Optional[Dict[str, Any]] = None,
        use_docker: bool = True,
        work_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the executor agent with proper security and monitoring.

        Args:
            name: Agent identifier
            llm_config: Language model configuration
            use_docker: Whether to use Docker for code execution
            work_dir: Working directory for code execution
            **kwargs: Additional configuration options
        """
        llm_config = llm_config or Config.get_agent_config("executor")
        system_message = """
        You are responsible for executing and validating code implementations.
        
        Responsibilities:
        1. Execute provided code safely
        2. Validate outputs and behavior
        3. Monitor performance
        4. Report execution results
        
        Guidelines:
        - Ensure safe execution environment
        - Check for security concerns
        - Monitor resource usage
        - Provide detailed execution logs
        
        Use TERMINATE when execution is complete.
        """

        # Set up working directory
        self.work_dir = work_dir or tempfile.mkdtemp()
        os.makedirs(self.work_dir, exist_ok=True)

        # Configure code executor
        if use_docker:
            executor = DockerCommandLineCodeExecutor(
                image="python:3.12-slim",
                timeout=60,
                work_dir=self.work_dir,
                retry_on_timeout=True,
                resource_limits={
                    'memory': '512m',
                    'cpu_count': 1
                }
            )
        else:
            executor = LocalCommandLineCodeExecutor(
                timeout=60,
                work_dir=self.work_dir,
                retry_on_timeout=True
            )

        # Configure execution settings
        code_execution_config = {
            "executor": executor,
            "use_docker": use_docker,
            "last_n_messages": 3,
            "work_dir": self.work_dir
        }

        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER",
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute provided code safely and return results.

        Args:
            code: Code to execute
            language: Programming language of the code
            context: Additional execution context

        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Prepare execution context
            execution_context = {
                "code": code,
                "language": language,
                **(context or {})
            }

            # Execute code using AutoGen's native execution
            result = await self.execute(
                code,
                language=language,
                context=execution_context
            )

            return {
                'success': True,
                'output': result.output,
                'execution_time': result.execution_time,
                'metadata': {
                    'language': language,
                    'work_dir': self.work_dir,
                    'resource_usage': result.resource_usage
                }
            }

        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'language': language,
                    'work_dir': self.work_dir
                }
            }

    async def validate_execution(
        self,
        code: str,
        expected_output: Any,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate code execution against expected output and test cases.

        Args:
            code: Code to validate
            expected_output: Expected execution result
            test_cases: Optional list of test cases to run

        Returns:
            Dict containing validation results
        """
        try:
            results = []

            # Run main validation
            main_result = await self.execute_code(code)
            results.append({
                'case': 'main',
                'success': main_result['success'],
                'matches_expected': main_result.get('output') == expected_output
            })

            # Run additional test cases
            if test_cases:
                for test_case in test_cases:
                    test_result = await self.execute_code(
                        code,
                        context=test_case.get('context')
                    )
                    results.append({
                        'case': test_case.get('name', 'unnamed'),
                        'success': test_result['success'],
                        'matches_expected': test_result.get('output') == test_case.get('expected')
                    })

            return {
                'success': all(r['success'] and r['matches_expected'] for r in results),
                'results': results,
                'metadata': {
                    'test_count': len(results),
                    'pass_count': sum(1 for r in results if r['success'] and r['matches_expected'])
                }
            }

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'metadata': {}
            }

    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
