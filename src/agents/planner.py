from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from autogen import AssistantAgent
import logging
from src.config import Config

logger = logging.getLogger(__name__)

@dataclass
class PlanningResult:
    """Container for planning decisions"""
    is_complete: bool
    next_phase: Optional[str]
    next_steps: Optional[List[Dict[str, str]]] = None
    message: Optional[str] = None

class PlanningAgent(AssistantAgent):
    def __init__(
        self,
        name: str = "planner",
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        llm_config = llm_config or Config.get_agent_config("planner")
        
        super().__init__(
            name=name,
            system_message=self._get_system_message(),
            llm_config=llm_config,
            **kwargs
        )
        self._initialize_workflow_templates()
    
    def _get_system_message(self) -> str:
        """Define the planner's core capabilities and responsibilities"""
        return """
        You are the lead architect and development coordinator. Your responsibilities:

        1. Initial Planning:
           - Analyze user requirements thoroughly
           - Break down complex tasks into manageable steps
           - Identify required agent specialists for each step

        2. Workflow Management:
           - Coordinate between specialized agents
           - Maintain project context and state
           - Ensure logical task progression

        3. Quality Control:
           - Review all agent outputs
           - Verify task completion criteria
           - Request additional work if needed

        4. Communication:
           - Provide clear instructions to other agents
           - Maintain context in agent communications
           - Format results for user presentation

        Always think step-by-step and maintain clear documentation of decisions.
        """
    
    def _initialize_workflow_templates(self):
        """Initialize common workflow patterns"""
        self.workflow_templates = {
            'basic_development': [
                {'agent': 'coder', 'phase': 'implementation'},
                {'agent': 'executor', 'phase': 'testing'},
                {'agent': 'tester', 'phase': 'validation'}
            ],
            'bug_fix': [
                {'agent': 'debugger', 'phase': 'analysis'},
                {'agent': 'coder', 'phase': 'fix'},
                {'agent': 'tester', 'phase': 'verification'}
            ]
        }
    
    async def plan_next_steps(
        self,
        task: str,
        current_state: Dict[str, Any]
    ) -> PlanningResult:
        """
        Determine the next steps in the development process
        
        Args:
            task: Original task description or current objective
            current_state: Current project state and context
            
        Returns:
            PlanningResult with next steps or completion status
        """
        # Analyze current state
        phase = current_state['current_phase']
        history = current_state['history']
        results = current_state['results']
        
        # Check if this is the initial planning phase
        if phase == 'planning' and not history:
            return self._create_initial_plan(task)
        
        # Review latest results
        latest_result = history[-1]['result'] if history else None
        
        # Check for task completion
        if self._is_task_complete(task, current_state):
            return PlanningResult(
                is_complete=True,
                next_phase=None,
                message="All requirements have been met successfully."
            )
        
        # Handle failures or continue workflow
        if latest_result and not latest_result.success:
            return self._handle_failure(latest_result, current_state)
        
        # Determine next steps based on current phase and results
        return self._determine_next_steps(task, current_state)
    
    def _create_initial_plan(self, task: str) -> PlanningResult:
        """Create the initial development plan"""
        # Analyze task requirements
        steps = self._breakdown_task(task)
        
        return PlanningResult(
            is_complete=False,
            next_phase='implementation',
            next_steps=steps,
            message="Initial plan created"
        )
    
    def _breakdown_task(self, task: str) -> List[Dict[str, str]]:
        """Break down a task into specific steps for each agent"""
        # This would use the LLM to analyze and break down the task
        # Simplified example:
        return [
            {
                'agent': 'coder',
                'task': f'Implement the following requirement: {task}'
            }
        ]
    
    def _is_task_complete(self, task: str, state: Dict[str, Any]) -> bool:
        """
        Check if all requirements have been met
        
        Args:
            task: Original task description
            state: Current project state
            
        Returns:
            Boolean indicating if task is complete
        """
        # This would use the LLM to analyze results against requirements
        if not state['history']:
            return False
            
        last_phase = state['history'][-1]['phase']
        last_result = state['history'][-1]['result']
        
        return (
            last_phase == 'validation' and 
            last_result.success and 
            not last_result.next_steps
        )
    
    def _handle_failure(
        self,
        failed_result: Any,
        state: Dict[str, Any]
    ) -> PlanningResult:
        """Create a recovery plan for failed tasks"""
        return PlanningResult(
            is_complete=False,
            next_phase='error_recovery',
            next_steps=[{
                'agent': 'debugger',
                'task': f'Analyze and fix the following error: {failed_result.message}'
            }]
        )
    
    def _determine_next_steps(
        self,
        task: str,
        state: Dict[str, Any]
    ) -> PlanningResult:
        """Determine the next steps based on current state"""
        current_phase = state['current_phase']
        
        # Use workflow templates as a guide
        workflow = self.workflow_templates['basic_development']
        current_index = next(
            (i for i, step in enumerate(workflow) 
             if step['phase'] == current_phase),
            -1
        )
        
        if current_index < len(workflow) - 1:
            next_step = workflow[current_index + 1]
            return PlanningResult(
                is_complete=False,
                next_phase=next_step['phase'],
                next_steps=[{
                    'agent': next_step['agent'],
                    'task': f'Continue with {next_step["phase"]} phase for: {task}'
                }]
            )
        
        # If we've completed the workflow, mark as done
        return PlanningResult(
            is_complete=True,
            next_phase=None,
            message="Workflow completed successfully"
        )
    
    async def format_final_response(self, state: Dict[str, Any]) -> str:
        """
        Format the final results for user presentation
        
        Args:
            state: Final project state
            
        Returns:
            Formatted string of results
        """
        # This would use the LLM to create a well-formatted summary
        if state['status'] == 'completed':
            return f"""
Development Task Completed Successfully!

Final Results:
{self._format_results(state['results'])}

Development History:
{self._format_history(state['history'])}
"""
        else:
            return f"""
Development Task Incomplete

Current Status: {state['status']}
Last Phase: {state['current_phase']}

Issues:
{self._format_results(state['results'])}
"""
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format the results from each agent"""
        formatted = []
        for agent, result in results.items():
            formatted.append(f"- {agent}: {result.message}")
        return "\n".join(formatted)
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the development history"""
        formatted = []
        for entry in history:
            formatted.append(
                f"- Phase: {entry['phase']}\n"
                f"  Agent: {entry['agent']}\n"
                f"  Result: {entry['result'].message}\n"
            )
        return "\n".join(formatted)