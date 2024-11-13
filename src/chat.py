import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_ext.models import OpenAIChatCompletionClient
from src.agents.planner import PlanningAgent
from src.agents.coder import CodingAgent
from src.agents.executor import ExecutionAgent
from src.agents.debugger import DebuggingAgent
from src.agents.tester import TestingAgent

@dataclass
class TaskResult:
    """Container for task execution results from each agent"""
    success: bool
    output: Any
    message: str
    next_steps: Optional[List[str]] = None

class DevelopmentChat:
    """
    Manages the development workflow chat, with the planner agent as the central coordinator.
    All other agents communicate through the planner.
    """
    
    def __init__(self):
        """Initialize the chat with all required agents"""
        # Initialize all agents
        self.planner = PlanningAgent()
        self._initialize_agent_pool()
        
        # Set up the planner's system message to handle coordination
        self.planner.update_system_message("""
        You are the lead architect and project coordinator. Your responsibilities:
        1. Analyze user requirements and break them down into specific tasks
        2. Delegate tasks to appropriate specialized agents
        3. Review results from each agent
        4. Determine if additional work is needed
        5. Maintain the overall project state
        6. Communicate final results back to the user
        
        Only communicate directly with the user and specialized agents.
        All other agents must communicate through you.
        Ensure each task is properly completed before moving to the next phase.
        """)
    
    def _initialize_agent_pool(self):
        """Initialize the pool of specialized agents"""
        self.agent_pool = {
            'coder': CodingAgent(),
            'executor': ExecutionAgent(),
            'debugger': DebuggingAgent(),
            'tester': TestingAgent()
        }
    
    async def _execute_agent_task(
        self,
        agent_name: str,
        task: str,
        context: Dict[str, Any]
    ) -> TaskResult:
        """
        Execute a task with a specific agent
        
        Args:
            agent_name: Name of the agent to execute the task
            task: Task description
            context: Additional context and requirements
            
        Returns:
            TaskResult containing the execution results
        """
        agent = self.agent_pool[agent_name]
        
        try:
            # Create a message stream between planner and specific agent
            stream = await agent.execute_task(
                task=task,
                context=context,
                reply_to=self.planner
            )
            
            # Process the results
            result = await stream.get_final_response()
            
            return TaskResult(
                success=True,
                output=result.output,
                message=result.message,
                next_steps=result.suggested_next_steps
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                output=None,
                message=f"Error during {agent_name} task execution: {str(e)}"
            )

    async def _plan_and_execute(self, task: str) -> Dict[str, Any]:
        """
        Core workflow orchestration method
        
        Args:
            task: Initial task description from user
            
        Returns:
            Dictionary containing final results and status
        """
        project_state = {
            'status': 'in_progress',
            'current_phase': 'planning',
            'results': {},
            'context': {},
            'history': []
        }
        
        while project_state['status'] == 'in_progress':
            # Get next steps from planner
            plan_result = await self.planner.plan_next_steps(
                task=task,
                current_state=project_state
            )
            
            if plan_result.is_complete:
                project_state['status'] = 'completed'
                break
                
            # Execute each planned step
            for step in plan_result.next_steps:
                agent_name = step['agent']
                agent_task = step['task']
                
                # Execute the task with the appropriate agent
                result = await self._execute_agent_task(
                    agent_name=agent_name,
                    task=agent_task,
                    context=project_state['context']
                )
                
                # Update project state
                project_state['results'][agent_name] = result
                project_state['history'].append({
                    'phase': project_state['current_phase'],
                    'agent': agent_name,
                    'task': agent_task,
                    'result': result
                })
                
                # If any step fails, let planner decide how to proceed
                if not result.success:
                    break
            
            # Update current phase
            project_state['current_phase'] = plan_result.next_phase
        
        return project_state

    async def chat_loop(self):
        """Main chat loop for interaction with the user"""
        print("Welcome to the Development Chat! Type 'exit' to quit.")
        
        while True:
            # Get user input
            user_input = input("\nWhat would you like to develop? ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            print("\nPlanning and executing your request...")
            
            try:
                # Execute the development workflow
                result = await self._plan_and_execute(user_input)
                
                # Let planner format and present the final results
                final_response = await self.planner.format_final_response(result)
                
                print("\nResults:")
                print(final_response)
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again or type 'exit' to quit.")

    async def process_single_task(self, task: str) -> Dict[str, Any]:
        """
        Process a single task without continuous chat loop
        Useful for programmatic access
        
        Args:
            task: Task description
            
        Returns:
            Dictionary containing final results and status
        """
        return await self._plan_and_execute(task)

def main():
    """Entry point for the chat application"""
    chat = DevelopmentChat()
    asyncio.run(chat.chat_loop())

if __name__ == "__main__":
    main()