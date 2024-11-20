from autogen import AssistantAgent, GroupChat, GroupChatManager
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging
from src.config import Config
from src.monitor import measure_time, PerformanceMonitor

# Update imports for specialized agents
from src.agents.planner import PlanningAgent
from src.agents.coder import CoderAgent
from src.agents.debugger import DebuggingAgent
from src.agents.executor import ExecutorAgent
from src.agents.tester import TestingAgent

logging.basicConfig(**Config.get_logging_config())
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    success: bool
    output: Any
    message: str
    next_steps: Optional[List[str]] = None

class DevelopmentChat:
    def __init__(self):
        """Initialize with centralized configuration"""
        self.monitor = PerformanceMonitor()
        self.planner = PlanningAgent()
        self._initialize_agent_pool()
        self._setup_group_chat()
    
    def _initialize_agent_pool(self):
        """Initialize agents using centralized configuration"""
        self.agent_pool = {
            'coder': CoderAgent(),
            'executor': ExecutorAgent(),
            'debugger': DebuggingAgent(),
            'tester': TestingAgent()
        }
    
    def _setup_group_chat(self):
        """Setup GroupChat for better agent coordination"""
        agents = [self.planner, *self.agent_pool.values()]
        self.group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=10
        )
        self.chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=Config.get_agent_config("planner")
        )
    
    @measure_time
    async def _execute_agent_task(
        self,
        agent_name: str,
        task: str,
        context: Dict[str, Any]
    ) -> TaskResult:
        """Execute task with better error handling"""
        try:
            agent = self.agent_pool[agent_name]
            
            # Use AutoGen's built-in retry mechanism
            response = await self.chat_manager.send(
                message=task,
                sender=self.planner,
                recipient=agent
            )
            
            return TaskResult(
                success=True,
                output=response.content,
                message=response.key_points,
                next_steps=response.suggested_actions
            )
            
        except Exception as e:
            logger.error(f"Error in {agent_name}: {str(e)}", exc_info=True)
            return TaskResult(
                success=False,
                output=None,
                message=f"Error: {str(e)}"
            )

    async def _plan_and_execute(self, task: str) -> Dict[str, Any]:
        """Use AutoGen's built-in state management"""
        try:
            # Initialize group chat for this task
            chat_result = await self.chat_manager.run(
                initial_message=task,
                sender=self.planner
            )
            
            return {
                'status': 'completed',
                'results': chat_result.content,
                'history': self.group_chat.messages,
                'metrics': chat_result.metrics
            }
            
        except Exception as e:
            logger.error("Task execution failed", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'history': self.group_chat.messages
            }

def main():
    """Entry point for the chat application"""
    chat = DevelopmentChat()
    asyncio.run(chat.chat_loop())

if __name__ == "__main__":
    main()