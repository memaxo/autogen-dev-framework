import pytest
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from src.chat import DevelopmentChat
from src.config import get_default_llm_config

@pytest.fixture
def chat_instance():
    return DevelopmentChat(get_default_llm_config())

@pytest.mark.asyncio
async def test_agent_communication(chat_instance):
    """Test agent communication using AutoGen's built-in features"""
    result = await chat_instance._plan_and_execute(
        "Create a simple hello world function"
    )
    assert result['status'] == 'completed'
    assert 'history' in result

@pytest.mark.asyncio
async def test_error_handling(chat_instance):
    """Test error handling and recovery"""
    result = await chat_instance._execute_agent_task(
        'non_existent_agent',
        'test task',
        {}
    )
    assert not result.success
    assert 'Error' in result.message 