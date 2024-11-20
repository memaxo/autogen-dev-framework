import time
from functools import wraps
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of async functions.
    
    Args:
        func: The async function to measure
        
    Returns:
        Wrapped function with timing measurement
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            
            # Log execution time
            execution_time = end_time - start_time
            logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
            
            # Add timing metadata to result if it's a dict
            if isinstance(result, dict):
                result['execution_time'] = execution_time
                
            return result
            
        except Exception as e:
            end_time = time.time()
            logger.error(
                f"{func.__name__} failed after {end_time - start_time:.2f} seconds: {str(e)}",
                exc_info=True
            )
            raise
            
    return wrapper

class PerformanceMonitor:
    """Monitors and logs performance metrics for agent operations"""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, float]] = {
            'execution_times': {},
            'success_rates': {},
            'error_counts': {}
        }
    
    @measure_time
    async def monitor_task(
        self,
        task_name: str,
        task_func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Monitor a task execution with performance metrics.
        
        Args:
            task_name: Name of the task to monitor
            task_func: Async function to execute
            args: Positional arguments for task_func
            kwargs: Keyword arguments for task_func
            
        Returns:
            Result from the task execution
        """
        try:
            result = await task_func(*args, **kwargs)
            
            # Update metrics
            self._update_metrics(task_name, True)
            
            return result
            
        except Exception as e:
            # Update error metrics
            self._update_metrics(task_name, False)
            logger.error(f"Task {task_name} failed: {str(e)}", exc_info=True)
            raise
    
    def _update_metrics(self, task_name: str, success: bool) -> None:
        """Update performance metrics for a task"""
        if task_name not in self.metrics['success_rates']:
            self.metrics['success_rates'][task_name] = {
                'success': 0,
                'total': 0
            }
            self.metrics['error_counts'][task_name] = 0
        
        self.metrics['success_rates'][task_name]['total'] += 1
        if success:
            self.metrics['success_rates'][task_name]['success'] += 1
        else:
            self.metrics['error_counts'][task_name] += 1
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current performance metrics"""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        self.metrics = {
            'execution_times': {},
            'success_rates': {},
            'error_counts': {}
        }

# Example usage:
# @measure_time
# async def process_single_task(self, task: str) -> Dict[str, Any]:
#     result = await self._execute_task_with_monitoring(task)
#     return result