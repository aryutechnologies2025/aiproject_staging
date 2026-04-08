import asyncio
import logging
import time
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SectionPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


PRIORITY_MAP = {
    "header": SectionPriority.CRITICAL,
    "summary": SectionPriority.HIGH,
    "experience": SectionPriority.HIGH,
    "education": SectionPriority.HIGH,
    "skills": SectionPriority.MEDIUM,
    "certifications": SectionPriority.MEDIUM,
    "projects": SectionPriority.MEDIUM,
    "languages": SectionPriority.LOW,
    "other": SectionPriority.LOW,
}

SECTION_DELAYS = {
    SectionPriority.CRITICAL: 0.0,
    SectionPriority.HIGH: 1.0,
    SectionPriority.MEDIUM: 2.0,
    SectionPriority.LOW: 2.0,
}


@dataclass
class SectionTask:
    section_name: str
    content: str
    priority: SectionPriority
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 2

    def __lt__(self, other):
        return self.priority.value < other.priority.value


class BatchProcessor:
    def __init__(self, max_concurrent: int = 2, max_tokens_per_request: int = 6000):
        self.max_concurrent = max_concurrent
        self.max_tokens_per_request = max_tokens_per_request
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.failed_tasks: List[SectionTask] = []

    @staticmethod
    def _get_priority(section_name: str) -> SectionPriority:
        return PRIORITY_MAP.get(section_name, SectionPriority.LOW)

    async def add_section(self, section_name: str, content: str, force_single_chunk: bool = False):
        if not content.strip():
            return
        priority = self._get_priority(section_name)
        task = SectionTask(section_name=section_name, content=content, priority=priority)
        await self.task_queue.put((priority.value, task))

    async def _process_task(self, task: SectionTask, parser_func: Callable) -> Tuple[str, Any, bool]:
        async with self.semaphore:
            delay = SECTION_DELAYS.get(task.priority, 1.0)
            if delay > 0:
                await asyncio.sleep(delay)

            while task.retry_count <= task.max_retries:
                try:
                    result = await parser_func(task.section_name, task.content)
                    logger.info(f"Parsed section: {task.section_name}")
                    return (task.section_name, result, True)
                except Exception as e:
                    err = str(e)
                    is_rate_limit = "429" in err or "rate limit" in err.lower()
                    task.retry_count += 1
                    if task.retry_count <= task.max_retries:
                        backoff = (4.0 if is_rate_limit else 2.0) * (2 ** task.retry_count)
                        logger.warning(f"Retry {task.retry_count} for {task.section_name}, waiting {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"Failed after retries: {task.section_name}")
                        self.failed_tasks.append(task)
                        return (task.section_name, None, False)

    async def process_all(self, parser_func: Callable) -> Dict[str, Any]:
        tasks_list = []
        while not self.task_queue.empty():
            _, task = await self.task_queue.get()
            tasks_list.append(task)

        tasks_list.sort(key=lambda t: t.priority.value)

        if not tasks_list:
            return {}

        critical_high = [t for t in tasks_list if t.priority.value <= 2]
        medium_low = [t for t in tasks_list if t.priority.value > 2]

        results = []

        if critical_high:
            batch_coros = [self._process_task(t, parser_func) for t in critical_high]
            batch_results = await asyncio.gather(*batch_coros, return_exceptions=False)
            results.extend(batch_results)
            await asyncio.sleep(1.5)

        if medium_low:
            batch_coros = [self._process_task(t, parser_func) for t in medium_low]
            batch_results = await asyncio.gather(*batch_coros, return_exceptions=False)
            results.extend(batch_results)

        merged: Dict[str, Any] = {}
        for section_name, result, success in results:
            if not success or result is None:
                continue
            if section_name not in merged:
                merged[section_name] = result
            else:
                existing = merged[section_name]
                if isinstance(existing, list) and isinstance(result, list):
                    merged[section_name] = existing + result
                elif isinstance(existing, dict) and isinstance(result, dict):
                    merged[section_name] = {**existing, **result}

        logger.info(f"Batch done. Success: {len(merged)}, Failed: {len(self.failed_tasks)}")
        return merged

    def get_failed_tasks(self) -> List[SectionTask]:
        return self.failed_tasks

    def reset(self):
        self.task_queue = asyncio.PriorityQueue()
        self.failed_tasks = []
        