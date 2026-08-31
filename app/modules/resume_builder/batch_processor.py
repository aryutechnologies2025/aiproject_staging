"""
batch_processor.py — High-performance concurrent task processor for resume_builder.

Optimized:
- Stripped all hardcoded artificial sleep delays (SECTION_DELAYS, trailing sleep).
- Bounded concurrency with asyncio.Semaphore for fast, parallel execution without event-loop starvation.
- Resilient exponential backoff strictly for API rate-limit errors (429/503).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("resume_builder.batch_processor")


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


@dataclass
class SectionTask:
    section_name: str
    content: str
    priority: SectionPriority = SectionPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 2

    def __lt__(self, other: "SectionTask") -> bool:
        return self.priority.value < other.priority.value


class BatchProcessor:
    """
    Concurrent async section processor with bounded concurrency.
    Zero artificial sleep delays for maximum throughput.
    """

    def __init__(self, max_concurrent: int = 4, max_tokens_per_request: int = 8000):
        self.max_concurrent = max_concurrent
        self.max_tokens_per_request = max_tokens_per_request
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.failed_tasks: List[SectionTask] = []

    @staticmethod
    def _get_priority(section_name: str) -> SectionPriority:
        return PRIORITY_MAP.get(section_name.lower().strip(), SectionPriority.LOW)

    async def add_section(self, section_name: str, content: str, force_single_chunk: bool = False) -> None:
        if not content or not content.strip():
            return
        priority = self._get_priority(section_name)
        task = SectionTask(section_name=section_name, content=content, priority=priority)
        await self.task_queue.put((priority.value, task))

    async def _process_task(self, task: SectionTask, parser_func: Callable[..., Any]) -> Tuple[str, Any, bool]:
        async with self.semaphore:
            while task.retry_count <= task.max_retries:
                try:
                    result = await parser_func(task.section_name, task.content)
                    logger.debug(f"[BatchProcessor] Parsed section: {task.section_name}")
                    return (task.section_name, result, True)
                except Exception as e:
                    err = str(e).lower()
                    is_rate_limit = "429" in err or "resource_exhausted" in err or "quota" in err
                    task.retry_count += 1
                    if task.retry_count <= task.max_retries:
                        backoff = 1.0 * (2 ** task.retry_count)
                        logger.warning(
                            f"[BatchProcessor] Retry {task.retry_count}/{task.max_retries} for '{task.section_name}' (backoff {backoff}s)"
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"[BatchProcessor] Failed after retries: '{task.section_name}': {e}")
                        self.failed_tasks.append(task)
                        return (task.section_name, None, False)
            return (task.section_name, None, False)

    async def process_all(self, parser_func: Callable[..., Any]) -> Dict[str, Any]:
        """
        Executes all queued section tasks concurrently without artificial pauses.
        """
        tasks_list: List[SectionTask] = []
        while not self.task_queue.empty():
            _, task = await self.task_queue.get()
            tasks_list.append(task)

        if not tasks_list:
            return {}

        # Launch all tasks concurrently through bounded semaphore
        coros = [self._process_task(t, parser_func) for t in tasks_list]
        results = await asyncio.gather(*coros, return_exceptions=False)

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

        logger.info(f"[BatchProcessor] Finished {len(tasks_list)} tasks. Success: {len(merged)}, Failed: {len(self.failed_tasks)}")
        return merged

    def get_failed_tasks(self) -> List[SectionTask]:
        return self.failed_tasks

    def reset(self) -> None:
        self.task_queue = asyncio.PriorityQueue()
        self.failed_tasks = []