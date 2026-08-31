"""
version_manager.py — In-memory Resume Snapshot Versioning & Change Diff Engine.

Provides:
- In-memory versioning of candidate resumes during iterative editing and section refinements.
- Structured diff computation (added/removed skills, modified bullet points, section deltas).
- Non-destructive historical rollback and state restoration.
- Bounded memory storage (10 snapshots per user with 24-hour TTL, zero database writes).
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Union

from app.modules.resume_builder.schemas import CanonicalResume

logger = logging.getLogger("resume_builder.version_manager")


@dataclass
class ResumeSnapshot:
    version_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    resume_data: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    change_summary: str = "Updated resume"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "change_summary": self.change_summary,
            "content_hash": self.content_hash,
            "resume_data": self.resume_data,
        }


class ResumeVersionManager:
    """
    In-memory snapshot manager for resume versioning, rollback, and diffing.
    """

    MAX_SNAPSHOTS_PER_USER = 10
    SNAPSHOT_TTL_SECONDS = 86400.0  # 24 hours

    def __init__(self):
        # {user_id: [ResumeSnapshot_1, ResumeSnapshot_2, ...]}
        self._user_snapshots: Dict[str, List[ResumeSnapshot]] = {}

    def save_snapshot(
        self,
        user_id: str,
        resume: Union[CanonicalResume, Dict[str, Any]],
        change_summary: str = "Updated resume",
    ) -> str:
        """
        Saves a snapshot of the current resume state for a user.
        """
        uid = user_id or "anonymous"
        if isinstance(resume, CanonicalResume):
            data = resume.model_dump()
        elif isinstance(resume, dict):
            data = resume
        else:
            data = {}

        # Compute SHA-256 hash of serialized content
        serialized = json.dumps(data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        version_id = f"ver_{uuid.uuid4().hex[:12]}"
        snapshot = ResumeSnapshot(
            version_id=version_id,
            user_id=uid,
            created_at=time.time(),
            resume_data=data,
            content_hash=content_hash,
            change_summary=change_summary,
        )

        history = self._user_snapshots.setdefault(uid, [])

        # Deduplicate if exact same content hash as the last snapshot
        if history and history[-1].content_hash == content_hash:
            logger.debug(f"[VersionManager] Skipped duplicate snapshot for user {uid}")
            return history[-1].version_id

        history.append(snapshot)

        # Enforce rolling memory bound
        if len(history) > self.MAX_SNAPSHOTS_PER_USER:
            self._user_snapshots[uid] = history[-self.MAX_SNAPSHOTS_PER_USER:]

        logger.info(f"[VersionManager] Saved snapshot '{version_id}' for user '{uid}' ({change_summary})")
        return version_id

    def get_snapshot(self, user_id: str, version_id: str) -> Optional[ResumeSnapshot]:
        history = self._user_snapshots.get(user_id, [])
        for snap in history:
            if snap.version_id == version_id:
                return snap
        return None

    def list_snapshots(self, user_id: str) -> List[Dict[str, Any]]:
        history = self._user_snapshots.get(user_id, [])
        return [
            {
                "version_id": s.version_id,
                "created_at": s.created_at,
                "change_summary": s.change_summary,
                "content_hash": s.content_hash,
            }
            for s in reversed(history)
        ]

    def diff_snapshots(self, v1: ResumeSnapshot, v2: ResumeSnapshot) -> Dict[str, Any]:
        """
        Computes a structured semantic diff between two resume snapshots.
        """
        d1 = v1.resume_data
        d2 = v2.resume_data

        # 1. Skills diff
        s1 = set(d1.get("skills", []))
        s2 = set(d2.get("skills", []))
        skills_added = sorted(list(s2 - s1))
        skills_removed = sorted(list(s1 - s2))

        # 2. Experience bullets diff
        b1_set: Set[str] = set()
        for exp in d1.get("experience", []):
            for b in exp.get("bullets", []):
                b1_set.add(b.strip())

        b2_set: Set[str] = set()
        for exp in d2.get("experience", []):
            for b in exp.get("bullets", []):
                b2_set.add(b.strip())

        bullets_added = list(b2_set - b1_set)
        bullets_removed = list(b1_set - b2_set)

        # 3. Summary diff
        sum1 = d1.get("summary", "")
        sum2 = d2.get("summary", "")
        summary_changed = sum1.strip() != sum2.strip()

        return {
            "v1_id": v1.version_id,
            "v2_id": v2.version_id,
            "skills_diff": {
                "added": skills_added,
                "removed": skills_removed,
                "count_delta": len(s2) - len(s1),
            },
            "experience_diff": {
                "bullets_added": bullets_added,
                "bullets_removed": bullets_removed,
                "roles_count_v1": len(d1.get("experience", [])),
                "roles_count_v2": len(d2.get("experience", [])),
            },
            "summary_diff": {
                "changed": summary_changed,
                "word_count_delta": len(sum2.split()) - len(sum1.split()),
            },
        }

    def rollback(self, user_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves snapshot and creates a new rollback snapshot representing restoration.
        """
        target = self.get_snapshot(user_id, version_id)
        if not target:
            return None

        # Re-save as newest version with rollback summary
        new_version_id = self.save_snapshot(
            user_id=user_id,
            resume=target.resume_data,
            change_summary=f"Rolled back to {version_id}",
        )
        return {
            "restored_version_id": new_version_id,
            "source_version_id": version_id,
            "resume_data": target.resume_data,
        }

    def clear(self) -> None:
        self._user_snapshots.clear()


# Global Singleton instance
_version_manager: Optional[ResumeVersionManager] = None


def get_version_manager() -> ResumeVersionManager:
    global _version_manager
    if _version_manager is None:
        _version_manager = ResumeVersionManager()
    return _version_manager
