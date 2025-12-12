"""Resume manager for skipping completed fetch tasks."""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .config_manager import StorageConfig

logger = logging.getLogger(__name__)


class ResumeManager:
    """Track fetch progress to support resume without redundant API calls."""

    def __init__(self, storage_config: StorageConfig, enabled: bool = True):
        self.enabled = enabled
        self.storage_config = storage_config
        self.state: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.filepath = os.path.join(storage_config.save_dir, '.resume_state.json')
        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            with open(self.filepath, 'r', encoding='utf-8') as handle:
                self.state = json.load(handle)
        except FileNotFoundError:
            self.state = {}
        except json.JSONDecodeError as exc:
            logger.warning("Resume state file damaged, ignoring: %s", exc)
            self.state = {}

    def _save(self) -> None:
        if not self.enabled:
            return
        os.makedirs(self.storage_config.save_dir, exist_ok=True)
        tmp_path = f"{self.filepath}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(self.state, handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.filepath)

    def filter_pending(self, entity: str, items: Iterable[str]) -> List[str]:
        if not self.enabled:
            return list(items)
        completed = self.state.get(entity, {})
        return [value for value in items if value not in completed]

    def mark_completed(
        self,
        entity: str,
        items: Iterable[str],
        start_date: str,
        end_date: str
    ) -> None:
        if not self.enabled:
            return
        entity_state = self.state.setdefault(entity, {})
        timestamp = datetime.utcnow().isoformat()
        for value in items:
            entity_state[value] = {
                'start': start_date,
                'end': end_date,
                'updated_at': timestamp
            }
        self._save()

    def record_existing(
        self,
        entity: str,
        existing_items: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        if not self.enabled:
            return
        entity_state = self.state.setdefault(entity, {})
        timestamp = datetime.utcnow().isoformat()
        for value in existing_items:
            if value in entity_state:
                continue
            entity_state[value] = {
                'start': start_date or '',
                'end': end_date or '',
                'updated_at': timestamp
            }
        self._save()

    def invalidate(self, entity: str, items: Optional[Iterable[str]] = None) -> None:
        if not self.enabled:
            return
        if entity not in self.state:
            return
        if items is None:
            self.state.pop(entity, None)
        else:
            entity_state = self.state[entity]
            for value in items:
                entity_state.pop(value, None)
        self._save()
