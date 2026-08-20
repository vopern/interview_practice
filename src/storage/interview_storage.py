"""Repository for interview sessions and static company context.

Encapsulates ALL knowledge of file paths and on-disk JSON layout. Nothing else
in the codebase touches the filesystem, so swapping this for S3 or a database
later means reimplementing this one class with the same interface.

Layout: <base_dir>/interviews/<slug>/session_<id>.json
        <base_dir>/context/<company>/*  (static context files, any text format)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.models.interview import InterviewSession, InterviewSettings

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


class InterviewStorage:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir) / "interviews"
        self.context_dir = Path(base_dir) / "context"

    def _slug_dir(self, slug: str) -> Path:
        return self.base_dir / slug

    def _session_path(self, slug: str, session_id: str) -> Path:
        return self._slug_dir(slug) / f"session_{session_id}.json"

    def save_session(self, session: InterviewSession) -> Path:
        path = self._session_path(session.settings.slug, session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def load_sessions(
        self, settings: InterviewSettings, before: Optional[str] = None
    ) -> list[InterviewSession]:
        """All stored sessions for these settings, oldest first."""
        return self.load_all(settings.slug, before)

    def load_all(
        self, slug: str, before: Optional[str] = None
    ) -> list[InterviewSession]:
        """Stored sessions for a slug, oldest first.

        `before` is a session id and cuts the list off strictly before it, which
        is how a replayed round is given the history its source ran with,
        without being shown the source itself. Ids are "%Y%m%d_%H%M%S", so
        comparing them as strings orders them by time; the sorted glob below
        already relies on that.
        """
        sessions = []
        for path in sorted(self._slug_dir(slug).glob("session_*.json")):
            session = self._load_file(path)
            if session and (before is None or session.id < before):
                sessions.append(session)
        return sessions

    def _load_file(self, path: Path) -> Optional[InterviewSession]:
        try:
            return InterviewSession.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Skipping unreadable session file %s: %s", path, e)
            return None

    def _context_folders(self, company: str) -> list[Path]:
        """Every context folder whose name normalizes to this company.

        Matching is case- and punctuation-insensitive ("ABC Corp" matches
        "abc_corp"), which is why this scans the directory instead of
        building a path.
        """
        target = _normalize(company)
        if not target or not self.context_dir.exists():
            return []
        return [
            path
            for path in sorted(self.context_dir.iterdir())
            if path.is_dir() and _normalize(path.name) == target
        ]

    @staticmethod
    def _folder_files(folder: Path) -> list[Path]:
        """Files in a context folder, hidden ones (.gitkeep etc.) skipped."""
        return [
            path
            for path in sorted(folder.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]

    def load_company_context(self, company: str) -> str:
        """Concatenated static context files from <base_dir>/context/<company>/.

        Returns "" when there is no matching folder or no readable text files.
        """
        sections = []
        for folder in self._context_folders(company):
            for path in self._folder_files(folder):
                text = self.read_context_file(path)
                if text:
                    sections.append(f"--- {path.relative_to(folder)} ---\n{text}")
        return "\n\n".join(sections)

    def read_context_file(self, path: Path) -> str:
        """Text of one context file, or "" if it cannot be decoded.

        An unreadable file is skipped rather than raised on, so one bad drop-in
        never takes down a whole company's context.
        """
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Skipping unreadable context file %s: %s", path, e)
            return ""

    def list_context_files(self, company: str) -> list[Path]:
        """Context files for a company, in the order load_company_context reads them."""
        return [
            path
            for folder in self._context_folders(company)
            for path in self._folder_files(folder)
        ]

    def list_context_companies(self) -> list[str]:
        """Companies with at least one context file, as a person would write them.

        Folder names are the normalized form ("abc_corp"); everything
        downstream of a picked name — the prompts, the transcription priming, the
        interview slug — wants the readable one. Lookups here are
        normalization-insensitive, so the name handed back still resolves to the
        folder it came from. An acronym folder comes back title-cased ("abc" ->
        "Abc"); typing the real name over it costs nothing and lands in the same
        folder.
        """
        if not self.context_dir.exists():
            return []
        return [
            path.name.replace("_", " ").title()
            for path in sorted(self.context_dir.iterdir())
            if path.is_dir() and self._folder_files(path)
        ]

    def save_company_context(self, company: str, filename: str, text: str) -> Path:
        """Write a context file, reusing the company's existing folder if it has one.

        "ABC Corp" has to land in the folder the interview planner already
        reads, not next to it in a second one.
        """
        folders = self._context_folders(company)
        folder = folders[0] if folders else self.context_dir / _normalize(company)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_text(text, encoding="utf-8")
        return path

    def list_slugs(self) -> list[tuple[str, InterviewSettings]]:
        """All (slug, settings) pairs that have at least one stored session."""
        result = []
        if not self.base_dir.exists():
            return result
        for slug_dir in sorted(p for p in self.base_dir.iterdir() if p.is_dir()):
            for path in sorted(slug_dir.glob("session_*.json")):
                session = self._load_file(path)
                if session:
                    result.append((slug_dir.name, session.settings))
                    break
        return result
