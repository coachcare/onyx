"""Unified GitHub connector – complete file with binary-skip + logging.

Merges the legacy code-file crawler with checkpoint-based PR/Issue logic and
implements all abstract methods required by `CheckpointedConnector`.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Generator, Iterator
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, cast

from github import Github, RateLimitExceededException
from github.ContentFile import ContentFile
from github.GithubException import GithubException
from github.Issue import Issue
from github.PaginatedList import PaginatedList
from github.PullRequest import PullRequest
from github.Repository import Repository
from github.Requester import Requester
from pydantic import BaseModel
from typing_extensions import override

from onyx.configs.app_configs import GITHUB_CONNECTOR_BASE_URL, INDEX_BATCH_SIZE
from onyx.configs.constants import DocumentSource
from onyx.connectors.exceptions import (
    ConnectorValidationError,
    CredentialExpiredError,
    InsufficientPermissionsError,
    UnexpectedValidationError,
)
from onyx.connectors.interfaces import (
    CheckpointOutput,
    CheckpointedConnector,
    ConnectorCheckpoint,
    ConnectorFailure,
    SecondsSinceUnixEpoch,
)
from onyx.connectors.models import (
    ConnectorMissingCredentialError,
    Document,
    DocumentFailure,
    TextSection,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Constants / globals
# ---------------------------------------------------------------------------
ITEMS_PER_PAGE = 100
CURSOR_LOG_FREQUENCY = 50
_MAX_NUM_RATE_LIMIT_RETRIES = 5
ONE_DAY = timedelta(days=1)
_CODE_BATCH_SIZE = INDEX_BATCH_SIZE if "INDEX_BATCH_SIZE" in globals() else 25

# ---------------------------------------------------------------------------
# Helper: safe text decoding
# ---------------------------------------------------------------------------
_TEXT_CHARSETS = ("utf-8", "latin-1")


def _safe_decode(data: bytes) -> str | None:
    """Try common encodings; return None if binary/undecodable."""
    for enc in _TEXT_CHARSETS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Rate-limit helper
# ---------------------------------------------------------------------------
def _sleep_after_rate_limit_exception(client: Github) -> None:
    remaining = (
        client.get_rate_limit().core.reset.replace(tzinfo=timezone.utc)
        - datetime.now(tz=timezone.utc)
    )
    remaining += timedelta(minutes=1)
    logger.notice(f"Hit GitHub rate-limit – sleeping {remaining.seconds} s")
    time.sleep(remaining.seconds)


# ---------------------------------------------------------------------------
# PaginatedList “nextUrl” helpers
# ---------------------------------------------------------------------------
def _discover_next_url_key(pag_list: PaginatedList) -> str:
    for key in pag_list.__dict__:
        if key.endswith("__nextUrl") or key.endswith("nextUrl"):
            return key
    return ""


def _get_next_url(pag_list: PaginatedList, key: str) -> str | None:
    return getattr(pag_list, key) if key else None


def _set_next_url(pag_list: PaginatedList, key: str, url: str | None) -> None:
    if key:
        setattr(pag_list, key, url)
    elif url:
        raise ValueError("Could not locate next-URL attribute on PaginatedList")


# ---------------------------------------------------------------------------
# Cursor-aware pagination fallback
# ---------------------------------------------------------------------------
def _paginate_until_error(
    get_paginated: Callable[[], PaginatedList],
    cursor_url: str | None,
    prev_count: int,
    cursor_cb: Callable[[str | None, int], None],
    retrying: bool = False,
) -> Generator[Any, None, None]:
    count = prev_count
    pag_list = get_paginated()
    key = _discover_next_url_key(pag_list)

    if cursor_url:
        _set_next_url(pag_list, key, cursor_url)
    elif retrying:
        logger.warning(
            "Retrying from expired cursor – will re-download earlier pages."
        )
        pag_list = cast(PaginatedList, pag_list[prev_count:])
        count = 0

    try:
        for obj in pag_list:
            count += 1
            yield obj
            cursor_cb(_get_next_url(pag_list, key), count)
            if count % CURSOR_LOG_FREQUENCY == 0:
                logger.info(
                    f"Fetched {count} objects … cursor={_get_next_url(pag_list, key)}"
                )
    except Exception as exc:  # pragma: no cover
        logger.exception("Cursor-pagination error: %s", exc)
        if count - prev_count > 0:
            raise
        if _get_next_url(pag_list, key) and not retrying:
            yield from _paginate_until_error(
                get_paginated, None, prev_count, cursor_cb, retrying=True
            )
            return
        raise


# ---------------------------------------------------------------------------
# Unified batch loader (page or cursor) with RL handling
# ---------------------------------------------------------------------------
def _get_batch_rate_limited(
    get_paginated: Callable[[], PaginatedList],
    page_num: int,
    cursor_url: str | None,
    prev_count: int,
    cursor_cb: Callable[[str | None, int], None],
    client: Github,
    attempt: int = 0,
) -> Generator[Any, None, None]:
    if attempt > _MAX_NUM_RATE_LIMIT_RETRIES:
        raise RuntimeError("Exceeded retry limit while fetching GitHub data")
    try:
        if cursor_url:
            yield from _paginate_until_error(
                get_paginated, cursor_url, prev_count, cursor_cb
            )
            return
        objs = list(get_paginated().get_page(page_num))
        for obj in objs:  # trigger RL here
            if hasattr(obj, "raw_data"):
                getattr(obj, "raw_data")
        yield from objs
    except RateLimitExceededException:
        _sleep_after_rate_limit_exception(client)
        yield from _get_batch_rate_limited(
            get_paginated,
            page_num,
            cursor_url,
            prev_count,
            cursor_cb,
            client,
            attempt + 1,
        )
    except GithubException as exc:
        if exc.status == 422 and (
            "cursor" in (exc.data or {}).get("message", "")
            or "cursor" in (exc.message or "")
        ):
            yield from _paginate_until_error(
                get_paginated, cursor_url, prev_count, cursor_cb
            )
        else:
            raise


# ---------------------------------------------------------------------------
# Document converters
# ---------------------------------------------------------------------------
def _pr_to_doc(pr: PullRequest) -> Document:
    return Document(
        id=pr.html_url,
        sections=[TextSection(link=pr.html_url, text=pr.body or "")],
        source=DocumentSource.GITHUB,
        semantic_identifier=pr.title,
        doc_updated_at=(
            pr.updated_at.replace(tzinfo=timezone.utc) if pr.updated_at else None
        ),
        metadata={"merged": str(pr.merged), "state": pr.state},
    )


def _issue_to_doc(issue: Issue) -> Document:
    return Document(
        id=issue.html_url,
        sections=[TextSection(link=issue.html_url, text=issue.body or "")],
        source=DocumentSource.GITHUB,
        semantic_identifier=issue.title,
        doc_updated_at=issue.updated_at.replace(tzinfo=timezone.utc),
        metadata={"state": issue.state},
    )


def _code_file_to_doc(cf: ContentFile, repo_name: str) -> Document | None:
    link = f"https://github.com/{repo_name}/blob/{cf.sha}/{cf.path}"
    text = _safe_decode(cf.decoded_content)
    if text is None:
        logger.info(f"Skipping binary file: {repo_name}/{cf.path} ({cf.size} bytes)")
        return None
    return Document(
        id=link,
        sections=[TextSection(link=link, text=text)],
        source=DocumentSource.GITHUB,
        semantic_identifier=cf.path,
        doc_updated_at=datetime.now(tz=timezone.utc),
        metadata={"file_size": str(cf.size)},
    )


# ---------------------------------------------------------------------------
# Checkpoint schema
# ---------------------------------------------------------------------------
class SerializedRepository(BaseModel):
    id: int
    headers: dict[str, str | int]
    raw_data: dict[str, Any]

    def to_repository(self, requester: Requester) -> Repository:  # noqa: N802
        return Repository(requester, self.headers, self.raw_data, completed=True)


class Stage(str, Enum):
    CODE = "code"
    PRS = "prs"
    ISSUES = "issues"


class GithubCheckpoint(ConnectorCheckpoint):
    stage: Stage
    curr_page: int
    cached_repo_ids: list[int] | None = None
    cached_repo: SerializedRepository | None = None
    num_retrieved: int = 0
    cursor_url: str | None = None

    def reset(self) -> None:  # noqa: D401
        self.curr_page = 0
        self.num_retrieved = 0
        self.cursor_url = None


def _make_cursor_cb(cp: GithubCheckpoint) -> Callable[[str | None, int], None]:
    def _cb(cursor: str | None, num: int) -> None:
        if cursor:
            cp.cursor_url = cursor
        cp.num_retrieved = num

    return _cb


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------
class GithubConnector(CheckpointedConnector[GithubCheckpoint]):
    """Fetch code, PRs and Issues from GitHub with checkpointing."""

    # ----------------------- init / credentials -----------------------
    def __init__(
        self,
        *,
        repo_owner: str,
        repositories: str | None = None,
        state_filter: str = "all",
        include_code: bool = True,
        include_prs: bool = True,
        include_issues: bool = False,
        code_batch_size: int = _CODE_BATCH_SIZE,
    ) -> None:
        self.repo_owner = repo_owner
        self.repositories = repositories
        self.state_filter = state_filter
        self.include_code = include_code
        self.include_prs = include_prs
        self.include_issues = include_issues
        self.code_batch_size = code_batch_size
        self.github_client: Github | None = None

    def load_credentials(self, creds: dict[str, Any]) -> dict[str, Any] | None:  # noqa: D401
        self.github_client = (
            Github(
                creds["github_access_token"],
                base_url=GITHUB_CONNECTOR_BASE_URL,
                per_page=ITEMS_PER_PAGE,
            )
            if GITHUB_CONNECTOR_BASE_URL
            else Github(creds["github_access_token"], per_page=ITEMS_PER_PAGE)
        )
        return None

    # ----------------------- repo utilities -----------------------
    def _split_repo_names(self) -> list[str]:
        return [n.strip() for n in (self.repositories or "").split(",") if n.strip()]

    def _repo_safe(self, full_name: str, attempt: int = 0) -> Repository:
        if attempt > _MAX_NUM_RATE_LIMIT_RETRIES:
            raise RuntimeError("Repo fetch retries exceeded")
        try:
            return self.github_client.get_repo(full_name)  # type: ignore[attr-defined]
        except RateLimitExceededException:
            _sleep_after_rate_limit_exception(self.github_client)  # type: ignore[arg-type]
            return self._repo_safe(full_name, attempt + 1)

    def _explicit_repos(self) -> list[Repository]:
        return [
            self._repo_safe(f"{self.repo_owner}/{name}")
            for name in self._split_repo_names()
        ]

    def _all_repos(self, attempt: int = 0) -> list[Repository]:
        if attempt > _MAX_NUM_RATE_LIMIT_RETRIES:
            raise RuntimeError("Repo list retries exceeded")
        try:
            try:
                org = self.github_client.get_organization(self.repo_owner)  # type: ignore[attr-defined]
                return list(org.get_repos())
            except GithubException:
                user = self.github_client.get_user(self.repo_owner)  # type: ignore[attr-defined]
                return list(user.get_repos())
        except RateLimitExceededException:
            _sleep_after_rate_limit_exception(self.github_client)  # type: ignore[arg-type]
            return self._all_repos(attempt + 1)

    # ----------------------- code traversal -----------------------
    def _walk_code_files(self, repo: Repository) -> Iterator[ContentFile]:
        def _walk(path: str) -> Iterator[ContentFile]:
            tries = 0
            while tries < _MAX_NUM_RATE_LIMIT_RETRIES:
                try:
                    contents = repo.get_contents(path)
                    if not isinstance(contents, list):
                        contents = [contents]
                    for entry in contents:
                        if entry.type == "dir":
                            yield from _walk(entry.path)
                        elif entry.size < 1_000_000:
                            yield entry
                    return
                except RateLimitExceededException:
                    _sleep_after_rate_limit_exception(self.github_client)  # type: ignore[arg-type]
                    tries += 1
                except GithubException as exc:
                    logger.warning(f"Error listing {path} in {repo.full_name}: {exc}")
                    return
            logger.error(f"Too many retries walking {repo.full_name}:{path}")

        yield from _walk("")

    def _yield_code_documents(
        self, repo: Repository
    ) -> Generator[Document, None, None]:
        buffer: list[ContentFile] = []
        for cf in self._walk_code_files(repo):
            buffer.append(cf)
            if len(buffer) >= self.code_batch_size:
                for f in buffer:
                    doc = _code_file_to_doc(f, repo.full_name)
                    if doc:
                        yield doc
                buffer.clear()
        for f in buffer:
            doc = _code_file_to_doc(f, repo.full_name)
            if doc:
                yield doc

    # ----------------------- core fetch loop -----------------------
    def _fetch_from_github(
        self,
        checkpoint: GithubCheckpoint,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Generator[Document | ConnectorFailure, None, GithubCheckpoint]:
        if self.github_client is None:
            raise ConnectorMissingCredentialError("GitHub")

        cp = copy.deepcopy(checkpoint)
        cursor_cb = _make_cursor_cb(cp)

        # 1️⃣ initial repo discovery
        if cp.cached_repo_ids is None:
            repos = self._explicit_repos() if self.repositories else self._all_repos()
            if not repos:
                cp.has_more = False
                return cp

            current = repos.pop()
            cp.cached_repo_ids = [r.id for r in repos]
            cp.cached_repo = SerializedRepository(
                id=current.id,
                headers=current.raw_headers,
                raw_data=current.raw_data,
            )
            cp.stage = Stage.CODE if self.include_code else Stage.PRS
            cp.reset()
            return cp

        # 2️⃣ rebuild repo object
        if cp.cached_repo is None:
            raise ValueError("Checkpoint corrupt: missing cached_repo")

        try:
            requester = (
                self.github_client._requester  # type: ignore[attr-defined]
                if hasattr(self.github_client, "_requester")
                else self.github_client._Github__requester  # type: ignore[attr-defined]
            )
            repo = cp.cached_repo.to_repository(requester)
        except Exception:  # pragma: no cover
            repo = self.github_client.get_repo(cp.cached_repo.id)

        # 3️⃣ CODE stage
        if self.include_code and cp.stage == Stage.CODE:
            logger.info(f"[GitHub] Crawling code for {repo.full_name}")
            for doc in self._yield_code_documents(repo):
                yield doc
            cp.stage = Stage.PRS
            cp.reset()
            return cp

        # 4️⃣ PR stage
        if self.include_prs and cp.stage == Stage.PRS:
            pr_batch = _get_batch_rate_limited(
                lambda: repo.get_pulls(
                    state=self.state_filter, sort="updated", direction="desc"
                ),
                cp.curr_page,
                cp.cursor_url,
                cp.num_retrieved,
                cursor_cb,
                self.github_client,
            )
            cp.curr_page += 1
            done = False
            produced = 0
            for pr in pr_batch:
                produced += 1
                if (
                    start
                    and pr.updated_at
                    and pr.updated_at.replace(tzinfo=timezone.utc) < start
                ):
                    done = True
                    break
                if (
                    end
                    and pr.updated_at
                    and pr.updated_at.replace(tzinfo=timezone.utc) > end
                ):
                    continue
                try:
                    yield _pr_to_doc(cast(PullRequest, pr))
                except Exception as exc:
                    logger.exception("PR conversion error: %s", exc)
            if produced > 0 and not done and cp.cursor_url is None:
                return cp
            cp.stage = Stage.ISSUES
            cp.reset()
            if cp.cursor_url:
                return cp

        # 5️⃣ Issue stage
        if self.include_issues and cp.stage == Stage.ISSUES:
            issue_batch = _get_batch_rate_limited(
                lambda: repo.get_issues(
                    state=self.state_filter, sort="updated", direction="desc"
                ),
                cp.curr_page,
                cp.cursor_url,
                cp.num_retrieved,
                cursor_cb,
                self.github_client,
            )
            cp.curr_page += 1
            done = False
            produced = 0
            for issue in issue_batch:
                produced += 1
                issue = cast(Issue, issue)
                if start and issue.updated_at.replace(tzinfo=timezone.utc) < start:
                    done = True
                    break
                if end and issue.updated_at.replace(tzinfo=timezone.utc) > end:
                    continue
                if issue.pull_request:
                    continue
                try:
                    yield _issue_to_doc(issue)
                except Exception as exc:
                    logger.exception("Issue conversion error: %s", exc)
            if produced > 0 and not done and cp.cursor_url is None:
                return cp
            cp.stage = Stage.PRS  # next repo
            cp.reset()

        # 6️⃣ next repo or done
        cp.has_more = bool(cp.cached_repo_ids)
        if cp.cached_repo_ids:
            next_id = cp.cached_repo_ids.pop()
            nxt = self.github_client.get_repo(next_id)
            cp.cached_repo = SerializedRepository(
                id=next_id, headers=nxt.raw_headers, raw_data=nxt.raw_data
            )
            cp.stage = Stage.CODE if self.include_code else Stage.PRS
            cp.reset()
        return cp

    # ----------------------- abstract interface -----------------------
    @override
    def load_from_checkpoint(
        self,
        start: SecondsSinceUnixEpoch,
        end: SecondsSinceUnixEpoch,
        checkpoint: GithubCheckpoint,
    ) -> CheckpointOutput[GithubCheckpoint]:
        start_dt = datetime.fromtimestamp(start, tz=timezone.utc) - timedelta(hours=3)
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        if start_dt < epoch:
            start_dt = epoch
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc) + ONE_DAY
        return self._fetch_from_github(checkpoint, start=start_dt, end=end_dt)

    def validate_checkpoint_json(self, checkpoint_json: str) -> GithubCheckpoint:  # noqa: D401
        return GithubCheckpoint.model_validate_json(checkpoint_json)

    def build_dummy_checkpoint(self) -> GithubCheckpoint:  # noqa: D401
        init = Stage.CODE if self.include_code else Stage.PRS
        return GithubCheckpoint(stage=init, curr_page=0, has_more=True)

    # ----------------------- settings validation (unchanged) -----------------------
    def validate_connector_settings(self) -> None:  # noqa: C901
        if self.github_client is None:
            raise ConnectorMissingCredentialError("GitHub credentials not loaded.")

        if not self.repo_owner:
            raise ConnectorValidationError("'repo_owner' must be provided.")

        try:
            if self.repositories:
                names = self._split_repo_names()
                if not names:
                    raise ConnectorValidationError("No valid repository names provided.")
                any_valid = False
                errors: list[str] = []
                for name in names:
                    try:
                        self.github_client.get_repo(f"{self.repo_owner}/{name}").get_contents("")  # type: ignore[attr-defined]
                        any_valid = True
                        break
                    except GithubException as exc:
                        errors.append(f"{name}: {exc.data.get('message', str(exc))}")
                if not any_valid:
                    raise ConnectorValidationError("; ".join(errors))
            else:
                try:
                    org = self.github_client.get_organization(self.repo_owner)  # type: ignore[attr-defined]
                    if org.get_repos().totalCount == 0:
                        raise ConnectorValidationError(
                            f"Organization '{self.repo_owner}' has no accessible repositories."
                        )
                except GithubException:
                    user = self.github_client.get_user(self.repo_owner)  # type: ignore[attr-defined]
                    if user.get_repos().totalCount == 0:
                        raise ConnectorValidationError(
                            f"User '{self.repo_owner}' has no accessible repositories."
                        )
        except RateLimitExceededException:
            raise UnexpectedValidationError(
                "GitHub rate limit exceeded during validation – try again later."
            )
        except GithubException as exc:
            if exc.status == 401:
                raise CredentialExpiredError("Invalid/expired GitHub token (401).")
            if exc.status == 403:
                raise InsufficientPermissionsError(
                    "Token lacks permission to access repository (403)."
                )
            if exc.status == 404:
                raise ConnectorValidationError(
                    "Requested repository/user/org not found (404)."
                )
            raise ConnectorValidationError(
                f"Unexpected GitHub error {exc.status}: {exc.data}"
            )


# ---------------------------------------------------------------------------
# Stand-alone CLI harness (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import os
    from onyx.connectors.connector_runner import ConnectorRunner

    conn = GithubConnector(
        repo_owner=os.environ["REPO_OWNER"],
        repositories=os.environ.get("REPOSITORIES"),
        include_code=True,
        include_prs=True,
        include_issues=False,
    )
    conn.load_credentials({"github_access_token": os.environ["ACCESS_TOKEN_GITHUB"]})

    runner: ConnectorRunner[GithubCheckpoint] = ConnectorRunner(
        conn,
        batch_size=10,
        time_range=(
            datetime.fromtimestamp(0, tz=timezone.utc),
            datetime.now(tz=timezone.utc),
        ),
    )

    ckpt = conn.build_dummy_checkpoint()
    while ckpt.has_more:
        for docs, failure, ckpt in runner.run(ckpt):
            if docs:
                print("batch", len(docs))
            if failure:
                print("failure:", failure.failure_message)