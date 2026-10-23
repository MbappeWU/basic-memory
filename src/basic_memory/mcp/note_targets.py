"""Route a bm_* tool's target into its project and resolve notes exactly (#1829).

Every bm_* tool starts the same way: classify the `project` argument, let a
'<project>/path' prefix on the target choose or confirm the project, and open a
client routed to it. This module is that one path. It reuses the POSIX tools'
routing (`resolve_project_path_route`), so a bm_* tool accepts exactly the
project-qualified paths other tools return.

Resolution is exact. A target that names no note is a NOT_FOUND refusal that
points to `bm_search`; there is no fuzzy fallback, because a tool that edits or
deletes must never act on a near match.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError
from httpx import AsyncClient, HTTPStatusError

from basic_memory.mcp.async_client import local_project_path
from basic_memory.mcp.clients import KnowledgeClient
from basic_memory.mcp.project_context import (
    ProjectPathRoute,
    get_project_client,
    resolve_project_path_route,
)
from basic_memory.mcp.refusals import refusal
from basic_memory.mcp.tool_params import OutputFormat, parse_project, routing_arguments
from basic_memory.schemas.memory import memory_url_path
from basic_memory.schemas.project_info import ProjectItem
from basic_memory.schemas.v2.entity import EntityResolveResponse
from basic_memory.utils import generate_permalink, validate_project_path


@dataclass(frozen=True, slots=True)
class ProjectTarget:
    """A target routed into its project, with a client for that project.

    `route.path` is the project-relative target: the input with any project prefix
    removed, "" when the input named the project itself.
    """

    client: AsyncClient
    project: ProjectItem
    route: ProjectPathRoute


@asynccontextmanager
async def open_project_target(
    target: str,
    project: str | None,
    context: Context | None,
    *,
    output_format: OutputFormat = "text",
) -> AsyncIterator[ProjectTarget]:
    """Route `target` and yield a client for the project it lives in."""
    project_name, project_id = routing_arguments(parse_project(project))
    route = await resolve_project_path_route(
        target, project=project_name, project_id=project_id, context=context
    )
    async with get_project_client(route.project, context=context, project_id=route.project_id) as (
        client,
        active_project,
    ):
        # Trigger: the target climbs out of the project ('../x', an absolute path, '~').
        # Why: every bm_* tool, including the ones that write and delete, must stay
        #      inside its project. Remote projects are checked lexically; only a local
        #      project's root is resolved against the filesystem (#1807).
        # Outcome: an OUTSIDE_PROJECT refusal before any read or write.
        # A leading '/' anchors at the project root ('/' or '/specs'), never the
        # filesystem root, so it is dropped before the check.
        project_relative = memory_url_path(route.path).lstrip("/")
        if not validate_project_path(project_relative, local_project_path(client, active_project)):
            raise refusal(
                "OUTSIDE_PROJECT",
                f"'{target}' is outside the project. Paths must stay within the project.",
                try_instead="a path relative to the project root, such as 'notes/plan.md'",
                output_format=output_format,
            )
        yield ProjectTarget(client=client, project=active_project, route=route)


def _is_not_found(error: ToolError) -> bool:
    """True when a typed-client failure wraps an HTTP 404 from the API."""
    cause = error.__cause__
    return isinstance(cause, HTTPStatusError) and cause.response.status_code == 404


async def resolve_note(
    target: ProjectTarget,
    identifier: str,
    *,
    output_format: OutputFormat = "text",
    try_instead: str | None = None,
) -> EntityResolveResponse:
    """Resolve the routed target to its note, exactly.

    The response carries the note's external id, file path and title, which is
    what a tool needs to decide how to read or change it.

    `identifier` is the caller's original spelling, used in the refusal so the
    model sees what it asked for. `try_instead` replaces the default bm_search hint
    on a NOT_FOUND refusal (bm_edit points to bm_write).
    """
    # Trigger: the input was a bare project name ('research' or 'research/').
    # Why: routing consumed the whole identifier; there is no note left to resolve.
    # Outcome: refuse with the call that lists the project instead.
    if target.route.stripped and not target.route.path:
        raise refusal(
            "INVALID_ARGUMENTS",
            f"'{identifier}' names a project, not a note.",
            try_instead=f"bm_list(path='{identifier.rstrip('/')}/')",
            output_format=output_format,
        )

    knowledge_client = KnowledgeClient(target.client, target.project.external_id)
    try:
        return await knowledge_client.resolve_entity_response(target.route.path, strict=True)
    except ToolError as error:
        if not _is_not_found(error):
            raise
        raise refusal(
            "NOT_FOUND",
            f"No note matches '{identifier}' in project '{target.project.name}'.",
            try_instead=try_instead or f"bm_search(query='{identifier}')",
            output_format=output_format,
        ) from error


# --- Round-trip coherence ---
# A path a routed verb returns must be a path the resolver accepts. When a call
# addresses its project in the path ('ls research/notes'), the project prefix is
# stripped before the project-scoped API sees it, so that API answers in
# project-relative paths — '/notes'. Handing those back unchanged breaks the
# navigation loop the mount model promises: feeding '/notes' into `ls` refuses as
# unqualified, or worse, opens a *different* project that happens to be mounted
# as 'notes'.
#
# One rule decides *whether* and *with what* (_route_prefix); each response
# schema says *where*. That split is deliberate. Rewriting by key name anywhere
# in the payload also rewrote a note's own frontmatter when the author happened
# to use a `file_path:` key, so a routed `cat` returned frontmatter that
# disagreed with both its own `content` and the stored file. Transport metadata
# and note content are different things that can spell a key the same way, and
# only position tells them apart.


def _route_prefix(route: ProjectPathRoute) -> str | None:
    """The prefix a routed response must re-attach, or None if nothing was stripped.

    None means the caller addressed the project some other way (an explicit
    param, or a single-project session), so the project-relative paths it gets
    back are already the ones it can feed back.
    """
    if not route.stripped or route.project is None:
        return None
    # The permalink form is the spelling `ls "/"` advertises and the resolver
    # normalizes to, so it round-trips for display names too ('My Research').
    return generate_permalink(route.project)


def _requalified_path(value: str, prefix: str) -> str:
    """Re-attach a stripped project prefix, preserving the field's slash shape."""
    if value.startswith("/"):
        return f"/{prefix}" if value == "/" else f"/{prefix}{value}"
    return f"{prefix}/{value}" if value else prefix


def qualify_note_paths(payload: dict[str, Any], route: ProjectPathRoute) -> dict[str, Any]:
    """Re-qualify a note payload's transport path.

    ``file_path`` is the note's address and is re-qualified. ``frontmatter`` is
    the note's own YAML — canonical content that must come back byte for byte,
    even when it carries keys named like transport fields — and ``permalink`` is
    an identity with its own canonical form; neither is touched.
    """
    prefix = _route_prefix(route)
    if prefix is None:
        return payload
    return {**payload, "file_path": _requalified_path(payload["file_path"], prefix)}


def _requalified_directory_node(node: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Re-qualify one DirectoryNode's addressing fields, and its children."""
    requalified = dict(node)
    directory_path = node.get("directory_path")
    if isinstance(directory_path, str):
        requalified["directory_path"] = _requalified_path(directory_path, prefix)
    # file_path is Optional on DirectoryNode: directory rows carry no file.
    file_path = node.get("file_path")
    if isinstance(file_path, str):
        requalified["file_path"] = _requalified_path(file_path, prefix)
    children = node.get("children")
    if children:
        requalified["children"] = [_requalified_directory_node(child, prefix) for child in children]
    return requalified


def qualify_listing_paths(payload: dict[str, Any], route: ProjectPathRoute) -> dict[str, Any]:
    """Re-qualify a directory listing's node addressing fields."""
    prefix = _route_prefix(route)
    if prefix is None:
        return payload
    return {
        **payload,
        "nodes": [_requalified_directory_node(node, prefix) for node in payload["nodes"]],
    }


def qualify_search_paths(payload: dict[str, Any], route: ProjectPathRoute) -> dict[str, Any]:
    """Re-qualify a search response's per-hit transport paths.

    The third response shape a routed verb can answer with. `find --meta` returns
    search hits rather than listing nodes, and the rule is per response shape,
    not per verb: the metadata arm shipped without it and handed back
    project-relative paths that `cat` then refused — or, worse, opened in a
    different project mounted under that name (#1435).

    ``file_path`` is the hit's address, the same field the listing nodes carry;
    ``permalink`` is an identity with its own canonical form and is left alone,
    exactly as in ``qualify_note_paths``.
    """
    prefix = _route_prefix(route)
    if prefix is None:
        return payload
    return {
        **payload,
        "results": [
            {**row, "file_path": _requalified_path(row["file_path"], prefix)}
            for row in payload["results"]
        ],
    }
