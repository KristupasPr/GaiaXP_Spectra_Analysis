"""
Gaia AIP TAP and Simple Join Service helpers.

This module provides functions to download Gaia DR3 data by source IDs using:
- TAP (/sync and /async) for queries
- Simple Join Service (SJS) for joining async job results with another table

It supports:
- chunked downloads for large ID lists
- UWS job polling
- VOTable parsing into pandas DataFrames
"""

from __future__ import annotations

import io
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union, List

import pandas as pd
import pyvo as vo
import requests
from astropy.io.votable import parse_single_table

TAP_URL = "https://gaia.aip.de/tap"
SJS_URL = "https://gaia.aip.de/uws/simple-join-service"

_UWS_DONE = {"COMPLETED", "ERROR", "ABORTED"}


def iter_chunks(items: Sequence[int], chunk_size: int) -> Iterator[List[int]]:
    """Yield list chunks from a sequence."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    for i in range(0, len(items), chunk_size):
        yield list(items[i : i + chunk_size])


def create_session(token: Optional[str] = None) -> requests.Session:
    """Create authenticated session for AIP services.

    Token rules:
    - If token is None, reads GAIA_AIP_TOKEN from environment.
    - If token does not start with 'Token ' or 'Bearer ', prefixes with 'Token '.
    """
    token = (token or os.getenv("GAIA_AIP_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Missing GAIA_AIP_TOKEN (env) or token argument.")

    if not (token.startswith("Token ") or token.startswith("Bearer ")):
        token = f"Token {token}"

    session = requests.Session()
    session.headers.update({"Authorization": token})
    return session


def download_by_ids(
    source_ids: Iterable[int],
    *,
    token: Optional[str] = None,
    tap_url: str = TAP_URL,
    table: str = "gaiadr3.gaia_source",
    id_column: str = "source_id",
    columns: Union[str, Sequence[str]] = "*",
    timeout_s: int = 180,
) -> pd.DataFrame:
    """Download rows from a single table by source IDs using TAP /sync.

    This is the correct path when you want rows from `gaiadr3.gaia_source`
    itself (no SJS needed), or any other single table.

    Parameters
    ----------
    source_ids:
        Iterable of Gaia source IDs.
    table:
        Table to query, e.g. 'gaiadr3.gaia_source'.
    columns:
        '*' or list/tuple of column names.
    """
    ids = [int(x) for x in source_ids]
    if not ids:
        raise ValueError("source_ids is empty.")

    session = create_session(token)

    if isinstance(columns, str):
        select_cols = columns
    else:
        select_cols = ", ".join(columns)

    id_list_sql = ",".join(map(str, ids))
    query = f"""
    SELECT {select_cols}
    FROM {table}
    WHERE {id_column} IN ({id_list_sql})
    """

    table_out = tap_sync(session, query, tap_url=tap_url, timeout_s=timeout_s)
    return table_out.to_pandas()


def download_join_by_ids(
    source_ids: Iterable[int],
    join_table: str,
    *,
    token: Optional[str] = None,
    tap_url: str = TAP_URL,
    sjs_url: str = SJS_URL,
    base_table: str = "gaiadr3.gaia_source",
    id_column: str = "source_id",
    response_format: str = "votable",
    data_structure: str = "COMBINED",
    out_dir: Union[str, Path] = "out_gaia_downloads",
    tap_timeout_s: int = 180,
    sjs_timeout_s: int = 240,
) -> pd.DataFrame:
    """Download joined table rows for given source_ids via TAP async + SJS.

    Behavior
    --------
    - If join_table == base_table (case-insensitive), SJS is not used.
      A direct TAP /sync query is executed to return the full rows.
    - Otherwise:
      1) Create TAP async job that outputs only the IDs.
      2) Run SJS to join that async-job output with `join_table`.
      3) Download the result (VOTable) and parse to DataFrame.
    """
    ids = [int(x) for x in source_ids]
    if not ids:
        raise ValueError("source_ids is empty.")

    session = create_session(token)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Shortcut: joining gaia_source to itself is unnecessary and may fail in SJS.
    if join_table.strip().lower() == base_table.strip().lower():
        return download_by_ids(
            ids,
            token=token,
            tap_url=tap_url,
            table=base_table,
            id_column=id_column,
            columns="*",
            timeout_s=tap_timeout_s,
        )

    id_list_sql = ",".join(map(str, ids))
    query = f"""
    SELECT {id_column}
    FROM {base_table}
    WHERE {id_column} IN ({id_list_sql})
    """

    tap_job_id = tap_async_run(
        session=session,
        tap_url=tap_url,
        query=query,
        timeout_s=tap_timeout_s,
        run_id="ids_for_sjs",
    )

    votable_path = sjs_join_and_download(
        session=session,
        sjs_url=sjs_url,
        tap_job_id=tap_job_id,
        join_table=join_table,
        column_name=id_column,
        response_format=response_format,
        data_structure=data_structure,
        out_dir=out_path,
        timeout_s=sjs_timeout_s,
    )

    return votable_to_df(votable_path)


def download_join_chunked(
    source_ids: Sequence[int],
    join_table: str,
    *,
    token: Optional[str] = None,
    chunk_size: int = 1000,
    out_dir: Union[str, Path] = "out_gaia_downloads",
    save_chunks_parquet: bool = False,
) -> pd.DataFrame:
    """Chunked wrapper around download_join_by_ids()."""
    ids = [int(x) for x in source_ids]
    if not ids:
        return pd.DataFrame()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for chunk_idx, ids_part in enumerate(iter_chunks(ids, chunk_size), start=1):
        df_part = download_join_by_ids(
            source_ids=ids_part,
            join_table=join_table,
            token=token,
            out_dir=out_path / f"chunk_{chunk_idx:04d}",
        )
        df_part["__chunk__"] = chunk_idx
        parts.append(df_part)

        if save_chunks_parquet:
            chunk_dir = out_path / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            df_part.to_parquet(chunk_dir / f"chunk_{chunk_idx:04d}.parquet", index=False)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tap_sync(
    session: requests.Session,
    query: str,
    *,
    tap_url: str = TAP_URL,
    timeout_s: int = 180,
):
    """Run ADQL query via TAP /sync and return an Astropy Table."""
    url = f"{tap_url}/sync"
    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "QUERY": query.strip().rstrip(";"),
    }

    resp = session.post(url, data=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(
            "TAP /sync failed.\n"
            f"HTTP {resp.status_code}\n"
            f"Content-Type: {resp.headers.get('Content-Type')}\n"
            f"Body (first 1000 chars):\n{resp.text[:1000]}"
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    head = (resp.text[:500] or "").lower()
    if "html" in content_type or head.lstrip().startswith("<!doctype html") or "<html" in head:
        raise RuntimeError(
            "TAP returned HTML instead of VOTable.\n"
            f"Content-Type: {resp.headers.get('Content-Type')}\n"
            f"Body (first 1000 chars):\n{resp.text[:1000]}"
        )

    return parse_single_table(BytesIO(resp.content)).to_table(use_names_over_ids=True)


def tap_async_run(
    session: requests.Session,
    *,
    tap_url: str,
    query: str,
    timeout_s: int,
    run_id: str,
) -> str:
    """Create and run TAP async job; return job id."""
    query = query.strip().rstrip(";")

    resp = session.post(
        f"{tap_url}/async",
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": query,
            "RUNID": run_id,
        },
        allow_redirects=False,
        timeout=120,
    )

    if resp.status_code not in (302, 303) or "Location" not in resp.headers:
        raise RuntimeError(
            f"TAP /async create failed. HTTP {resp.status_code}. "
            f"Body: {resp.text[:500]!r}"
        )

    job_url = resp.headers["Location"]
    if job_url.startswith("/"):
        job_url = "https://gaia.aip.de" + job_url

    job_id = job_url.rstrip("/").split("/")[-1]

    session.post(f"{job_url}/phase", data={"PHASE": "RUN"}, timeout=60).raise_for_status()

    wait_uws_phase(session, f"{job_url}/phase", timeout_s, label="TAP")
    return job_id


def sjs_join_and_download(
    session: requests.Session,
    *,
    sjs_url: str,
    tap_job_id: str,
    join_table: str,
    column_name: str,
    response_format: str,
    data_structure: str,
    out_dir: Path,
    timeout_s: int,
) -> Path:
    """Run SJS join job and download first result as a VOTable file."""
    service = vo.dal.DALService(sjs_url, session=session)

    q = service.create_query(
        job_id=tap_job_id,
        column_name=column_name,
        responseformat=response_format,
        join_table=join_table,
        data_structure=data_structure,
    )

    resp = q.submit(post=True)
    if resp.status_code != 200:
        raise RuntimeError(
            "SJS submit failed.\n"
            f"HTTP {resp.status_code}\n"
            f"Body (first 1000 chars):\n{resp.text[:1000]}"
        )

    job = vo.io.uws.parse_job(io.BytesIO(resp.content))
    base = service._baseurl
    job_url = f"{base}/{job.jobid}"

    service._session.post(
        f"{job_url}/phase",
        data={"PHASE": "RUN"},
        stream=True,
    ).raise_for_status()

    wait_uws_phase(service._session, f"{job_url}/phase", timeout_s, label="SJS")

    job_xml = service._session.get(job_url, timeout=60).content
    job2 = vo.io.uws.parse_job(io.BytesIO(job_xml))

    href, key = pick_first_result_href(job2.results)

    out_path = out_dir / f"sjs_{job2.jobid}_{key}.vot"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = service._session.get(href, timeout=300).content
    out_path.write_bytes(content)
    return out_path


def wait_uws_phase(
    session: requests.Session,
    phase_url: str,
    timeout_s: int,
    *,
    label: str,
) -> None:
    """Poll a UWS job /phase endpoint until completion."""
    t0 = time.time()
    while True:
        phase = session.get(phase_url, timeout=60).text.strip()
        if phase in _UWS_DONE:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{label} job timeout (> {timeout_s}s).")
        time.sleep(1.5)

    if phase != "COMPLETED":
        raise RuntimeError(f"{label} job ended with phase={phase}.")


def pick_first_result_href(results) -> tuple[str, str]:
    """Return (href, key) for the first result item."""
    if hasattr(results, "keys") and callable(getattr(results, "keys")):
        first_key = sorted(list(results.keys()))[0]
        return results[first_key].href, str(first_key)

    res_list = list(results)
    if not res_list:
        raise RuntimeError("SJS job has no results.")
    return res_list[0].href, "result"


def votable_to_df(path: Path) -> pd.DataFrame:
    """Parse the first VOTable in a file into a pandas DataFrame."""
    table = parse_single_table(str(path)).to_table(use_names_over_ids=True)
    return table.to_pandas()