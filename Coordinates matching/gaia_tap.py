"""
Gaia AIP TAP helpers
This code provides functions to query the Gaia AIP TAP 
service for the nearest Gaia DR3 source to given coordinates, 
with flexible input formats and a simple match quality assessment. 
It uses the requests library for HTTP and astropy for coordinate parsing and VOTable handling.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple, Union, TypedDict

import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table

TAP_URL = "https://gaia.aip.de/tap"

CoordInput = Union[
    Tuple[float, float],  # (ra_deg, dec_deg)
    str,  # "ra dec" (deg) OR "h m s d m s" / "hh:mm:ss +dd:mm:ss"
]


class MatchResult(TypedDict):
    source_id: int
    separation_arcsec: float
    matched_ra: float
    matched_dec: float


class MatchResultWithQuality(TypedDict):
    source_id: Optional[int]
    separation_arcsec: Optional[float]
    matched_ra: Optional[float]
    matched_dec: Optional[float]
    quality: str


def create_session(token: Optional[str] = None) -> requests.Session:
    """Create an authenticated session for gaia.aip.de TAP.

    Parameters
    ----------
    token:
        - "Token <hex...>"
        - "<hex...>" (will be converted to "Token <hex...>")
        - None -> reads env var GAIA_AIP_TOKEN

    Returns
    -------
    requests.Session
    """
    token = token or os.getenv("GAIA_AIP_TOKEN")
    if not token:
        raise RuntimeError("Missing GAIA_AIP_TOKEN (env) or token argument.")

    if not token.startswith("Token "):
        token = f"Token {token}"

    session = requests.Session()
    session.headers.update({"Authorization": token})
    return session


def tap_sync(session: requests.Session, query: str, timeout: int = 120):
    """Run an ADQL query via TAP /sync and return an Astropy Table."""
    url = f"{TAP_URL}/sync"
    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "QUERY": query.strip().rstrip(";"),
    }

    resp = session.post(url, data=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            "TAP request failed.\n"
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

    return parse_single_table(BytesIO(resp.content)).to_table()


def _source_id_from_row(row, colnames) -> int:
    """Extract source_id-like column robustly, preserving int precision."""
    name_map = {name.lower(): name for name in colnames}
    for key in ("source_id", "src_id", "datalinkid"):
        if key in name_map:
            return int(row[name_map[key]])
    raise KeyError(f"No source_id-like column found. Columns: {colnames}")


def parse_coords(coords: CoordInput, dec: Optional[float] = None) -> Tuple[float, float]:
    """Parse coordinates to degrees.

    Accepts:
    - parse_coords(ra_deg, dec_deg) via parse_coords(ra, dec=<...>)
    - parse_coords((ra_deg, dec_deg))
    - parse_coords("ra dec") where both are degrees
    - parse_coords("h m s +d m s") or "hh:mm:ss +dd:mm:ss"

    Returns
    -------
    (ra_deg, dec_deg)
    """
    if dec is not None:
        return float(coords), float(dec)

    if isinstance(coords, (tuple, list)) and len(coords) == 2:
        return float(coords[0]), float(coords[1])

    if not isinstance(coords, str):
        raise TypeError("coords must be (ra, dec) floats, tuple/list, or a string.")

    s = coords.strip()
    parts = s.split()

    if len(parts) == 2:
        return float(parts[0]), float(parts[1])

    sky = SkyCoord(s, unit=(u.hourangle, u.deg), frame="icrs")
    return float(sky.ra.deg), float(sky.dec.deg)


def nearest_source(
    session: requests.Session,
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 2.0,
    debug: bool = False,
) -> Optional[MatchResult]:
    """Find nearest Gaia DR3 source to (ra, dec) within radius.

    Returns
    -------
    MatchResult or None
    """
    radius_deg = radius_arcsec / 3600.0
    ra_s = f"{ra_deg:.17f}"
    dec_s = f"{dec_deg:.17f}"

    query = f"""
    SELECT TOP 1
        source_id,
        ra AS ra_deg,
        dec AS dec_deg,
        DISTANCE(
            POINT('ICRS', ra, dec),
            POINT('ICRS', {ra_s}, {dec_s})
        ) AS dist_deg
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_s}, {dec_s}, {radius_deg})
    ) = 1
    ORDER BY dist_deg ASC
    """

    table = tap_sync(session, query)
    if len(table) == 0:
        return None

    if debug:
        print("Returned columns:", table.colnames)

    row = table[0]
    source_id = _source_id_from_row(row, table.colnames)

    dist_deg = float(row["dist_deg"])
    return {
        "source_id": source_id,
        "separation_arcsec": dist_deg * 3600.0,
        "matched_ra": float(row["ra_deg"]),
        "matched_dec": float(row["dec_deg"]),
    }


def nearest_source_from(
    session: requests.Session,
    coords: CoordInput,
    dec: Optional[float] = None,
    radius_arcsec: float = 2.0,
    debug: bool = False,
) -> Optional[MatchResult]:
    """Wrapper that accepts flexible coordinate input."""
    ra_deg, dec_deg = parse_coords(coords, dec=dec)
    return nearest_source(
        session=session,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=radius_arcsec,
        debug=debug,
    )


def match_quality(sep_arcsec: Optional[float], radius_arcsec: float) -> str:
    """Simple match quality label based on separation."""
    if sep_arcsec is None:
        return "no_match"
    if sep_arcsec <= 0.3:
        return "good"
    if sep_arcsec <= 1.0:
        return "ok"
    if sep_arcsec <= radius_arcsec:
        return "suspicious"
    return "bad"


def nearest_source_with_quality(
    session: requests.Session,
    coords: CoordInput,
    dec: Optional[float] = None,
    radius_arcsec: float = 2.0,
    debug: bool = False,
) -> MatchResultWithQuality:
    """Nearest-source wrapper that always returns a dict with a quality flag."""
    res = nearest_source_from(
        session=session,
        coords=coords,
        dec=dec,
        radius_arcsec=radius_arcsec,
        debug=debug,
    )

    if res is None:
        return {
            "source_id": None,
            "separation_arcsec": None,
            "matched_ra": None,
            "matched_dec": None,
            "quality": "no_match",
        }

    return {
        "source_id": res["source_id"],
        "separation_arcsec": res["separation_arcsec"],
        "matched_ra": res["matched_ra"],
        "matched_dec": res["matched_dec"],
        "quality": match_quality(res["separation_arcsec"], radius_arcsec),
    }