import os
import sys

from astroflow import create_session, nearest_source_with_quality


def main():
    """
    Usage:
        py main.py "18 43 53.2248 +43 52 32.0597"
        py main.py "280.97177 43.87557"
        py main.py 280.97177 43.87557

    Token:
        - Set environment variable GAIA_AIP_TOKEN
        - Or hardcode it below (not recommended)
    """
    token = os.getenv("GAIA_AIP_TOKEN")
    if not token:
        print("Missing GAIA_AIP_TOKEN environment variable.")
        sys.exit(1)

    session = create_session(token)

    args = sys.argv[1:]
    if not args:
        print(
            'Provide coordinates, e.g.:\n'
            '  "18 43 53.22 +43 52 32.05"\n'
            '  "280.97 43.87"\n'
            '  280.97 43.87'
        )
        sys.exit(1)

    if len(args) == 1:
        coords = args[0]
        result = nearest_source_with_quality(
            session=session,
            coords=coords,
            dec=None,
            radius_arcsec=2.0,
            debug=True,
        )
    else:
        ra = float(args[0])
        dec = float(args[1])
        result = nearest_source_with_quality(
            session=session,
            coords=ra,
            dec=dec,
            radius_arcsec=2.0,
            debug=True,
        )

    print(result)


if __name__ == "__main__":
    main()