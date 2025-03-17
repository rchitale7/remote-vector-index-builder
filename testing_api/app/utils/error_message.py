# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
from typing import Any

def get_field_path(location: tuple[Any, ...]) -> str:
    """Convert location tuple to a readable field path."""
    field_path = []
    for loc in location:
        if isinstance(loc, int):
            field_path.append(f"[{loc}]")
        elif isinstance(loc, str):
            if field_path:
                field_path.append("." + loc)
            else:
                field_path.append(loc)
    return "".join(field_path)