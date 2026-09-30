#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from member3_common import canonical_sha256, read_json
p=argparse.ArgumentParser(); p.add_argument("config",type=Path); args=p.parse_args()
print(canonical_sha256(read_json(args.config)))
