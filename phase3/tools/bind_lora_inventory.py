#!/usr/bin/env python3
"""Create new pilot config from real inventory and separately authorized limits."""
import argparse
from pathlib import Path
from member3_common import read_json, write_json, canonical_sha256, validate_resource_limits
from validate_lora_config import runtime_checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('config','inventory','resource-limits','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('new config path required')
    c=read_json(a.config);inventory=read_json(a.inventory);limits=read_json(a.resource_limits)
    if inventory.get('report_sha256')!=canonical_sha256(inventory,omit_key='report_sha256'):raise ValueError('inventory report hash mismatch')
    errors=validate_resource_limits(limits)
    if errors:raise ValueError('; '.join(errors))
    if inventory.get('model_revision')!=c['runtime']['expected_model_revision'] or inventory.get('upstream_commit')!=c['runtime']['expected_upstream_commit']:raise ValueError('inventory source identity mismatch')
    c['inventory_state']='inventory_verified'
    c['lora'].update(target_modules=inventory['suffix_rules'],target_source='runtime_inventory',inventory_sha256=inventory['report_sha256'],matched_module_count=inventory['matched_module_count'])
    c['resource_limits']=limits
    errors=runtime_checks(c,a.inventory)
    if errors:raise ValueError('; '.join(errors))
    c['config_sha256']=canonical_sha256(c,omit_key='config_sha256');write_json(a.output,c)
    print(a.output)

if __name__=='__main__':main()
