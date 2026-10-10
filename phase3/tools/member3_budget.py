"""Conservative optimizer reservations; not a RunPod billing meter."""
import fcntl
import math
from pathlib import Path
from member3_common import canonical_sha256, read_json, write_json, validate_resource_limits
from common import utc_now


def reserve(path, mode, limits, attempt_id, config_hash, minutes):
    errors=validate_resource_limits(limits)
    if errors: raise ValueError('; '.join(errors))
    if isinstance(minutes,bool) or not isinstance(minutes,(int,float)) or not math.isfinite(minutes) or minutes<=0:
        raise ValueError('attempt minutes must be finite and positive')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            ledger=read_json(path)
            if ledger.get('ledger_sha256')!=canonical_sha256(ledger,omit_key='ledger_sha256'): raise ValueError('ledger hash mismatch')
            if ledger['mode']!=mode or ledger['resource_limits']!=limits: raise ValueError('ledger scope/limits mismatch; changing limits needs new authorization')
        else:
            ledger=dict(schema_version=1,kind='optimizer_budget_ledger',mode=mode,resource_limits=limits,attempts=[])
        attempts=ledger['attempts']
        if any(a['attempt_id']==attempt_id for a in attempts): raise ValueError('attempt ID already reserved')
        if len(attempts)>=limits['max_attempts']: raise ValueError('maximum attempts exhausted')
        cost=minutes*limits['max_hourly_usd']/60
        if sum(a['reserved_usd'] for a in attempts)+cost>limits['total_budget_usd']+1e-9: raise ValueError('cumulative optimizer budget exhausted')
        if sum(a['reserved_minutes'] for a in attempts)+minutes>limits['max_wall_clock_minutes']+1e-9: raise ValueError('cumulative optimizer time exhausted')
        attempts.append(dict(attempt_id=attempt_id,config_sha256=config_hash,reserved_minutes=minutes,reserved_usd=cost,reserved_at_utc=utc_now()))
        ledger['ledger_sha256']=canonical_sha256(ledger,omit_key='ledger_sha256');write_json(path,ledger)
        # Reservations are never automatically refunded, including failed attempts.
        return dict(seconds=minutes*60,ledger_sha256=ledger['ledger_sha256'],reserved_usd=cost)
