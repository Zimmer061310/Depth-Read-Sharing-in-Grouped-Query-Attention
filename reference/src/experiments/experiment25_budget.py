"""Provider-neutral, fail-closed cost guard. No provider credentials or adapter.

The independent watcher runs OUTSIDE the rented machine. A separately tested
provider-side hard-off deadline must survive loss of this watcher too. Guest
shutdown, a dead SSH connection, or a worker's exit is never an off receipt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

from src.experiments.experiment1_partition_compatibility import atomic_write_json, sha256_file


def read(path):
    return json.loads(Path(path).read_text())


class CommandProvider:
    """Explicit out-of-band JSON adapter: argv receives request JSON on stdin.

    Contract operations: verify_failsafe, inspect, stop. Inspect returns actual
    all-in accrued CNY and cumulative billed GPU hours, including earlier phases.
    Never use a guest SSH command for this adapter. Adapter deployment/testing
    and provider authentication are future launch prerequisites, not fabricated
    by this module. No shell evaluation; a hung request has a bounded timeout.
    """
    def __init__(self, argv):
        if not argv or not all(isinstance(a,str) and a for a in argv):
            raise ValueError("explicit provider adapter argv required")
        self.argv = argv

    def request(self, operation, lock):
        result = subprocess.run(self.argv, input=json.dumps({"operation":operation,
            "instance_id":lock["instance_id"], "hard_off_epoch":lock["hard_off_epoch"],
            "failsafe_id":lock["failsafe_id"],
            **({"checkpoint_receipt":lock["checkpoint_receipt"]} if "checkpoint_receipt" in lock else {})}), text=True, capture_output=True,
            timeout=15, check=True)
        response = json.loads(result.stdout)
        if response.get("instance_id") != lock["instance_id"]:
            raise ValueError("provider receipt instance mismatch")
        return response


def validate_lock(lock, provider, now, *, require_forecast_fit=True):
    numbers = ("billing_start_epoch","hard_off_epoch","work_cutoff_epoch",
               "maximum_gpu_hours","maximum_spend_cny","gpu_hour_rate_ceiling_cny",
               "non_gpu_cost_ceiling_cny","forecast_remaining_hours","shutdown_reserve_seconds",
               "retention_until_epoch")
    if any(not isinstance(lock.get(k),(float,int)) or isinstance(lock[k],bool)
           or not math.isfinite(lock[k]) for k in numbers):
        raise ValueError("missing finite, actual budget/price/deadline lock")
    if (lock.get("separate_execution_authorized") is not True
        or type(lock.get("billed_gpu_count")) is not int or lock["billed_gpu_count"] != 1 or not lock.get("instance_id")
        or not lock.get("failsafe_id") or not lock.get("watcher_host_id")
        or lock.get("watcher_host_id") == lock.get("rented_host_id")
        or not lock.get("rented_host_id") or lock.get("release_disk") is not False
        or lock.get("provider_failsafe_tested_before_billing") is not True
        or lock.get("retention_expiry_action") not in ("download_before_expiry","explicitly_renew_storage")
        or not lock.get("price_receipt_sha256") or len(lock["price_receipt_sha256"]) != 64):
        raise ValueError("unverified authorization/provider independence/price/retention policy")
    start, hard, cutoff = (lock[k] for k in ("billing_start_epoch","hard_off_epoch","work_cutoff_epoch"))
    reserve = lock["shutdown_reserve_seconds"]
    hours = (hard-start)/3600
    if (not start < cutoff < hard < lock["retention_until_epoch"] or now < start
        or reserve < 1800 or hard-cutoff < reserve
        or not 0 < lock["maximum_gpu_hours"] <= 22
        or hours > lock["maximum_gpu_hours"]
        or not 0 < lock["maximum_spend_cny"] <= 63.36
        or lock["gpu_hour_rate_ceiling_cny"] <= 0 or lock["non_gpu_cost_ceiling_cny"] < 0
        or hours*lock["gpu_hour_rate_ceiling_cny"]+lock["non_gpu_cost_ceiling_cny"] > lock["maximum_spend_cny"]
        or lock["forecast_remaining_hours"] <= 0
        or (require_forecast_fit and now+3600*lock["forecast_remaining_hours"] > cutoff)):
        raise ValueError("forecast plus reserve cannot fit actual all-in cost/time ceiling")
    receipt = provider.request("verify_failsafe",lock)
    if (receipt.get("armed") is not True or receipt.get("tested") is not True
        or receipt.get("survives_watcher_and_guest_loss") is not True
        or receipt.get("hard_off_epoch") != hard
        or receipt.get("failsafe_id") != lock["failsafe_id"]
        or receipt.get("release_disk") is not False
        or receipt.get("price_receipt_sha256") != lock["price_receipt_sha256"]
        or receipt.get("gpu_hour_rate_ceiling_cny") != lock["gpu_hour_rate_ceiling_cny"]
        or receipt.get("non_gpu_cost_ceiling_cny") != lock["non_gpu_cost_ceiling_cny"]
        or receipt.get("retention_until_epoch") != lock["retention_until_epoch"]):
        raise ValueError("independent provider hard-stop is not armed and tested")
    return receipt


def inspect_usage(provider, lock):
    receipt = provider.request("inspect",lock)
    for key in ("all_in_incremental_spend_cny","cumulative_billed_gpu_hours"):
        if (not isinstance(receipt.get(key),(int,float)) or isinstance(receipt[key],bool) or not math.isfinite(receipt[key])
            or receipt[key] < 0):
            raise ValueError("provider billing unavailable; cannot assume zero")
    if (receipt.get("status") not in ("running","off")
        or type(receipt.get("billed_gpu_count")) is not int or receipt["billed_gpu_count"] != 1):
        raise ValueError("unknown provider state or changed billed card count")
    if receipt.get("work_failed") not in (True,False) or not isinstance(receipt.get("work_failed"),bool):
        raise ValueError("out-of-band worker/failure status unavailable")
    return receipt


def stop_and_verify(provider, lock):
    provider.request("stop",lock)
    receipt = inspect_usage(provider,lock)
    if receipt["status"] != "off":
        raise RuntimeError("provider has not verified OFF; retry stop, never release disk")
    return receipt


class Watchdog:
    def __init__(self, lock_path, provider, heartbeat, event_dir, clock=time.time):
        self.lock_path, self.provider, self.clock = Path(lock_path), provider, clock
        self.lock = read(lock_path)
        self.heartbeat, self.events = Path(heartbeat), Path(event_dir)
        self.events.mkdir(parents=True,exist_ok=True)
        validate_lock(self.lock,provider,clock(),require_forecast_fit=False)
        previous=sorted(self.events.glob("????????.json"))
        if previous and read(previous[-1]).get("lock_sha256")!=sha256_file(self.lock_path):
            raise ValueError("watcher recovery cannot change the budget lock")
        self.sequence = int(previous[-1].stem)+1 if previous else 0
        prior_usage=read(previous[-1]).get("provider",{}) if previous else {}
        self.last_usage = prior_usage if all(k in prior_usage for k in
            ("all_in_incremental_spend_cny","cumulative_billed_gpu_hours")) else None

    def tick(self, durable_finish=False, external_failure=False):
        now, reason = self.clock(), None
        try:
            usage = inspect_usage(self.provider,self.lock)
            if self.last_usage and any(usage[k] < self.last_usage[k] for k in
                ("all_in_incremental_spend_cny","cumulative_billed_gpu_hours")):
                raise ValueError("provider cumulative billing moved backwards")
            self.last_usage = usage
            if usage["work_failed"] or external_failure: reason = "worker_or_transfer_failure"
            elif durable_finish: reason = "durable_gpu_work_complete"
            elif now >= self.lock["hard_off_epoch"]: reason = "hard_deadline"
            elif usage["all_in_incremental_spend_cny"] >= self.lock["maximum_spend_cny"]: reason = "spend_ceiling"
            elif usage["cumulative_billed_gpu_hours"] >= self.lock["maximum_gpu_hours"]: reason = "gpu_hour_ceiling"
        except Exception as error:
            usage, reason = {"billing_error":type(error).__name__}, "billing_unknown"
        if reason:
            # Upload/worker failures cannot veto this operation. Keep trying on
            # the next watcher tick if OFF is not yet verified.
            try:
                usage = stop_and_verify(self.provider,self.lock)
            except Exception as error:
                usage = {**usage,"stop_error":type(error).__name__,"status":"unverified"}
        pulse = {"time_epoch":now,"lock_sha256":sha256_file(self.lock_path),
            "sequence":self.sequence,"work_allowed":reason is None and now < self.lock["work_cutoff_epoch"]
                and usage.get("status")=="running", "stop_reason":reason,"provider":usage}
        event = self.events/f"{self.sequence:08d}.json"
        if event.exists(): raise FileExistsError("watcher event collision; preserve previous attempt")
        atomic_write_json(event,pulse)
        atomic_write_json(self.heartbeat,pulse)
        self.sequence += 1
        return pulse


def require_work_allowed(lock_path, heartbeat, now=None):
    now = time.time() if now is None else now
    lock, pulse = read(lock_path), read(heartbeat)
    if lock.get("shutdown_mode") == "autodl-console-timer-guest-worker-v1":
        from src.experiments.experiment25_timer import MODE, validate_timer_lock
        validate_timer_lock(lock, now, require_forecast_fit=False)
        if (pulse.get("lock_sha256") != sha256_file(lock_path)
            or pulse.get("shutdown_mode") != MODE or pulse.get("work_allowed") is not True
            or not 0 <= now-pulse.get("time_epoch", 0) <= 60
            or pulse.get("provider_billing_measured") is not False):
            raise RuntimeError("timer watchdog stale/unauthorized/cutoff; no new GPU work")
        return pulse
    if (pulse.get("lock_sha256") != sha256_file(lock_path)
        or pulse.get("work_allowed") is not True
        or not 0 <= now-pulse.get("time_epoch",0) <= 60
        or now >= lock["work_cutoff_epoch"]
        or pulse.get("provider",{}).get("status") != "running"):
        raise RuntimeError("budget cutoff/stale independent watcher: preserve atomic state; no new GPU work")
    return pulse
