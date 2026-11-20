"""Explicitly authorized regular-AutoDL timer deviation, not a provider API.

The provider console deadline is the independent backstop. Guest pulses and
elapsed-price upper bounds are deliberately NOT provider billing/OFF receipts.
"""
from __future__ import annotations

import math
import os
import platform
import signal
from pathlib import Path
import subprocess
import time

from src.experiments.experiment1_partition_compatibility import atomic_write_json, sha256_file
from src.experiments.experiment25_workflow import read, write_new

MODE = "autodl-console-timer-guest-worker-v1"


def _pidfd_syscall(number, *arguments):
    """Older AutoDL Python lacks wrappers even when its Linux kernel supports pidfds."""
    if platform.system()!="Linux" or platform.machine() not in ("x86_64","aarch64"):
        raise RuntimeError("Linux pidfd shutdown support required")
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    result=libc.syscall(ctypes.c_long(number),*arguments)
    if result==-1:
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))
    return result


def _open_pidfd(pid, flags):
    return _pidfd_syscall(434,pid,flags)


def _send_pidfd_signal(fd, sig, info, flags):
    if info is not None:raise ValueError("only default signal metadata supported")
    import ctypes
    return _pidfd_syscall(424,fd,sig,ctypes.c_void_p(),flags)


def stop_autodl_compute(*, proc_root="/proc", open_pidfd=None, send_signal=None, close_fd=None):
    """Target the verified AutoDL supervisor, never execute its vendor shell fragment.

    A pidfd prevents PID reuse between identity verification and SIGTERM.
    This is only a guest stop request; provider OFF must be observed separately.
    The vendor /usr/bin/shutdown has no shebang and also deletes unrelated Trash.
    """
    expected=(b"/bin/supervisord", b"-c", b"/init/supervisor/supervisor.ini")
    root=Path(proc_root)
    def argv(path):
        try: return tuple(path.read_bytes().rstrip(b"\0").split(b"\0"))
        except (FileNotFoundError, ProcessLookupError): return ()
    candidates=[p for p in root.iterdir() if p.name.isdecimal() and int(p.name)>1
                and int(p.name)!=os.getpid() and argv(p/"cmdline")==expected]
    if len(candidates)!=1:
        raise RuntimeError("expected exactly one verified AutoDL supervisor")
    target=candidates[0]; pid=int(target.name)
    opener=open_pidfd or getattr(os,"pidfd_open",None) or _open_pidfd
    sender=send_signal or getattr(signal,"pidfd_send_signal",None) or _send_pidfd_signal
    fd=opener(pid,0)
    try:
        if argv(target/"cmdline")!=expected:
            raise RuntimeError("AutoDL supervisor identity changed")
        sender(fd,signal.SIGTERM,None,0)
    finally:
        (close_fd or os.close)(fd)
    return {"target_pid":pid,"signal":"SIGTERM","provider_off_verified":False,"release_disk":False}


def validate_timer_lock(lock, now, *, require_forecast_fit=True):
    if (lock.get("shutdown_mode") != MODE
        or lock.get("separate_execution_authorized") is not True
        or lock.get("timer_deviation_authorized") is not True
        or lock.get("release_disk") is not False
        or type(lock.get("billed_gpu_count")) is not int or lock["billed_gpu_count"] != 1
        or not lock.get("instance_id") or not lock.get("authorization_text")
        or not lock.get("incremental_non_gpu_cost_basis")):
        raise ValueError("explicit timer deviation/authority/fee basis required")
    keys = ("billing_start_epoch", "hard_off_epoch", "work_cutoff_epoch",
            "maximum_gpu_hours", "maximum_spend_cny", "gpu_hour_rate_ceiling_cny",
            "non_gpu_cost_ceiling_cny", "forecast_remaining_hours", "shutdown_reserve_seconds",
            "retention_until_epoch")
    if any(isinstance(lock.get(k), bool) or not isinstance(lock.get(k), (int, float))
           or not math.isfinite(lock[k]) for k in keys):
        raise ValueError("finite timer budget required")
    start, cutoff, hard = (lock[k] for k in ("billing_start_epoch", "work_cutoff_epoch", "hard_off_epoch"))
    max_hours, max_spend = 22., 63.36
    extension = lock.get("budget_extension")
    if extension is not None:
        import re
        if (not isinstance(extension, dict)
            or extension.get("explicitly_authorized") is not True
            or extension.get("maximum_gpu_hours") != 24.
            or extension.get("maximum_spend_cny") != 69.12
            or not isinstance(extension.get("authorization_text"), str)
            or not extension["authorization_text"].strip()
            or not re.fullmatch(r"[0-9a-f]{64}", extension.get("previous_budget_sha256", ""))):
            raise ValueError("explicit bounded24-hour budget amendment required")
        max_hours, max_spend = 24., 69.12
    if (not start <= now < cutoff < hard < lock["retention_until_epoch"]
        or not 0 < lock["maximum_gpu_hours"] <= max_hours
        or (hard-start)/3600 > lock["maximum_gpu_hours"]
        or not 0 < lock["maximum_spend_cny"] <= max_spend
        or lock["gpu_hour_rate_ceiling_cny"] <= 0 or lock["non_gpu_cost_ceiling_cny"] < 0
        or (hard-start)/3600*lock["gpu_hour_rate_ceiling_cny"]+lock["non_gpu_cost_ceiling_cny"] > lock["maximum_spend_cny"]
        or lock["shutdown_reserve_seconds"] < 1800
        or hard-cutoff < lock["shutdown_reserve_seconds"]
        or lock["forecast_remaining_hours"] <= 0
        or (require_forecast_fit and now+3600*lock["forecast_remaining_hours"] > cutoff)):
        raise ValueError("timer forecast/reserve/cap cannot fit")
    path = Path(lock["timer_observation_path"])
    if sha256_file(path) != lock.get("timer_observation_sha256"):
        raise ValueError("timer observation changed")
    observation = read(path)
    pending = lock.get("user_deferred_timer_extension")
    deferred = (isinstance(pending, dict)
        and pending.get("explicitly_authorized") is True
        and isinstance(pending.get("authorization_text"), str)
        and bool(pending["authorization_text"].strip())
        and pending.get("observed_earlier_deadline_utc") == observation.get("provider_scheduled_shutdown_utc")
        and pending.get("requested_deadline_utc") == lock["hard_off_utc"]
        and pending.get("extension_saved") is False
        and lock.get("budget_extension", {}).get("explicitly_authorized") is True)
    if (observation.get("instance_id") != lock["instance_id"]
        or observation.get("deadline_persisted_after_completed_page_reload") is not True
        or observation.get("provider_console_showed_save_success") is not True
        or (observation.get("provider_scheduled_shutdown_utc") != lock["hard_off_utc"] and not deferred)
        or observation.get("release_instance_or_disk_requested") is not False):
        raise ValueError("actual provider console deadline/instance not verified")
    return {"shutdown_mode": MODE, "provider_console_timer_observation_sha256": sha256_file(path),
            "provider_deadline_matches_budget": observation.get("provider_scheduled_shutdown_utc") == lock["hard_off_utc"],
            "user_deferred_timer_extension": deferred,
            "provider_timer_functionally_tested": observation.get("timer_functionally_tested", False),
            "independent_provider_api_verified": False, "guest_shutdown_observed_provider_off":
            lock.get("guest_shutdown_observed_provider_off") is True}


def timer_pulse(lock_path, heartbeat, *, now=None, allowed=True, reason=None):
    now = time.time() if now is None else now
    lock = read(lock_path)
    hours = max(0., (now-lock["billing_start_epoch"])/3600)
    # Counting intervening OFF time is a conservative upper bound, not measured billing.
    spend = hours*lock["gpu_hour_rate_ceiling_cny"]+lock["non_gpu_cost_ceiling_cny"]
    allowed = (allowed and now < lock["work_cutoff_epoch"]
               and hours < lock["maximum_gpu_hours"] and spend < lock["maximum_spend_cny"])
    pulse = {"time_epoch": now, "lock_sha256": sha256_file(lock_path), "work_allowed": allowed,
             "shutdown_mode": MODE, "guest_watchdog_pid": os.getpid(), "stop_reason": reason,
             "elapsed_gpu_hours_upper_bound": hours, "spend_cny_upper_bound": spend,
             "provider_billing_measured": False, "provider_off_verified": False}
    atomic_write_json(heartbeat, pulse)
    return pulse


def _stop_with_retries(stop, root, clock, sleep):
    attempt=0
    while True:
        try:
            stop()
            # A successful guest request is still not provider OFF proof.
            return
        except Exception as error:
            try:
                write_new(root/f"shutdown-error-{attempt:04d}.json", {
                    "epoch":clock(),"error_type":type(error).__name__,"message":str(error),
                    "provider_off_verified":False})
            except OSError:
                # Full/unwritable storage cannot veto the authorized stop.
                pass
            attempt+=1
            sleep(10)


def supervise(lock_path, heartbeat, operations_root, command, *, stop=None, clock=time.time,
              sleep=time.sleep, spawn=subprocess.Popen, failure_marker=None,
              guest_shutdown_enabled=True, shutdown_amendment=None):
    """Own one pipeline; failure/cutoff terminates work, with optional guest OFF.

    Completion grants only the existing 30-minute raw-transfer reserve. A
    hash-verified outside-copy acknowledgment stops earlier. Never release disk.
    """
    if type(guest_shutdown_enabled) is not bool:
        raise ValueError("guest shutdown setting must be boolean")
    if not guest_shutdown_enabled and not (isinstance(shutdown_amendment,str) and shutdown_amendment.strip()):
        raise ValueError("explicit user shutdown amendment required")
    lock = read(lock_path)
    # Preflight is authorized to measure feasibility; it cannot dispatch
    # training unless its separate measured full-forecast gate passes.
    validate_timer_lock(lock, clock(), require_forecast_fit=False)
    root = Path(operations_root)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/"supervisor-intent.json", {"argv": command, "lock_sha256": sha256_file(lock_path),
              "epoch": clock(), "shutdown_mode": MODE, "automatic_scientific_retry": False,
              "guest_shutdown_enabled":guest_shutdown_enabled,"shutdown_amendment":shutdown_amendment})
    stop = stop or stop_autodl_compute
    process = None
    reason = "supervisor_failure"
    try:
        timer_pulse(lock_path, heartbeat, now=clock())
        with (root/"pipeline.log").open("xb") as log:
            process = spawn(command, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
        write_new(root/"pipeline-dispatched.json", {"pid": process.pid, "epoch": clock()})
        raw_finished = None
        while True:
            now = clock()
            pulse = timer_pulse(lock_path, heartbeat, now=now)
            code = process.poll()
            # A failed CUDA child can hang during NCCL teardown. Its immutable
            # failure marker ends paid work without waiting for process exit.
            if failure_marker is not None and Path(failure_marker).exists():
                reason = "pipeline_failed"; break
            if code is not None and code != 0:
                reason = "pipeline_failed"; break
            if now >= lock["work_cutoff_epoch"] or not pulse["work_allowed"]:
                reason = "work_or_budget_cutoff"; break
            if code == 0:
                if raw_finished is None:
                    raw_finished = now
                    write_new(root/"gpu-pipeline-exit.json", {"exit_code": 0, "epoch": now,
                              "independent_raw_transfer_pending": True})
                ack = root/"independent-transfer-verified.json"
                if ack.exists():
                    receipt = read(ack)
                    if (receipt.get("raw_hashes_verified") is not True
                        or receipt.get("full_checkpoint_hash_verified") is not True
                        or receipt.get("lock_sha256") != sha256_file(lock_path)):
                        reason = "invalid_transfer_receipt"; break
                    reason = "durable_gpu_work_complete"; break
                if now-raw_finished >= lock["shutdown_reserve_seconds"]:
                    reason = "transfer_reserve_expired"; break
            sleep(10)
    finally:
        try:
            timer_pulse(lock_path, heartbeat, now=clock(), allowed=False, reason=reason)
            write_new(root/("shutdown-request.json" if guest_shutdown_enabled else "operator-attention.json"),
                      {"reason": reason, "epoch": clock(), "guest_shutdown_enabled":guest_shutdown_enabled,
                       "operator_shutdown_after_completion":not guest_shutdown_enabled,
                       "release_disk": False, "provider_off_verified": False})
        finally:
            try:
                if process is not None and process.poll() is None:
                    try: os.killpg(process.pid, 15)
                    except ProcessLookupError: pass
                    # Bounded preservation, never wait beyond provider deadline.
                    try: process.wait(timeout=min(30, max(0, lock["hard_off_epoch"]-clock())))
                    except subprocess.TimeoutExpired: pass
            finally:
                if guest_shutdown_enabled:
                    _stop_with_retries(stop,root,clock,sleep)
