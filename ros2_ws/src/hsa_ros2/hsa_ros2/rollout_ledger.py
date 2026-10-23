"""
The evaluation ledger for policy rollouts: one CSV row per trial, keyed in from the terminal.

    ros2 run hsa_ros2 rollout_ledger --ros-args -p ledger:=data/key_in_hole/wrist/v1/eval/ep025/ledger.csv \
        -p checkpoint:=ep025 -p condition:=wrist_extended

The rollout counterpart of the demo recorder's s/f keys. A trial's clock starts when the policy client's
first chunk reaches /polyumi/target_poses_traj (the moment the arm is told to move) and stops at the key:

    s  success                    f  failure, then a code (miss, jam, timeout, ...)
    p  placement for the trials that follow: id dx_mm dy_mm dtheta_deg
    u  undo the last row          q  quit

After s or f the depth, HSA hours and a note are asked on one line each (Enter leaves them empty). Rows
carry the columns of config/eval_ledger_example.csv (plot_outcomes.py reads them) plus the checkpoint, the
condition, the newest bag under <ledger dir>/bags and the start time, so a row finds its bag.
"""

from __future__ import annotations

import csv
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory

#: Column order of the ledger; the first eleven are config/eval_ledger_example.csv's.
COLUMNS = (
    'rollout',
    'placement_id',
    'dx_mm',
    'dy_mm',
    'dtheta_deg',
    'outcome',
    'failure_code',
    'insertion_depth_mm',
    'time_s',
    'hsa_hours',
    'notes',
    'checkpoint',
    'condition',
    'bag',
    'started_at',
)
#: A gap this long without a chunk, then a chunk, is a new trial (a relaunched client).
DEFAULT_IDLE_GAP_S = 3.0


@dataclass
class Placement:
    """Where the lock sits for the next trials, as the ledger records it."""

    placement_id: str = 'P00'
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dtheta_deg: float = 0.0

    @classmethod
    def parse(cls, text: str, current: Placement | None = None) -> Placement:
        """Parse ``id [dx dy [dtheta]]``; missing numbers keep the current placement's."""
        parts = text.split()
        if not parts:
            raise ValueError('placement needs at least an id, e.g. "P02 20 0 0"')
        base = current or cls()
        nums = [float(v) for v in parts[1:4]]
        return cls(
            parts[0],
            nums[0] if len(nums) > 0 else base.dx_mm,
            nums[1] if len(nums) > 1 else base.dy_mm,
            nums[2] if len(nums) > 2 else base.dtheta_deg,
        )

    def label(self) -> str:
        """``P02 (+20, 0, 0 deg)``."""
        return f'{self.placement_id} ({self.dx_mm:+g}, {self.dy_mm:+g}, {self.dtheta_deg:+g} deg)'


@dataclass
class TrialClock:
    """
    When the current trial started, from the chunk stream.

    The first chunk after ``idle_gap_s`` of silence (or ever) starts a trial; chunks inside the gap
    continue it. ``mark`` closes it (the s/f key), so the next chunk starts a new one.
    """

    idle_gap_s: float = DEFAULT_IDLE_GAP_S
    start: float | None = None
    last_chunk: float | None = None
    chunks: int = 0

    def on_chunk(self, t: float) -> bool:
        """Account for a chunk at ``t``; True when it started a trial."""
        started = self.start is None or (self.last_chunk is not None and t - self.last_chunk > self.idle_gap_s)
        if started:
            self.start = t
            self.chunks = 0
        self.last_chunk = t
        self.chunks += 1
        return started

    def mark(self, t: float) -> float:
        """Close the trial at ``t`` and return its duration (0 when no chunk started one)."""
        elapsed = 0.0 if self.start is None else max(0.0, t - self.start)
        self.start = None
        self.last_chunk = None
        self.chunks = 0
        return elapsed

    def state(self, now: float) -> str:
        """``waiting`` (no chunk yet), ``running`` (chunks flowing) or ``stalled`` (chunks stopped)."""
        if self.start is None:
            return 'waiting'
        if self.last_chunk is not None and now - self.last_chunk > self.idle_gap_s:
            return 'stalled'
        return 'running'


@dataclass
class Live:
    """The latest readings shown on the status line."""

    force: np.ndarray | None = None
    plate_z_m: float | None = None
    last_row: dict | None = None
    placement: Placement = field(default_factory=Placement)


# ---------------------------------------------------------------------------------- the CSV


def read_rows(path: pathlib.Path) -> list[dict]:
    """The ledger's rows, or [] when the file does not exist yet."""
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return [r for r in csv.DictReader(f) if any((v or '').strip() for v in r.values())]


def next_rollout(rows: list[dict]) -> int:
    """One past the largest ``rollout`` number in the ledger (1 for an empty one)."""
    nums = []
    for r in rows:
        try:
            nums.append(int(float(r.get('rollout') or 0)))
        except ValueError:
            continue
    return (max(nums) if nums else 0) + 1


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    """Rewrite the ledger with ``COLUMNS`` (extra keys from an older file are dropped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def append_row(path: pathlib.Path, row: dict) -> None:
    """Add one row, writing the header when the file is new."""
    rows = read_rows(path)
    rows.append(row)
    write_rows(path, rows)


def drop_last_row(path: pathlib.Path) -> dict | None:
    """Remove and return the last row, or None when the ledger is empty."""
    rows = read_rows(path)
    if not rows:
        return None
    last = rows.pop()
    write_rows(path, rows)
    return last


def newest_bag(bags_dir: pathlib.Path) -> str:
    """Name of the most recently modified directory under ``bags_dir`` ('' when none)."""
    if not bags_dir.is_dir():
        return ''
    dirs = [d for d in bags_dir.iterdir() if d.is_dir()]
    if not dirs:
        return ''
    return max(dirs, key=lambda d: d.stat().st_mtime).name


def make_row(
    rollout: int,
    placement: Placement,
    outcome: str,
    time_s: float,
    *,
    failure_code: str = '',
    insertion_depth_mm: str = '',
    hsa_hours: str = '',
    notes: str = '',
    checkpoint: str = '',
    condition: str = '',
    bag: str = '',
    started_at: str = '',
) -> dict:
    """One ledger row in ``COLUMNS`` order."""
    return {
        'rollout': rollout,
        'placement_id': placement.placement_id,
        'dx_mm': f'{placement.dx_mm:g}',
        'dy_mm': f'{placement.dy_mm:g}',
        'dtheta_deg': f'{placement.dtheta_deg:g}',
        'outcome': outcome,
        'failure_code': failure_code,
        'insertion_depth_mm': insertion_depth_mm,
        'time_s': f'{time_s:.1f}',
        'hsa_hours': hsa_hours,
        'notes': notes,
        'checkpoint': checkpoint,
        'condition': condition,
        'bag': bag,
        'started_at': started_at,
    }


def status_line(rollout: int, clock: TrialClock, live: Live, now: float) -> str:
    """The one-line status: trial, placement, clock, force, plate, last row."""
    state = clock.state(now)
    if state == 'waiting':
        when = 'waiting for the first chunk'
    else:
        when = f'{state.upper()} {now - clock.start:5.1f} s  {clock.chunks} chunks'
    force = 'F  --' if live.force is None else f'F {float(np.linalg.norm(live.force[:3])):4.1f} N'
    plate = 'plate  --' if live.plate_z_m is None else f'plate {live.plate_z_m * 1e3:+.1f} mm'
    last = ''
    if live.last_row:
        r = live.last_row
        code = f' {r["failure_code"]}' if r.get('failure_code') else ''
        last = f'  | last #{r["rollout"]} {r["outcome"]}{code} {r["time_s"]} s'
    return f'#{rollout} {live.placement.label()}  {when}  {force}  {plate}{last}  [s f p u q]'


# ---------------------------------------------------------------------------------- the terminal


class Keys:
    """Single keystrokes from a tty, with a way back to line mode for prompts."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        self._raw = False

    def raw(self) -> None:
        """Keystrokes arrive one at a time, unechoed."""
        if not self._raw:
            tty.setcbreak(self._fd)
            self._raw = True

    def cooked(self) -> None:
        """Back to line mode (for ``input()``)."""
        if self._raw:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._raw = False

    def get(self, timeout_s: float) -> str | None:
        """One key within ``timeout_s``, else None."""
        r, _, _ = select.select([self._fd], [], [], timeout_s)
        return os.read(self._fd, 1).decode(errors='ignore') if r else None

    def ask(self, prompt: str) -> str:
        """A whole line, echoed."""
        self.cooked()
        try:
            sys.stdout.write('\n')
            return input(prompt).strip()
        except EOFError:
            return ''
        finally:
            self.raw()


# ---------------------------------------------------------------------------------- the node


class RolloutLedger(Node):
    """Listen to the chunk stream, the wrench and the plate; the key loop runs in ``main``."""

    def __init__(self) -> None:
        super().__init__('rollout_ledger')
        p = self.declare_parameter
        self.ledger = pathlib.Path(str(p('ledger', 'ledger.csv').value)).expanduser()
        self.checkpoint = str(p('checkpoint', '').value)
        self.condition = str(p('condition', '').value)
        self.bags_dir = pathlib.Path(str(p('bags_dir', '').value) or self.ledger.parent / 'bags').expanduser()
        idle_gap = float(p('idle_gap_s', DEFAULT_IDLE_GAP_S).value)
        start_topic = str(p('start_topic', '/polyumi/target_poses_traj').value)
        wrench_topic = str(p('wrench_topic', '/hsa/wrench_100hz').value)
        plate_topic = str(p('plate_topic', '/platform_pose').value)
        placement = str(p('placement', '').value)

        self.lock = threading.Lock()
        self.clock = TrialClock(idle_gap_s=idle_gap)
        self.live = Live()
        if placement:
            self.live.placement = Placement.parse(placement)
        self.started_at = ''

        self.create_subscription(MultiDOFJointTrajectory, start_topic, self._chunk_cb, 10)
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)
        self.create_subscription(PoseStamped, plate_topic, self._plate_cb, 10)
        self.get_logger().info(
            f'ledger {self.ledger} (next rollout {next_rollout(read_rows(self.ledger))}), '
            f'trial clock from {start_topic}, bags in {self.bags_dir}'
        )

    def _chunk_cb(self, _msg: MultiDOFJointTrajectory) -> None:
        with self.lock:
            if self.clock.on_chunk(time.time()):
                self.started_at = time.strftime('%Y-%m-%d %H:%M:%S')

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        with self.lock:
            self.live.force = np.array([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])

    def _plate_cb(self, msg: PoseStamped) -> None:
        with self.lock:
            self.live.plate_z_m = float(msg.pose.position.z)

    # ------------------------------------------------------------------ the key loop

    def record(self, keys: Keys, outcome: str) -> dict:
        """Close the trial as ``outcome`` and append its row (asking for the free-text columns)."""
        with self.lock:
            time_s = self.clock.mark(time.time())
            started_at = self.started_at
            placement = self.live.placement
        code = keys.ask('failure code (miss, jam, timeout, ...): ') if outcome == 'failure' else ''
        depth = keys.ask('insertion depth mm (Enter = unknown): ')
        hours = keys.ask('HSA hours (Enter = skip): ')
        notes = keys.ask('notes: ')
        row = make_row(
            next_rollout(read_rows(self.ledger)),
            placement,
            outcome,
            time_s,
            failure_code=code,
            insertion_depth_mm=depth,
            hsa_hours=hours,
            notes=notes,
            checkpoint=self.checkpoint,
            condition=self.condition,
            bag=newest_bag(self.bags_dir),
            started_at=started_at,
        )
        append_row(self.ledger, row)
        with self.lock:
            self.live.last_row = row
        return row

    def run(self, keys: Keys) -> None:
        """Print the status line and act on keys until q."""
        keys.raw()
        while rclpy.ok():
            with self.lock:
                line = status_line(next_rollout(read_rows(self.ledger)), self.clock, self.live, time.time())
            sys.stdout.write('\r\x1b[2K' + line[: os.get_terminal_size().columns - 1])
            sys.stdout.flush()
            k = keys.get(0.25)
            if k is None:
                continue
            if k in ('s', 'f'):
                row = self.record(keys, 'success' if k == 's' else 'failure')
                sys.stdout.write(f'  -> row {row["rollout"]}: {row["outcome"]} {row["time_s"]} s, bag {row["bag"] or "-"}\n')
            elif k == 'p':
                text = keys.ask('placement: id dx_mm dy_mm dtheta_deg : ')
                try:
                    with self.lock:
                        self.live.placement = Placement.parse(text, self.live.placement)
                except ValueError as e:
                    sys.stdout.write(f'  {e}\n')
            elif k == 'u':
                gone = drop_last_row(self.ledger)
                sys.stdout.write('\n  nothing to undo\n' if gone is None else f'\n  dropped row {gone["rollout"]}\n')
                with self.lock:
                    rows = read_rows(self.ledger)
                    self.live.last_row = rows[-1] if rows else None
            elif k in ('q', '\x03'):
                sys.stdout.write('\n')
                return


def main(args=None) -> None:
    """Run the ledger: ROS in a background thread, keys on the terminal."""
    if not sys.stdin.isatty():
        print('rollout_ledger needs a terminal for its keys', file=sys.stderr)
        sys.exit(2)
    rclpy.init(args=args)
    node = RolloutLedger()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    keys = Keys()
    try:
        node.run(keys)
    except KeyboardInterrupt:
        pass
    finally:
        keys.cooked()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
