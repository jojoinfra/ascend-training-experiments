"""Calculate throughput from completed Megatron pretrain logs (log-interval=1)."""
import argparse
import json
import math
import re
from pathlib import Path


def measure(folder, world=None, last_timed_step=None):
    completions, records = {}, {}
    for path in sorted(folder.glob('node-*.log')):
        text = path.read_text()
        for match in re.finditer(r'BENCH_COMPLETE (\{[^\n]*?\})', text):
            report = json.loads(match[1])
            rank = report.pop('rank')
            if rank in completions:
                raise ValueError(f'Duplicate completion for rank {rank}')
            completions[rank] = report
        for line in text.splitlines():
            match = re.search(r'iteration\s+(\d+)/\s*(\d+)\s*\|', line)
            if not match:
                continue
            step, total = map(int, match.groups())
            if step in records:
                raise ValueError(f'Duplicate iteration {step}; merge each node log only once')
            fields = dict(part.strip().split(':', 1) for part in line[match.end():].split('|') if ':' in part)
            records[step] = dict(
                total=total, step_s=float(fields['elapsed time per iteration (ms)']) / 1000,
                loss=float(fields['lm loss']), batch=int(fields['global batch size']),
                samples=int(fields['consumed samples']),
                skipped=int(fields['number of skipped iterations']), nan=int(fields['number of nan iterations']),
            )
    if 0 not in completions:
        raise ValueError('Missing rank 0 completion; collect all node logs after torchrun succeeds')
    first = completions[0]
    world = first['world'] if world is None else world
    if set(completions) != set(range(world)) or any(r != first for r in completions.values()):
        raise ValueError('Missing ranks or inconsistent run configuration')
    if first['world'] != world or first['parameters'] <= 0 or world % (first['tp'] * first['pp']):
        raise ValueError('Invalid model or parallel configuration')
    steps, batch, seq = first['steps'], first['global_batch'], first['seq']
    if steps < 5 or set(records) != set(range(1, steps + 1)):
        raise ValueError('Need every iteration of a completed run with at least 5 steps and log-interval=1')
    for step, item in records.items():
        if (item['total'] != steps or item['batch'] != batch or item['samples'] != step * batch
                or item['skipped'] or item['nan'] or not math.isfinite(item['loss']) or item['loss'] <= 0
                or not math.isfinite(item['step_s']) or item['step_s'] <= 0):
            raise ValueError(f'Invalid or skipped iteration {step}: {item}')
    last = steps - 1 if last_timed_step is None else last_timed_step
    if not 3 <= last < steps:
        raise ValueError('last_timed_step must be between step 3 and the penultimate update')
    timed = list(range(3, last + 1))
    seconds = sum(records[i]['step_s'] for i in timed)
    tokens = batch * seq * len(timed)
    return dict(source='megatron-pretrain-iteration-log', ranks=world, tp=first['tp'], pp=first['pp'],
                dp=world // (first['tp'] * first['pp']), parameters=first['parameters'],
                global_batch=batch, micro_batch=first['micro_batch'], seq=seq, steps=steps,
                timed_steps=timed, timed_tokens=tokens, timed_seconds=seconds,
                seconds_per_step=seconds / len(timed), tokens_per_second=tokens / seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--world', type=int, help='Expected total ranks (default: infer from completion metadata)')
    parser.add_argument('--last-timed-step', type=int,
                        help='Inclusive final measured update (default: penultimate update)')
    args = parser.parse_args()
    print(json.dumps(measure(args.folder, args.world, args.last_timed_step), indent=2))
