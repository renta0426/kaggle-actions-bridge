#!/usr/bin/env python3
"""Focused context-scope regression for the two-space-indented CodeParrot YAML.

Not a general GitHub Actions/YAML validator. GitHub's own validation remains
mandatory. This catches runner/step context in job.env before publishing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re

ALLOWED = {'github', 'needs', 'strategy', 'matrix', 'vars', 'secrets', 'inputs'}


def check(source: str) -> None:
    if '\t' in source:
        raise ValueError('tabs_not_supported_by_focused_checker')
    in_job_env = False
    for line in source.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip(' '))
        if indent <= 4:
            in_job_env = line == '    env:'
        if in_job_env and indent == 6:
            for expr in re.findall(r'\$\{\{(.*?)\}\}', line):
                expr = re.sub(r"'(?:[^']|'')*'", "''", expr)
                roots = set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*[.\[]', expr))
                # Match only context roots; nested property names are not roots.
                roots = {root for root in roots if re.search(r'(?<![\w.])'+re.escape(root)+r'\s*[.\[]', expr)}
                bad = roots - ALLOWED
                if bad:
                    raise ValueError('forbidden_job_env_context:' + ','.join(sorted(bad)))


def self_test() -> None:
    good = 'jobs:\n  diagnose:\n    env:\n      SHA: ${{ github.sha }}\n    steps:\n      - name: guard\n        env:\n          R: ${{ runner.environment }}\n'
    check(good)
    for context in ('runner.environment', 'steps.result', 'job.status', "runner['environment']"):
        bad = 'jobs:\n  diagnose:\n    env:\n      R: ${{ '+context+' }}\n    steps: []\n'
        try:
            check(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('forbidden_job_env_fixture_accepted')
    print('CODEPARROT_WORKFLOW_CONTEXT_REGRESSION PASS job_runner_rejected=1 step_runner_accepted=1')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow', type=Path)
    args = parser.parse_args()
    self_test()
    if args.workflow:
        check(args.workflow.read_text(encoding='utf-8'))
        print('CODEPARROT_WORKFLOW_JOB_ENV_CONTEXTS PASS')


if __name__ == '__main__':
    main()
