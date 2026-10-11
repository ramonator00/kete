"""Offline threshold evolution; advisor labels are targets, never input features."""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.config import GA_CONFIG, MODELS_DIR, RANDOM_SEED
from src.domain import Action
from src.rl.reward_engine import immediate_alignment_reward


def decide(score, thresholds):
    review, decline = thresholds
    return Action.DECLINE if score >= decline else Action.REVIEW if score >= review else Action.APPROVE


def metrics(cases, thresholds):
    decisions = [decide(c['snapshot']['score'], thresholds) for c in cases]
    targets = [Action(c['advisor_decision']) for c in cases]
    groups = {}
    for case, decision in zip(cases, decisions):
        group = case['snapshot'].get('audit_group', 'unknown')
        groups.setdefault(str(group), []).append(decision == Action.APPROVE)
    rates = [sum(values) / len(values) for values in groups.values()]
    return {
        'mean_reward': sum(immediate_alignment_reward(a, b) for a, b in zip(decisions, targets)) / len(cases),
        'agreement_rate': sum(a == b for a, b in zip(decisions, targets)) / len(cases),
        'override_rate': sum(a != b for a, b in zip(decisions, targets)) / len(cases),
        'review_rate': decisions.count(Action.REVIEW) / len(cases),
        'approval_gap': max(rates) - min(rates) if len(rates) > 1 else None,
    }


def optimize(records, output_dir=None):
    # Keep the latest label per application; repeated feedback must not cross splits.
    latest = {}
    for record in sorted(records, key=lambda r: r['id']):
        if record.get('snapshot'):
            latest[record['application_id']] = record
    cases = list(latest.values())
    signatures = {c['snapshot']['policy_context'] for c in cases}
    if len(signatures) != 1:
        raise ValueError('Select feedback from one model and baseline policy context.')
    if len(cases) < 10:
        raise ValueError('At least 10 distinct applications with saved decision snapshots are required.')
    rng = random.Random(RANDOM_SEED)
    cases.sort(key=lambda c: c['application_id'])
    rng.shuffle(cases)
    split = max(1, int(len(cases) * .8))
    train, test = cases[:split], cases[split:]
    baseline = tuple(cases[0]['snapshot']['thresholds'])
    def individual():
        return tuple(sorted([rng.uniform(.05, .95), rng.uniform(.05, .95)]))
    population = [baseline] + [individual() for _ in range(GA_CONFIG['population_size'] - 1)]
    history = []
    for generation in range(GA_CONFIG['generations'] + 1):
        population.sort(key=lambda p: metrics(train, p)['mean_reward'], reverse=True)
        history.append({'generation': generation, 'training_reward': metrics(train, population[0])['mean_reward']})
        if generation == GA_CONFIG['generations']:
            break
        children = population[:GA_CONFIG['elite_size']]
        while len(children) < GA_CONFIG['population_size']:
            parents = [max(rng.sample(population, 3), key=lambda p: metrics(train, p)['mean_reward']) for _ in range(2)]
            child = list(parents[0])
            if rng.random() < GA_CONFIG['crossover_probability']:
                weight = rng.random()
                child = [weight * a + (1 - weight) * b for a, b in zip(*parents)]
            for index in range(2):
                if rng.random() < GA_CONFIG['mutation_probability']:
                    child[index] = max(.01, min(.99, child[index] + rng.gauss(0, .08)))
            children.append(tuple(sorted(child)))
        population = children
    best = population[0]
    comparisons = []
    for case in test:
        before = decide(case['snapshot']['score'], baseline)
        after = decide(case['snapshot']['score'], best)
        target = Action(case['advisor_decision'])
        comparisons.append({'application_id': case['application_id'], 'advisor': target.value,
                            'before': before.value, 'after': after.value,
                            'reward_before': immediate_alignment_reward(before, target),
                            'reward_after': immediate_alignment_reward(after, target)})
    now = datetime.now(timezone.utc)
    result = {
        'run_id': now.strftime('%Y%m%dT%H%M%S%fZ'), 'created_at': now.isoformat(),
        'seed': RANDOM_SEED, 'config': GA_CONFIG, 'policy_context': next(iter(signatures)),
        'optimized_parameters': ['review_threshold', 'decline_threshold'],
        'baseline_thresholds': baseline, 'candidate_thresholds': best,
        'train_ids': [c['application_id'] for c in train], 'test_ids': [c['application_id'] for c in test],
        'feedback_ids': [c['id'] for c in cases],
        'data_hash': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        'before': metrics(test, baseline), 'after': metrics(test, best),
        'history': history, 'comparison': comparisons, 'cases': cases,
        'status': 'offline_candidate',
    }
    destination = Path(output_dir) if output_dir is not None else MODELS_DIR / 'policy_runs'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / (result['run_id'] + '.json')).write_text(json.dumps(result, indent=2))
    return result
