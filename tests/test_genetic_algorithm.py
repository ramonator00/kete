import json
import sqlite3

from src.domain import Action
from src.evolutionary.genetic_algorithm import optimize
from src.feedback.feedback_repository import FeedbackRepository, SCHEMA
from src.feedback.feedback_service import FeedbackService


def test_snapshot_migrates_and_round_trips(tmp_path):
    path = tmp_path / 'feedback.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute(SCHEMA)
    repository = FeedbackRepository(path)
    snapshot = {'score': .3, 'thresholds': [.45, .65], 'policy_context': 'test'}
    FeedbackService(repository).submit('A', Action.APPROVE, Action.APPROVE, [], False, '', snapshot)
    assert repository.list_recent()[0]['snapshot'] == snapshot
    assert repository.list_recent()[0]['immediate_reward'] == 1


def test_evolution_is_reproducible_and_splits_applications(tmp_path):
    records = []
    for i in range(40):
        score = i / 40
        records.append({'id': i, 'application_id': str(i),
                        'advisor_decision': 'APPROVE' if score < .25 else 'REVIEW' if score < .55 else 'DECLINE',
                        'snapshot': {'score': score, 'thresholds': [.45, .65],
                                     'policy_context': 'test', 'audit_group': str(i % 2)}})
    records.append(dict(records[0], id=100))
    result = optimize(records, tmp_path)
    repeat = optimize(records, tmp_path)
    assert set(result['train_ids']).isdisjoint(result['test_ids'])
    assert len(result['train_ids']) + len(result['test_ids']) == 40
    assert 100 in result['feedback_ids'] and 0 not in result['feedback_ids']
    assert result['candidate_thresholds'] == repeat['candidate_thresholds']
    assert result['history'][-1]['training_reward'] >= result['history'][0]['training_reward']
    assert result['candidate_thresholds'][0] <= result['candidate_thresholds'][1]
    saved = json.loads((tmp_path / (result['run_id'] + '.json')).read_text())
    assert saved['data_hash'] == result['data_hash']
    assert len(saved['comparison']) == len(saved['test_ids'])
    # Changing only evaluation labels cannot influence selection.
    changed = [dict(record, advisor_decision='DECLINE') if record['application_id'] in result['test_ids'] else record for record in records]
    assert optimize(changed, tmp_path)['candidate_thresholds'] == result['candidate_thresholds']
