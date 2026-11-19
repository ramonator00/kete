"""Hypothetical advisor simulation; never writes to the real feedback database."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from src.evaluation.report import plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import GA_CONFIG
from src.data.preprocessing import TARGET_COLUMN
from src.domain import Action
from src.evolutionary.genetic_algorithm import decide, optimize
from src.ml.model_io import MODEL_PATH, load_metadata, load_model

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = [a.value for a in Action]


def advisor_decision(features, rng):
    """Synthetic rule, independent of outcome labels and CatBoost predictions.

    Repayment delay >=2 months or utilization >.9 with delayed repayment:
    decline. Smaller delays or utilization >.7: review. Otherwise approve.
    Five percent inconsistent decisions model hypothetical advisor noise.
    """
    delay = max(float(features[k]) for k in ['PAY_0', 'PAY_2', 'PAY_3'])
    utilization = max(float(features['BILL_AMT1']), 0) / max(float(features['LIMIT_BAL']), 1)
    if delay >= 2 or (utilization > .9 and delay >= 1):
        decision, reason = Action.DECLINE, 'repayment_delay_or_high_utilization'
    elif delay >= 1 or utilization > .7:
        decision, reason = Action.REVIEW, 'manual_check_required'
    else:
        decision, reason = Action.APPROVE, 'no_rule_triggered'
    noisy = rng.random() < .05
    if noisy:
        decision = rng.choice([a for a in Action if a != decision])
    return decision, reason, noisy


def run(output, n=1000, seed=42, generations=20):
    if n < 10:
        raise ValueError('At least 10 applications are required.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Use the risk model's training pool, preserving its existing test split.
    # Holdout below is independent only for policy fitting, not risk-model fitting.
    frame = pd.read_csv(ROOT / 'data/processed/uci_train.csv')
    features = load_metadata()['feature_columns']
    if n > len(frame):
        raise ValueError('Sample exceeds available unique applications.')
    sample = frame.sample(n=n, random_state=seed).reset_index(drop=True)
    probabilities = load_model().predict_proba(sample[features])[:, 1]
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    rng = random.Random(seed)
    records = []
    for i, row in sample.iterrows():
        decision, reason, noisy = advisor_decision(row[features], rng)
        records.append({'id': i+1, 'application_id': f"SIM-UCI-{int(row['ID'])}",
            'advisor_decision': decision.value, 'simulated_reason': reason,
            'noise_applied': noisy, 'simulation': True,
            'snapshot': {'score': float(probabilities[i]), 'thresholds': [.45, .65],
                         'policy_context': f'simulation:{digest}:0.45:0.65',
                         'audit_group': str(int(row['SEX']))}})
    result = optimize(records, output_dir=output / 'runs', seed=seed, generations=generations)
    result['simulation'] = True
    result['simulation_assumptions'] = {
        'source': 'UCI risk-model training split; no new credit applications',
        'advisor': 'repayment-delay / utilization rules, 5% random alternative decisions',
        'outcome_labels_used': False, 'risk_model_retrained': False,
        'holdout': '80/20 split by application, held out only from policy fitting',
        'structured_reasons_used_for_optimization': False,
        'sensitive_attributes': 'SEX used only for group audit; existing risk model includes demographic features',
        'selection': 'training reward only; holdout evaluated after final selection',
    }
    (output / 'simulation_result.json').write_text(json.dumps(result, indent=2))
    tables = {'learning_history': pd.DataFrame(result['history']),
              'test_metrics': pd.DataFrame({'Static baseline': result['before'], 'GA candidate': result['after']}),
              'test_decisions': pd.DataFrame(result['comparison']),
              'thresholds': pd.DataFrame({'Static baseline': result['baseline_thresholds'], 'GA candidate': result['candidate_thresholds']}, index=['review', 'decline'])}
    figures = {}
    def canvas(name, title, xlabel='', ylabel=''):
        fig, ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
        ax.set(title='SIMULATION · ' + title, xlabel=xlabel, ylabel=ylabel)
        figures[name] = fig
        return ax
    history = tables['learning_history']
    ax = canvas('01_learning_curve', 'GA training fitness', 'Generation', 'Mean alignment reward')
    ax.plot(history.generation, history.training_reward, label='Best individual')
    ax.plot(history.generation, history.mean_training_reward, label='Population mean')
    ax.axhline(result['before']['mean_reward'], ls='--', color='gray', label='Static policy · held-out reward')
    ax.legend(); ax.grid(alpha=.2)
    ax = canvas('02_test_comparison', 'Held-out policy metrics', ylabel='Metric value')
    tables['test_metrics'].loc[['mean_reward', 'agreement_rate', 'review_rate']].plot.bar(ax=ax, rot=0)
    ax.set_title('SIMULATION · Held-out policy metrics')
    test = tables['test_decisions']
    for key, label in [('before', 'Static baseline'), ('after', 'GA candidate')]:
        ax = canvas('03_confusion_' + key, label + ' vs simulated advisor')
        ConfusionMatrixDisplay.from_predictions(test.advisor, test[key], labels=ACTIONS,
            display_labels=ACTIONS, cmap='Blues', colorbar=False, ax=ax)
        ax.set(xlabel='Policy recommendation', ylabel='Simulated advisor decision')
    ax = canvas('04_thresholds', 'Decision thresholds', ylabel='Risk score threshold')
    tables['thresholds'].plot.bar(ax=ax, rot=0); ax.set_ylim(0, 1)
    ax.set_title('SIMULATION · Decision thresholds')
    ax = canvas('05_transitions', 'Policy recommendation changes')
    transition = pd.crosstab(test.before, test.after).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
    ax.imshow(transition.to_numpy(), cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=ACTIONS, yticklabels=ACTIONS,
           xlabel='GA candidate', ylabel='Static baseline')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(transition.iloc[i,j]), ha='center', va='center')
    tables['transitions'] = transition
    train_ids = set(result['train_ids'])
    test_cases = [c for c in records if c['application_id'] not in train_ids]
    groups = []
    for group in sorted({c['snapshot']['audit_group'] for c in test_cases}):
        cases = [c for c in test_cases if c['snapshot']['audit_group'] == group]
        groups.append({'group_SEX': group, 'n': len(cases),
                       **{label: np.mean([decide(c['snapshot']['score'], thresholds) == Action.APPROVE for c in cases])
                          for label, thresholds in [('Static baseline', result['baseline_thresholds']), ('GA candidate', result['candidate_thresholds'])]}})
    tables['group_approval_rates'] = pd.DataFrame(groups)
    ax = canvas('06_group_approval', 'Approval rates by audit group', ylabel='Approval rate')
    group_table = tables['group_approval_rates'].copy()
    group_table.index = [f"SEX={r['group_SEX']} (n={r['n']})" for r in groups]
    group_table[['Static baseline', 'GA candidate']].plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1); ax.set_title('SIMULATION · Approval rates by audit group')
    for name, fig in figures.items():
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{name}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv')
    before, after = result['before'], result['after']
    summary = f'''# Hypothetische Kundenberatersimulation

Ein GA-Trainingslauf mit {n} Fällen, Population {GA_CONFIG['population_size']}, {generations} evolutionären Generationen und Seed {seed}.
Trainingsfälle: {len(result['train_ids'])}; zurückgehaltene Policy-Testfälle: {len(result['test_ids'])}.

| Metrik (Policy-Testset) | Statische Policy | GA-Kandidat |
|---|---:|---:|
| Agreement | {before['agreement_rate']:.3f} | {after['agreement_rate']:.3f} |
| Mittlerer Reward | {before['mean_reward']:.3f} | {after['mean_reward']:.3f} |
| Review-Rate | {before['review_rate']:.3f} | {after['review_rate']:.3f} |
| Override-Rate | {before['override_rate']:.3f} | {after['override_rate']:.3f} |

Review-/Ablehnungsschwellen: {result['baseline_thresholds']} → {result['candidate_thresholds']}.

## Annahmen und Grenzen

Die Beraterentscheidungen sind vollständig simuliert: Rückzahlungsverzug und Kreditauslastung bestimmen eine Regelentscheidung; in 5 % der Fälle wird zufällig eine andere Aktion gewählt. Diese Regeln sind hypothetische Annahmen, keine validierte Bankpraxis. Historische Ausfalllabels werden nicht zur Simulation oder Optimierung verwendet. Gründe werden dokumentiert, aber vom GA nicht genutzt. Es wird keine Verbesserung gegenüber binärem Feedback nachgewiesen.

Die Fälle stammen aus dem Trainingspool des eingefrorenen CatBoost-Modells. Der 80/20-Holdout ist unabhängig für die Policy-Optimierung, nicht für das Risikomodell. Der vorhandene CatBoost-Testsplit bleibt unberührt. Der GA passt ausschließlich zwei Schwellenwerte an; er lernt keine neuen finanziellen Merkmale. Unterschiedliche Regeln des simulierten Beraters können mit zwei Schwellenwerten nicht vollständig abgebildet werden.

Reward: Übereinstimmung bei Genehmigung/Ablehnung +1, Widerspruch -1; beidseitige Prüfung +0,4; andere Fälle mit Prüfung -0,25. Gruppenraten sind deskriptiv und kein Fairnessnachweis. Ein einzelner Seed erlaubt keine Aussage zur Stabilität. Höhere Übereinstimmung bedeutet keine nachgewiesene Verbesserung realer Rückzahlungen. Der Kandidat bleibt offline; weder aktive Policy noch echte Feedbackdatenbank werden verändert.

## Abbildungen

01: Beste und durchschnittliche Trainingsfitness pro Generation. Die gestrichelte Referenz ist die statische Policy auf dem Holdout; sie wird nicht zur Auswahl genutzt.
02: Abschließender Vergleich auf denselben zurückgehaltenen Fällen.
03: Verwechslungsmatrizen gegenüber simulierten Beraterentscheidungen (keine Ausfalllabels).
04: Schwellenwerte vor und nach Optimierung.
05: Anzahl der Entscheidungswechsel auf dem Holdout.
06: Genehmigungsraten nach SEX-Gruppe einschließlich Fallzahlen.

PNG: 300 dpi; PDF/SVG: Vektorgrafiken; CSV: Tabellenwerte. Vollständige Konfiguration und Fallzuordnung: simulation_result.json.
'''
    (output / 'README.md').write_text(summary)
    with ZipFile(output.parent / (output.name + '.zip'), 'w', ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/ga_simulation')
    args = parser.parse_args()
    run(args.output, args.n, args.seed, args.generations)


if __name__ == '__main__':
    main()
