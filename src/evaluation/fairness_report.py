"""Descriptive held-out group audit: python -m src.evaluation.fairness_report."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool

from src.data.preprocessing import TARGET_COLUMN
from src.ml.model_io import MODEL_PATH, load_metadata, load_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'fairness'
THRESHOLD = 0.5


def group_statistics(frame, group_col):
    rows = []
    for group, cases in frame.groupby(group_col, observed=True, dropna=False):
        truth = cases[TARGET_COLUMN].to_numpy()
        prediction = cases['predicted_default'].to_numpy()
        tn = int(((truth == 0) & (prediction == 0)).sum())
        fp = int(((truth == 0) & (prediction == 1)).sum())
        fn = int(((truth == 1) & (prediction == 0)).sum())
        tp = int(((truth == 1) & (prediction == 1)).sum())
        rows.append({'group': group, 'n': len(cases), 'no_default_n': tn + fp,
                     'false_positives': fp, 'false_positive_rate': fp/(tn+fp) if tn+fp else np.nan,
                     'default_n': tp + fn, 'false_negatives': fn,
                     'false_negative_rate': fn/(tp+fn) if tp+fn else np.nan,
                     'true_positives': tp, 'true_negatives': tn})
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_path = ROOT / 'data/processed/uci_test.csv'
    frame = pd.read_csv(test_path)
    metadata, model = load_metadata(), load_model()
    features = metadata['feature_columns']
    frame['probability'] = model.predict_proba(frame[features])[:, 1]
    frame['predicted_default'] = (frame['probability'] >= THRESHOLD).astype(int)
    frame['gender_group'] = frame['SEX'].map({1: 'Male (SEX=1)', 2: 'Female (SEX=2)'}).fillna('Other/unknown')
    # Exploratory, transparent bins; no legal or normative interpretation.
    frame['age_group'] = pd.cut(frame['AGE'], bins=[0, 29, 39, 49, np.inf],
                                labels=['≤29', '30–39', '40–49', '≥50'])
    sex = group_statistics(frame, 'gender_group')
    age = group_statistics(frame, 'age_group')
    sex.to_csv(OUTPUT / 'false_positive_gender.csv', index=False)
    age.to_csv(OUTPUT / 'false_positive_age.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for ax, data, title in zip(axes, (sex, age), ('Geschlecht', 'Altersgruppe')):
        x = np.arange(len(data))
        ax.bar(x, data['false_positive_rate'] * 100, color='#477ba0')
        ax.set_xticks(x, data['group'], rotation=15)
        ax.set_title(title)
        ax.set_ylabel('Falsch-Positiv-Rate (%)')
        ax.grid(axis='y', alpha=.25)
        for i, row in data.reset_index(drop=True).iterrows():
            ax.annotate(f"{row.false_positive_rate:.1%}\n({row.false_positives}/{row.no_default_n})",
                        (i, row.false_positive_rate*100), ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(data['false_positive_rate'].max() * 125, 10))
    fig.suptitle('Falsch-Positiv-Raten · CatBoost-Testdaten · Schwellenwert 0,5')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / f'false_positive_rates.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

    pool = Pool(frame[features], cat_features=metadata['categorical_columns'])
    shap_values = model.get_feature_importance(pool, type='ShapValues')[:, :-1]
    shap_rows = []
    for col, grouping, attribute in [('gender_group', 'SEX', 'SEX'), ('age_group', 'AGE', 'AGE')]:
        values = shap_values[:, features.index(attribute)]
        for group in frame[col].dropna().unique():
            mask = frame[col].eq(group).to_numpy()
            shap_rows.append({'grouping': grouping, 'group': str(group), 'n': int(mask.sum()),
                              'feature': attribute, 'mean_shap': float(values[mask].mean()),
                              'mean_abs_shap': float(np.abs(values[mask]).mean())})
    pd.DataFrame(shap_rows).to_csv(OUTPUT / 'shap_demographic_features.csv', index=False)
    (OUTPUT / 'manifest.json').write_text(json.dumps({'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        'test_file_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(), 'n_test': len(frame),
        'threshold': THRESHOLD, 'positive_class': 'default_payment_next_month=1',
        'age_bins': '≤29, 30–39, 40–49, ≥50', 'age_bins_purpose': 'exploratory'}, indent=2))
    (OUTPUT / 'README.md').write_text('''# Explorative Fairness-Auswertung

Datensatz: gespeicherter UCI-Testsplit, eingefrorenes CatBoost-Modell, Schwellenwert 0,5. Positive Klasse: Zahlungsausfall im Folgemonat. Die Falsch-Positiv-Rate ist FP / (FP + TN), also der Anteil falsch als Ausfall eingestufter Fälle unter den tatsächlich nicht ausgefallenen Fällen. Der Klassifikationsschwellenwert ist nicht die dreistufige Policy-Empfehlung. Altersgrenzen wurden für diese explorative Analyse festgelegt. SEX=1/2 entsprechen den Datensatzkategorien male/female und bilden Geschlecht nur eingeschränkt ab.

Die Tabelle enthält zusätzlich Fallzahlen und Falsch-Negativ-Raten. Die Gruppen unterscheiden sich potenziell hinsichtlich weiterer Merkmale; Unterschiede sind deskriptiv, keine Kausal- oder Diskriminierungsnachweise. Keine Konfidenzintervalle oder Tests für statistische Signifikanz; kleine Gruppen und frei gewählte Altersgrenzen erschweren die Interpretation. Historische Kreditkartenlabels zeigen keine tatsächliche spätere Kreditbewilligung.

Die SHAP-Tabelle zeigt für SEX bzw. AGE den mittleren signierten und den mittleren absoluten Featurebeitrag je Gruppe, auf der Rohwert-/Log-Odds-Skala. Der Mittelwert hängt vom Modell, der Featureverteilung und dem SHAP-Hintergrund ab. Er ist kein kausaler Einfluss, kein Wahrscheinlichkeitsunterschied und kein eigenständiger Fairnessnachweis. Weitere Merkmale können als Proxys wirken. Im Modell werden AGE und SEX als Features verwendet. Nicht mit einer SHAP-Erklärung einzelner Entscheide verwechseln.
''')
    with ZipFile(ROOT / 'reports/fairness.zip', 'w', ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print('GROUPS BY SEX')
    print(sex.to_string(index=False))
    print('GROUPS BY AGE')
    print(age.to_string(index=False))
    print('SHAP')
    print(pd.DataFrame(shap_rows).to_string(index=False))


if __name__ == '__main__':
    main()
