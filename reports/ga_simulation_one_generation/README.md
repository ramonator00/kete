# Hypothetische Kundenberatersimulation

Ein GA-Trainingslauf mit 1000 Fällen, Population 30, 1 evolutionären Generationen und Seed 42.
Trainingsfälle: 800; zurückgehaltene Policy-Testfälle: 200.

| Metrik (Policy-Testset) | Statische Policy | GA-Kandidat |
|---|---:|---:|
| Agreement | 0.590 | 0.605 |
| Mittlerer Reward | 0.424 | 0.445 |
| Review-Rate | 0.160 | 0.135 |
| Override-Rate | 0.410 | 0.395 |

Review-/Ablehnungsschwellen: (0.45, 0.65) → (0.456005338201973, 0.6199271401119674).

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
