# Textbaustein: Hypothetische Kundenberatersimulation

Zur beispielhaften Untersuchung der genetischen Optimierung wurde eine hypothetische Kundenberatersimulation mit 1'000 unterschiedlichen Fällen aus dem UCI-Trainingspool durchgeführt. Ein regelbasierter Berater vergibt anhand der jüngsten Rückzahlungsstatus und der Kreditauslastung die Entscheidungen «Bewilligen», «Prüfen» oder «Ablehnen». Ein Rückstand von mindestens zwei Monaten oder eine Auslastung über 90 % bei gleichzeitigem Zahlungsverzug führt zur Ablehnung. Geringerer Zahlungsverzug oder eine Auslastung über 70 % löst eine manuelle Prüfung aus; andernfalls wird bewilligt. Mit einer Wahrscheinlichkeit von 5 % wird eine andere Entscheidung gewählt, um uneinheitliches Beraterverhalten zu simulieren. Die Regeln und die Störwahrscheinlichkeit sind hypothetische Annahmen. Die tatsächlichen Ausfalllabels werden hierfür nicht verwendet.

Das bestehende CatBoost-Modell bleibt unverändert. Der genetische Algorithmus optimiert ausschliesslich die Schwellenwerte für die drei Handlungsempfehlungen. Von den simulierten Rückmeldungen werden 800 zur Optimierung und 200 zur anschliessenden Evaluation verwendet. Diese 200 Fälle sind für die Policy-Optimierung zurückgehalten, stammen jedoch aus dem Trainingspool des Risikomodells. Sie sind daher kein unabhängiger Test der CatBoost-Vorhersagen. Der Algorithmus verwendet eine Population von 30 Individuen, Turnierselektion, Crossover, Mutation und Elitismus; der Zufallsseed beträgt 42. Die abschliessende Policy wird ausschliesslich anhand des Trainingsrewards ausgewählt.

Auf den 200 zurückgehaltenen Policy-Testfällen steigt die Übereinstimmung mit dem simulierten Berater von 59,0 % bei festen Schwellenwerten auf 60,5 % nach einer Generation und 63,0 % nach 20 Generationen. Der mittlere Reward steigt von 0,424 auf 0,445 beziehungsweise 0,492. Nach 20 Generationen sinkt die Review-Rate allerdings von 16,0 % auf 0,5 %. Die Schwellenwerte verschieben sich von 0,45/0,65 auf rund 0,536/0,546; dadurch wird der Bereich für die manuelle Prüfung stark verkleinert.

Der Versuch zeigt, dass der genetische Algorithmus die Schwellenwerte an simuliertes Beraterfeedback anpassen kann. Er zeigt zugleich eine Grenze der verwendeten Belohnungsfunktion: Da eine übereinstimmende Prüfung mit +0,4 geringer belohnt wird als eine übereinstimmende Bewilligung oder Ablehnung mit +1, kann die Optimierung einen sehr engen Prüfbereich bevorzugen. Zudem lassen sich die mehrdimensionalen Regeln des Beraters mit zwei Schwellenwerten auf einem einzigen Risikoscore nur eingeschränkt abbilden. Eine verbesserte Übereinstimmung ist daher kein Nachweis geringerer Kreditausfälle oder fairerer Entscheidungen. Zusätzliche Feedbackgründe werden dokumentiert, aber nicht zur Optimierung verwendet. Der einzelne Simulationslauf erlaubt keine Aussage zur statistischen Stabilität. Nächste Schritte wären mehrere Zufallsstarts, alternative Beraterregeln und eine fachlich begründete Überarbeitung der Reward-Funktion und Review-Vorgaben.

# Anpassungen im vorhandenen PDF

- Executive Summary: Der gespeicherte CatBoost-Recall beträgt 61,6 %, nicht 61,1 %.
- Kapitel 7: Die fehlende Schwellenwertoptimierung ausdrücklich auf das CatBoost-Risikomodell beziehen. Die separate Policy wird im Simulationsversuch optimiert.
- Kapitel 8: Die Suche nach einem Klassifikationsschwellenwert mit Mindest-Precision ist bislang ein nächster Schritt, kein umgesetztes Ergebnis.
- Kapitel 8: Eine logistische Regression wurde bislang nicht als Baseline trainiert. Im vorhandenen Export wird eine konstante Trainingsprävalenz-Baseline ausgewertet; deren Klassenvorhersage ist immer «kein Ausfall».
- Kapitel 8: Tatsächlich ausgewertet sind statische Policy und GA mit simulierten Entscheidungslabels. Den Vergleich mit begründetem Feedback als geplante Erweiterung formulieren.
- Kapitel 9: Die Gruppenraten der Simulation sind deskriptive Auditindikatoren. Sie belegen weder individuelle Gleichbehandlung noch umfassende Chancengleichheit.
- Kapitel 10: «manuelle Prüfung sicherer Fälle» sollte vermutlich «manuelle Prüfung unsicherer Fälle» lauten.

Das Original-PDF wurde nicht verändert.
