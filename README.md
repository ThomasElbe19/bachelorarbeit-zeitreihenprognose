# Implementierung zur Bachelorarbeit  
Dieses Repository enthält die vollständige Python-Implementierung der Bachelorarbeit  
"Vergleich klassischer Zeitreihenverfahren mit KI-basierten Modellen anhand der Prognose von Aktienkursen" von Thomas Elbe (2025).

Die Arbeit untersucht, wie gut klassische statistische Modelle (ARIMA, Prophet) und KI-basierte Modelle (Feedforward Neural Network, LSTM) tägliche Aktienrenditen prognostizieren können. Neben In-Domain-Analysen wird insbesondere die Cross-Domain-Generalisierbarkeit untersucht, also die Übertragbarkeit der Modellleistung zwischen verschiedenen Aktienmärkten.

Die Implementierung dient der Reproduzierbarkeit aller in der Arbeit beschriebenen Experimente.

---

## Projektstruktur

Die Implementierung befindet sich vollständig im Ordner `implementation/` und umfasst folgende Module:

- `data_pipeline.py` – Einlesen der Rohdaten, Berechnung der logarithmierten Renditen, Erstellung des konsistenten Feature-Sets, Train-Valid-Test-Split.
- `feature_engineering.py` – Konstruktion von Lags, Rolling-Statistiken und weiteren Merkmalen.
- `model_arima.py` – Implementierung und Training des ARIMA-Modells.
- `model_prophet.py` – Implementierung und Training des Prophet-Modells.
- `model_ffnn.py` – Implementierung des Feedforward Neural Networks.
- `model_lstm.py` – Implementierung des LSTM-Modells.
- `experiment_execution.py` – Steuerung aller In-Domain-Experimente.
- `experiment_cross_domain.py` – Steuerung aller Cross-Domain-Analysen.
- `plot_results.py` – Erstellung der Validierungs- und Testplots sowie Vergleichsgrafiken.
- `utils.py` – Hilfsfunktionen (Skalierung, Metriken, split-Logik).


---

## Ergebnisse

Alle erzeugten Plots, Metriken und Tabellen werden lokal in den Ordnern  
`results/` und `plots/` gespeichert.  
Diese Verzeichnisse sind bewusst nicht Teil des Repositories, um die Codebasis klar und reproduzierbar zu halten.

---

## Zweck des Repositories

Das Repository dient ausschließlich der Dokumentation und Reproduzierbarkeit der in der Bachelorarbeit durchgeführten Implementierung.  
Es enthält keine Datensätze und keine Ergebnisdateien, sondern ausschließlich die Python-Module, die für die Durchführung aller beschriebenen Experimente notwendig sind.

---

## Autor

Thomas Elbe  
Bachelorarbeit 2025  
GitHub: https://github.com/ThomasElbe19
