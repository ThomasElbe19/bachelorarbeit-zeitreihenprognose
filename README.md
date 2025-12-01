# Implementierung zur Bachelorarbeit  
Dieses Repository enthält die vollständige Python-Implementierung der Bachelorarbeit  
"Vergleich klassischer Zeitreihenverfahren mit KI-basierten Modellen anhand der Prognose von Aktienkursen" von Thomas Elbe (2025).

Die Arbeit untersucht, wie gut klassische statistische Modelle (ARIMA, Prophet) und KI-basierte Modelle (Feedforward Neural Network, LSTM) tägliche Aktienrenditen prognostizieren können. Neben In-Domain-Analysen wird insbesondere die Cross-Domain-Generalisierbarkeit untersucht, also die Übertragbarkeit der Modellleistung zwischen verschiedenen Aktienmärkten.

---

## Projektstruktur

Die Implementierung befindet sich vollständig im Ordner `implementation/` und umfasst folgende Module:

- `data_pipeline.py` – Einlesen der Kursdaten, Berechnung der logarithmierten Renditen, Erstellen des Feature-Sets sowie Train-Valid-Test-Split.
- `model_arima.py` – Implementierung, Schätzung und Anwendung des ARIMA-Modells.
- `model_prophet.py` – Implementierung und Konfiguration des Prophet-Modells (Trend- und Saisonkomponenten).
- `model_ffnn.py` – Konstruktion und Training des Feedforward Neural Networks auf standardisierten Merkmalen.
- `model_lstm.py` – Aufbau und Training des LSTM-Modells mittels sequenzieller Eingaben (Sliding Windows).
- `metrics.py` – Funktionen zur Berechnung der Fehlermaße (MAE, RMSE).
- `experiment_execution.py` – Ausführung aller In-Domain-Experimente für IBM, Nike und NVIDIA.
- `experiment_cross_domain.py` – Durchführung sämtlicher Cross-Domain-Analysen (Quelle → Ziel).
- `plot_results.py` – Erstellung aller Validierungs-, Test- und Vergleichsplots.
- `rolling_volatility_plot.py` – Berechnung und Visualisierung der rollenden Volatilität.

---

## Ergebnisse

Alle erzeugten Plots, Metriken und Tabellen werden lokal in den Ordnern  
`results/` und `plots/` gespeichert. Diese Verzeichnisse sind nicht Teil des Repositories.

---

## Zweck des Repositories

Das Repository dient ausschließlich der Dokumentation und Reproduzierbarkeit der in der Bachelorarbeit durchgeführten Implementierung.  
Es enthält keine Datensätze und keine Ergebnisdateien, sondern ausschließlich die Python-Module, die für die Durchführung aller beschriebenen Experimente notwendig sind.

---

## Autor

Thomas Elbe  
Bachelorarbeit 2025  
GitHub: https://github.com/ThomasElbe19
