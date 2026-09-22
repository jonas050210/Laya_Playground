# Laya Playground — Projektstand, Entscheidungen und Research-Fragen

Stand: 2026-09-22

## 1. Kontext

Dieses Projekt ist ein lokaler Playground für **Laya**, ein nicht-autoregressives Decision Model. Laya generiert keinen Text, sondern beantwortet typed decisions:

- `choice`: Auswahl zwischen Optionen
- `score`: ordinaler Score
- `noul`: Wahrscheinlichkeit für true/false

Der Playground zeigt Laya in Games, Workflows, Benchmarks und Failure-Cases. Wichtigste Regel des Projekts:

> Code berechnet Welt, Regeln, Features und Ground Truth. Laya bewertet beschriebene Entscheidungen. Python macht danach Argmax, Safety, Budgets und Aktionen.

## 2. User-Vorgaben / Wünsche

Der User hat gesagt:

- Sprache bevorzugt: Deutsch, gerne kurz und direkt.
- Reports normalerweise als Chat-Text, nicht als `.md`.
- Später explizit erlaubt: `project.md` erstellen.
- Snake soll **drin bleiben**.
- Snake ist wichtig, um zu zeigen, dass Laya live Decisions macht und wie schnell es ist.
- Qualität ist wichtiger als Speed.
- Ich darf autonom wie CEO/Lead Dev arbeiten und Entscheidungen treffen.
- Ich darf Features ändern, verbessern und priorisieren.
- Kein Ollama nötig, wenn Laya alleine läuft.
- Ziel: `start.py` soll möglichst alles automatisch vorbereiten und starten.

## 3. User-PC-Spezifikation

Der User-PC:

- Betriebssystem: Windows 11
- CPU: Intel Core i7-12700F
- GPU: NVIDIA GeForce RTX 4060 Ti
- VRAM: 8 GB

Einschätzung:

- Deutlich schneller als die Sandbox.
- Laya sollte auf CUDA laufen.
- 8 GB VRAM reichen für den 322M multilingual Checkpoint gut aus.
- Zielbefehl lokal:

```powershell
py start.py
```

Optional hart erzwungen:

```powershell
py start.py --gpu --device cuda
```

Profiling:

```powershell
py start.py --profile
```

## 4. Ollama-Entscheidung

Aktuell braucht das Projekt **kein Ollama**.

Grund:

- Laya läuft direkt mit PyTorch und HuggingFace Checkpoint.
- Backend lädt Laya lokal in den Prozess.
- Kein zweites Terminal nötig.
- Kein `ollama start` nötig.

Ollama wäre nur später optional sinnvoll für:

- lokalen LLM-Erklärer
- automatische Prompt-Vorschläge
- Laya-vs-LLM-Vergleich
- Natural-language Analyse über Benchmark-Ergebnisse

CEO-Entscheidung aktuell:

> Ollama nicht einbauen, solange Laya selbst noch nicht perfekt als One-Command-App läuft.

## 5. Was bisher geändert wurde

### 5.1 GPU / Device Runtime

Geändert in:

- `server/laya_runtime.py`
- `start.py`
- `tools/verify_runtime.py`
- `web/index.html`

Verbesserungen:

- Neues Device-System: `auto`, `cpu`, `cuda`
- Runtime nutzt CUDA automatisch, wenn verfügbar.
- Modellgewichte werden auf das gewählte Device geladen.
- Batches werden auf das richtige Device gelegt.
- Outputs werden vor NumPy-Konvertierung nach CPU kopiert.
- CUDA-Latenz wird mit `torch.cuda.synchronize()` korrekt gemessen.
- Runtime-Status enthält jetzt CUDA-/VRAM-Informationen.
- UI zeigt Device im Footer an.

### 5.2 Windows-Start verbessert

Geändert in:

- `start.py`
- `server/laya_runtime.py`

Verbesserungen:

- `resource`-Modul ist auf Windows nicht verfügbar; Fallback für Speicheranzeige wurde ergänzt.
- NVIDIA-Erkennung ohne Torch-Import über `nvidia-smi` / Windows-Fallback.
- Wenn NVIDIA GPU erkannt wird, wird CUDA-Torch bevorzugt.
- Wenn CPU-Torch schon installiert ist, aber CUDA gebraucht wird, wird Torch neu installiert.
- Danach startet Python automatisch neu, damit nicht das alte CPU-Torch im Prozess bleibt.

### 5.3 Neuer CLI Profile Mode

Neue Datei:

- `tools/profile_runtime.py`

Geändert in:

- `start.py`
- `README.md`

Neuer Befehl:

```powershell
py start.py --profile
```

Misst lokal:

- 1 kurze `noul`-Frage
- 3 gemischte Fragen
- 8 `noul`-Fragen
- 4 States × 1 Choice via `predict_many`

Ausgabe nur als Terminal-Text, keine Report-Datei.

### 5.4 Benchmark / CEO Mode

Neue Datei:

- `server/benchmarks.py`

Geändert in:

- `server/app.py`
- `web/index.html`

Neue API:

```txt
POST /api/benchmark/run
```

Neuer UI-Tab:

```txt
CEO Benchmark
```

Benchmark läuft über:

- Snake
- Aim Cascade
- 3D Shooter
- Triage Rush
- Guardrail Arena
- optional Calibration im Full-Modus

Zeigt:

- Panel
- Tier
- Score
- Baseline
- Accuracy
- Value
- Durchschnitts-ms
- Fragenanzahl
- kurze Bewertung

### 5.5 CEO-Einteilung der Panels

#### Showcase

- 3D Shooter
- Aim Cascade

Warum:

- echte Game-Entscheidungen
- klare Messwerte
- Code/Laya-Aufgabenteilung wird sichtbar

#### Product Proof

- Triage Rush
- Guardrail Arena
- RAG Filter
- Calibration Lab

Warum:

- zeigt echte Laya-Zielanwendungen
- Routing, Moderation, Guardrails, Kalibrierung

#### Speed Loop

- Snake

Warum:

- bleibt drin
- zeigt wiederholte Decisions
- zeigt Latenz und Geschwindigkeit sichtbar im Spiel

#### Failure / Legacy

- Minesweeper
- Maze
- Basic Aim Trainer
- Trading Floor

Warum:

- nicht löschen
- aber nicht als Hauptbeweis verkaufen
- nützlich als Failure-Lab, Prompt-Sensitivity und ehrliche Grenzen

### 5.6 Fixes und Cleanups

Geändert:

- `server/app.py`: ungenutzter Import entfernt
- `server/shooter.py`: veraltete Docstring-Aussage korrigiert
- `server/workflows.py`: ungenutzter Import entfernt
- `tools/verify_runtime.py`: CUDA-kompatibler gemacht
- `web/index.html`: Free Playground Beispiel gefixt
- `README.md`: Flags und Check-Zahlen aktualisiert

Weitere Fixes:

- `predict_many()` wirft jetzt Fehler, wenn Optionen nicht in `head_max_len` passen, statt Fragen still zu überspringen.
- Warm-up-Call zählt nicht mehr als echter User-Call.
- Frontend zeigt 68 Checks statt alter falscher Zahl.

## 6. Wichtige Research-Fragen

### 6.1 Kann man Laya mit Reinforcement Learning verbessern?

Antwort: Ja, aber nicht als Pixel-Agent.

Besserer Weg:

1. Spielcode erzeugt Features und Ground Truth.
2. Laya bekommt beschriebene Einzelentscheidungen.
3. Python macht Vergleich, Argmax, Safety und Aktion.
4. Logs werden als Trainingsdaten gesammelt.
5. Danach Fine-Tuning / RLCD / Policy-Optimierung.

Beispiel Shooter:

- Code kennt Position, Distanz, Damage, Rollout-Truth.
- Laya bekommt nur Text wie:
  - „A rusher is very close and attacking the player right now.“
- Laya beantwortet:
  - Ist dieser Kontakt am Angreifen?
  - Ist es ein Gegner?
- Python sortiert Kontakte.

### 6.2 Welche RL-Form ist sinnvoll?

Priorität:

1. **Evaluation zuerst**
2. Schwellen/Kaskaden offline optimieren
3. Klassische RL-Baselines bauen
4. Erst danach Laya-Fine-Tuning

Mögliche Tools später:

- Gymnasium für Environments
- Stable-Baselines3 für PPO/DQN/SAC-Baselines
- CleanRL für lesbaren RL-Code
- RLlib/PettingZoo für Multi-Agent
- Hugging Face TRL für LLM-Post-Training, falls später generative KI dazukommt

### 6.3 Welche Games sind sinnvoll?

#### Stark

- Shooter
- Aim Cascade
- Snake als Speed Demo

#### Mittel

- Triage Rush, Guardrail, RAG, Calibration sind eigentlich stärker als echte Product-Proof-Panels.

#### Schwach als „Laya ist gut“-Beweis

- Maze: Solver macht viel
- Minesweeper: eher Failure/Prompt-Sensitivity
- Basic Aim: durch Aim Cascade ersetzt
- Trading: interessant, aber kein klares „Laya gewinnt“-Signal

## 7. Aktueller technischer Stand

Geänderte Dateien aktuell:

- `README.md`
- `server/app.py`
- `server/laya_runtime.py`
- `server/shooter.py`
- `server/workflows.py`
- `server/benchmarks.py`
- `start.py`
- `tools/verify_runtime.py`
- `tools/profile_runtime.py`
- `web/index.html`

Keine Ollama-Integration.

Keine automatische `.md`-Reports außer dieser vom User gewünschten `project.md`.

## 8. Checks, die ausgeführt wurden

Erfolgreich:

```bash
python3 -m compileall -q .
```

```bash
ruff check server start.py tools --select F,E9
```

```bash
node --check /tmp/laya_web_script.js
```

```bash
python3 start.py --check
```

Nicht ausgeführt:

```bash
python3 start.py --verify
```

Grund:

- Sandbox hat aktuell nicht alle Runtime-Dependencies installiert.
- Checkpoint wäre groß.
- Zielsystem des Users ist sowieso Windows/NVIDIA und deutlich schneller.

## 9. Lokale Startbefehle für den User

Normal:

```powershell
py start.py
```

GPU erzwingen:

```powershell
py start.py --gpu --device cuda
```

Nur Systemcheck:

```powershell
py start.py --check
```

Lokale Performance messen:

```powershell
py start.py --profile
```

Verifikation:

```powershell
py start.py --verify
```

## 10. Nächste sinnvolle Schritte

### Prio 1 — Windows One-Command perfektionieren

- bessere Fehlertexte, wenn CUDA nicht verfügbar ist
- klare Anzeige Torch-Version / CUDA-Version / GPU Name / VRAM
- eventuell `--doctor` als ausführlicher Check

### Prio 2 — CEO Benchmark verbessern

- mehr Baselines
- bessere Full-Suite
- klare Score-Gewichtung
- eventuell Benchmark-Ergebnis im UI speichern, aber nicht als `.md`

### Prio 3 — Session-Isolation

Aktuell sind Sessions global im Server.

Später besser:

- Session-ID pro Browser
- keine Kollision zwischen mehreren Nutzern

### Prio 4 — Eval-Daten sammeln

Später Tools bauen für:

- Shooter-Szenen exportieren
- Aim-Cascade-Runs exportieren
- Triage/Guardrail Eval speichern
- JSONL statt Markdown

### Prio 5 — RL / Fine-Tuning vorbereiten

Erst wenn genug Daten da sind:

- Laya-Frageformen versionieren
- Train/Val/Test Seeds trennen
- Soft Labels aus Value/Damage ableiten
- Fine-Tuning oder Policy-Optimierung testen

## 11. CEO-Grundsatz

Nicht behaupten, dass Laya alles kann.

Stattdessen:

- Stärken sichtbar machen
- Schwächen sichtbar lassen
- Laya nie rohe Pixel/Koordinaten geben
- Code macht Wahrnehmung und Regeln
- Laya macht schnelle, kalibrierte Decisions
- Benchmarks statt Bauchgefühl
