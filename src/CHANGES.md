# Was wir am Code verbessert haben — Übersicht fürs Team

Diese Datei fasst zusammen, welche Fehler wir vor der Abgabe behoben haben und
warum. Sie ist als Handover für euch gedacht, damit ihr nachvollziehen könnt, was
sich geändert hat. Bitte einmal durchlesen — **danach kann die Datei gelöscht
werden**, sie gehört nicht zur Abgabe.

Stil, Aufbau und Docstring-Format (NumPy-Style) der Originaldateien wurden
beibehalten. Alle Änderungen wurden mit Unit-Tests geprüft (Cleaner end-to-end,
EDA, Kategorie-Kodierung, predict-Pfad, save/load).

---

## cleaner.py

**Merge der beiden Condition-Spalten abgesichert.** Bisher konnte das Zusammen-
führen von `cond` und `condition` abstürzen, wenn eine der beiden Spalten fehlte
(`KeyError` erst viel später irgendwo in einem Plot). Jetzt werden alle Fälle
sauber behandelt: beide da → mergen; nur `condition` → umbenennen; nur `cond` →
unverändert; keine von beiden → sofort ein klarer `KeyError` mit aussagekräftiger
Meldung.

**Prüfung, ob die Datendatei überhaupt existiert.** Bei falschem Pfad gab es vorher
eine kryptische pandas-Fehlermeldung tief im Code. Der Konstruktor wirft jetzt
direkt `FileNotFoundError: Dataset not found: <pfad>`.

**Grenzen in den Docstrings begründet.** `_clean_size()` erklärt jetzt, warum nur
20–60 mm als gültig gelten (echte Gehäusedurchmesser; alles andere sind Parsing-
Artefakte wie Bandanstossbreiten). `_clean_yop()` erklärt die Spanne 1500 bis
aktuelles Jahr (davor gab es keine tragbaren Uhren, Zukunftsjahre sind unmöglich;
bewusst grosszügig, um antike Stücke nicht zu verlieren).

## eda.py

**Spalten werden beim Erstellen geprüft.** Fehlt eine erwartete Spalte, gibt es
jetzt sofort einen `KeyError` mit Auflistung der fehlenden Spalten — statt eines
schwer auffindbaren Absturzes mitten in einer Plot-Methode.

**Schutz gegen ungültige Preise.** Fehlende oder nicht-positive Preise würden bei
`np.log()` stillschweigend `-inf`/`NaN` erzeugen und alle Diagramme verfälschen.
Solche Daten lösen jetzt einen klaren `ValueError` aus.

**`top_n` wird validiert.** Ein `top_n` von 0 oder negativ erzeugte vorher ein
leeres/sinnloses Diagramm ohne Fehler. Jetzt muss es eine positive ganze Zahl sein,
sonst `ValueError`.

**`log_price` taucht nicht mehr im Missing-Values-Plot auf.** Diese künstlich
berechnete Spalte erschien fälschlich in der Fehlwerte-Grafik, als wäre sie Teil
der Rohdaten. Sie wird vor der Berechnung entfernt.

## model.py

**`y_train` wird jetzt mitgespeichert.** Vorher fehlte es im gespeicherten Modell,
dadurch ist `evaluate()` nach dem Laden eines Modells abgestürzt. Jetzt landet es
im Checkpoint und wird beim Laden wiederhergestellt.
⚠️ **Wichtig:** Alte gespeicherte Modelle (`.joblib`) sind dadurch inkompatibel —
einmal `python train.py` neu ausführen.

**Kategorien zwischen Training und Test korrekt verankert (wichtigster Fix).**
XGBoost arbeitet intern mit Zahlencodes für Kategorien. Vorher wurden Trainings-
und Testdaten unabhängig kodiert, sodass derselbe Wert (z. B. „Steel") in beiden
unterschiedliche Codes bekommen konnte — das Modell wurde also auf teils falsch
kodierten Daten bewertet, ohne Fehlermeldung. Jetzt wird der Testsatz fest an den
Trainingskategorien ausgerichtet.

**Gleicher Kategorie-Fix in `predict()`.** Bei einer einzelnen Vorhersage bekam
vorher *jede* Eingabe den Code 0 — d. h. alle fünf einfachen Kategorie-Merkmale
(Movement, Case-Material, Bracelet-Material, Condition, Sex) wurden bei jeder
Vorhersage in der App falsch interpretiert. Auch hier sind die Eingaben jetzt an
den Trainingskategorien verankert.

**`size`-Datentyp in `predict()` korrigiert.** `size=None` erzeugte einen
Datentyp, mit dem XGBoost abstürzte. `size` wird jetzt sauber als Fliesskommazahl
gebaut (analog zu `yop`), und ein leeres Feld funktioniert problemlos.

**`get_valid_options()` abgesichert.** Wenn `size`/`yop` komplett leer sind, gab es
einen unverständlichen `TypeError`. Jetzt kommt ein klarer `ValueError`.

**Eingabe-Prüfung in `predict()`.** Alle Kategorie-Eingaben werden gegen den
echten Datensatz geprüft (dieselbe Quelle, die auch die App-Dropdowns füllt) —
unbekannte Werte werden mit klarer Meldung abgelehnt. `size` muss leer oder
zwischen 20 und 60 mm liegen, `yop` leer oder zwischen 1500 und dem aktuellen Jahr.
Der Platzhalter `"Unknown"` ist bewusst immer erlaubt, weil die App ihn als
Standardwert in den Dropdowns nutzt und das Modell ihn selbst für fehlende
Kategorien verwendet.

**Eingabe-Prüfung in `fit()`.** `test_size` muss echt zwischen 0 und 1 liegen,
`n_iter` eine positive ganze Zahl sein, Pflichtspalten müssen vorhanden und Preise
gültig sein — sonst gibt es jeweils eine klare Fehlermeldung statt eines
undurchsichtigen Absturzes tief in scikit-learn.

**Hyperparameter-Suche vereinheitlicht (`n_iter = 25`).** Vorher durchsuchte das
ausgelieferte Modell nur 5 Kombinationen, während das Notebook 25 nutzte — die App
lieferte also ein schwächer abgestimmtes Modell als im Notebook gezeigt. `n_iter`
steht jetzt überall auf 25 (Standard in `fit()`, explizit in `train.py`, im
Notebook bereits 25), damit App und Notebook dasselbe Modell ergeben.

**Docstrings vervollständigt.** `__init__` hat jetzt einen Docstring; `fit()`
erklärt, dass `n_iter` × 5 (Cross-Validation) die Anzahl Trainingsläufe ergibt;
`predict()` beschreibt jeden Parameter inkl. Einheiten und gültiger Bereiche; und
das Beispiel ganz oben im Modul zeigt jetzt einen vollständigen `predict()`-Aufruf
mit allen neun nötigen Argumenten (vorher mit `...` abgekürzt, was nicht lauffähig
war).

## app.py

**Fehlerbehandlung um die Vorhersage.** Ungültige Eingaben zeigen jetzt eine
freundliche Fehlermeldung in der App (`st.error`) statt eines technischen
Streamlit-Stacktraces.

## LuxuryWatches.ipynb

**Dieselben Kategorie-Fixes wie in `model.py`** wurden auch im Notebook nachgezogen:
Testsatz und interaktiver Schätzer sind jetzt an den Trainingskategorien verankert,
jeweils mit erklärendem Kommentar. Der `size`-Datentyp-Fix ist hier nicht nötig,
weil das Notebook ohnehin `np.nan` verwendet. `n_iter` war im Notebook bereits 25.

## README.md

**Projektstruktur ehrlicher beschrieben.** Die Aussage, der Code habe „eine einzige
Quelle der Wahrheit", war nicht korrekt — Notebook und `src/` sind zwei parallele
Umsetzungen derselben Logik, die von Hand synchron gehalten werden. Der Text sagt
jetzt, dass das Notebook von Hand Schritt für Schritt geschrieben ist (für
Nachvollziehbarkeit) und die `src/`-Klassen dieselbe Logik in deploybarer Form
spiegeln. Ausserdem ist der Hinweis „Python ≥ 3.11" ergänzt (plus ein Mac-Hinweis
zu `brew install libomp` für XGBoost).

---

## Bewusst (noch) nicht geändert

Ein paar kleinere Punkte haben wir absichtlich offen gelassen, weil sie
funktionieren oder Geschmackssache sind:

- RMSE wird nur fürs XGBoost-Modell ausgegeben, nicht für die Baselines.
- Das Ridge-Baseline-Modell wird als Nebeneffekt innerhalb von `_split()` trainiert
  (sauberer wäre ein eigener Aufruf).
- Das gespeicherte Modell enthält die kompletten Datensätze und ist dadurch gross.
- Es braucht Python ≥ 3.11: Die aktuellen Abhängigkeiten (pandas ≥ 3.0, numpy ≥ 2.4)
  unterstützen 3.10 nicht mehr, und der Code nutzt moderne Typ-Hinweise (`X | None`).

## Vor der Abgabe noch zu tun

1. `python train.py` einmal neu ausführen (alte gespeicherte Modelle sind
   inkompatibel, und das Modell wird durch die Fixes neu/besser trainiert).
2. `evaluate()` laufen lassen und die MAPE-Zahlen im README („Part 6") prüfen —
   durch die Kategorie- und `n_iter`-Änderungen können sie sich verschieben.
3. Diese Datei (`CHANGES.md`) löschen — sie ist nur ein internes Handover.
