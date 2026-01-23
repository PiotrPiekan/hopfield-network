# Sieć Hopfielda

Program do tworzenia sieci hopfielda na podstawie czarno-białych obrazków.

Funkcje:

- rysowanie czarnobiałych obrazków bądź pobranie wzorców z bazy Fashion MNIST
- stworzenie na podstawie wzorców sieci hopfielda
- sprawdzenie działania sieci hopfielda krok po kroku, na podstawie wzorca.

Aplikacja działa w Pythonie w wersji 3.12.3. Wymagane są również następujące
pakiety pip:

- `PyQt6` (6.9.1)
- `numpy` (2.3.2)
- `pillow` (11.3.0)
- `datasets` (4.0.0)
- `matplotlib` (3.10.6)

oraz wszelkie ich zależności. Pełną listę pakietów można znaleźć w pliku
requirements.txt, w głównym katalogu repozytorium. Można zainstalować
wszystkie pakiety używając polecenia:

`pip install -r requirements.txt`

Do funkcji wczytania wzorców MNIST wymagane jest połączenie sieciowe.
Aby uruchomić aplikację wystarczy uruchomić poprzez pythona skrypt `app.py`, w
folderze `app/` repozytorium:

- Windows:
  - `python ./app/app.py`
- Linux:
  - `python3 ./app/app.py`
