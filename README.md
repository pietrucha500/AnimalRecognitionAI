# Rozpoznawanie obrazów - klasyfikacja zwierząt

### Cel:
Celem projektu jest rozpoznawanie zwierząt na obrazach spośród 10 kategorii:  
- motyle
- koty
- kury
- kurczaki
- krowy
- psy
- słonie
- konie
- owce
- pająki
- wiewiórki
### Architektura:
Swój model zbudowałem na podstawie [ResNet50](https://arxiv.org/abs/1512.03385) z kilku powodów:
- Wysoka głębia sieci (50 warstw) pozwala na lepsze wyodrębnienie cech
- Sprawdzona skuteczność w klasyfikacji wizualnej
- Szybkość i skuteczność dzięki skip connections  

Dodatkowo zastosowałem augmentację danych poprzez modyfikację danych podczas treningu:
- Przycięcie obrazu do losowego fragmentu obrazu (80%-100%)
- Losowe odbicie poziome
- Losowy obrót o maksymalnie +- 10 stopni
- Losowa zmiana jasności, kontrastu i nasycenia
- Losowa translacja obrazu

Dane zostały podzielone w proporcji train/val/test = 70/15/15
### Wyniki:
Wytrenowany model osiągnął następujące wyniki:  
- Test Loss: 0.2395  
- Test Accuracy: 0.9352

| Klasa      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| butterfly  | 0.96      | 0.94   | 0.95     | 580     |
| cat        | 0.91      | 0.92   | 0.92     | 464     |
| chicken    | 0.95      | 0.95   | 0.95     | 851     |
| cow        | 0.95      | 0.86   | 0.90     | 524     |
| elephant   | 0.95      | 0.94   | 0.94     | 401     |
| horse      | 0.95      | 0.91   | 0.93     | 726     |
| sheep      | 0.88      | 0.94   | 0.91     | 512     |
| spider     | 0.97      | 0.96   | 0.97     | 1342    |
| squirrel   | 0.90      | 0.96   | 0.93     | 518     |
| **accuracy**    |           |        | **0.94**     | 7288    |
| **macro avg**   | 0.93      | 0.93   | 0.93     | 7288    |
| **weighted avg**| 0.94      | 0.94   | 0.94     | 7288    |

---

## Spis treści

- [Wymagania](#wymagania)  
- [Instalacja](#instalacja)  
- [Użycie](#użycie)  
- [Trening modelu](#trening-modelu)  
- [Testowanie / Ewaluacja](#testowanie--ewaluacja)  
- [Struktura folderów](#struktura-folderów)  
- [Autorzy](#autorzy)  
- [Licencja](#licencja)  

---

## Wymagania

- Python 3.10  
#### PyTorch + CUDA 12.1
- torch 2.3.0
- torchvision 0.18.0
#### Przetwarzanie obrazów i dane
- numpy<2.0
- pillow
- matplotlib
- scikit-learn
#### Zapis modeli i wyniki
- tqdm
- pillow
- torchcam
- torchviz
- streamlit 
#### (Opcjonalnie) Notatniki Jupyter
- notebook

Wszystkie używane biblioteki znajdują się w pliku requirements.txt, który przy pomocy poniższej komendy zainstaluje wymagane biblioteki:
```bash
pip install -r requirements.txt
```
## Instalacja
```bash
git clone https://github.com/pietrucha500/AnimalRecognitionAI.git
cd AnimalRecognitionAI
pip install -r requirements.txt
```

## Użycie
Aby uruchomić trening modelu, wykonaj:
```bash
python main.py --train_dir ".\data\train" --val_dir ".\data\val" --epochs 1000 --batch_size 32
```
W przypadku zatrzymania treningu, można go wznowić od wybranego checkpointu:
```bash
python main.py --train_dir ".\data\train" --val_dir ".\data\val" --checkpoint checkpoints/epoch_number.pth --epochs 200 --batch_size 32
```
Aby przeprowadzić ewaluację modelu:
```bash
python test.py --test_dir ".\data\test" --model_path ".\best_model.pth"
```
## Trening modelu

Dataset użyty do treningu to [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10?select=raw-img).
Oryginalnie jest w języku włoskim, stąd zalecana jest ręczna zmiana nazwy na język angielski.
Dodatkowo, trzeba go podzielić na foldery train, val, test. Można to zrobić za pomocą pliku datacreator.py

## Testowanie / Ewaluacja

Model można testować przy pomocy:
- Jupyter Notebook'a, który posiada przykładową prezentację modelu (wymaga dodatkowego programu [GraphViz](https://graphviz.org/) do wizualizacji modelu)
- app.py, który przy pomocy biblioteki streamlit tworzy prostą aplikację w której możemy uploadować zdjęcia, uruchamianie:
```bash
streamlit run app.py
```

## Struktura folderów
```aiignore
/project
├── checkpoints/        # zapisywane modele i checkpointy
├── data/               # dane treningowe, walidacyjne i testowe
│   ├── train/
│   └── val/
│   └── test/
├── utils/              # pliki pomocniczne pozwalające na obróbkę danych
├── models/             # definicje architektury modelu
├── main.py             # plik do nauki modelu
├── test.py             # plik do testowania modelu
├── app.py              # plik tworzący interaktywną aplikację do testowania modelu
├── modelshowcase.ipynb # plik prezentujący działanie modelu
├── best_model.pth      # plik z zapisanym modelem
├── model.png           # wizualizacja modelu
├── README.md           # plik README
├── LICENSE             # plik licencji
└── requirements.txt    # lista pakietów Python
```
## Autorzy
- **Piotr Litych** – [GitHub](https://github.com/pietrucha500) – piotr.litych@student.uj.edu.pl

## Licencja
Ten projekt jest udostępniony na licencji MIT.  
Zobacz plik [LICENSE](LICENSE) po więcej informacji.

