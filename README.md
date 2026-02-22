# Repozytorium: samochód – kamera BEV, map fitting i SLAM offline


To repozytorium zawiera narzędzia do analizy przejazdów małego pojazdu z kamerą (rybie oko) i markerami ArUco. Pipeline składa się z trzech głównych etapów:


1. FEV → BEV: przygotowanie transformacji z obrazu rybiego oka (FEV) do widoku z lotu ptaka (BEV).

2. Map fitting (offline): estymacja położeń landmarków oraz parametrów układu odniesienia kamery przy zamrożonej trajektorii z odometrii.

3. SLAM offline: jednoczesna korekta trajektorii i budowa mapy landmarków na podstawie logów z przejazdu.

Repo jest podzielone na katalogi odpowiadające tym etapom.


---

## Struktura repozytorium
- FEV_BEV_Transform/
Narzędzia do przygotowania map remapowania FEV ↔ BEV oraz maski ROI. 
Wynikiem są pliki opisujące transformację (mapy remapowania), które potem są używane w segmentacji pasów i detekcji ArUco.
- MAP_FITTING/
Narzędzia do:

	- synchronizacji logów sterowania z klatkami kamery,

	- integracji modelu ruchu (odometria),

	- ekstrakcji obserwacji z kamery (pasy ruchu + ArUco),

	- czyszczenia obserwacji (outliery, skoki, domknięcia),

	- offline kalibracji „efektywnej” ekstrynsyki kamery/BEV względem punktu kinematycznego pojazdu,

	- dopasowania mapy landmarków do trajektorii (map fitting, bez korekty trajektorii).
- SlamOffline/
Narzędzia do SLAM offline:

	- etap kalibracyjny (estymacja globalnych biasów systematycznych na logach),

	- etap „runtime” offline (biasy zamrożone, poprawiane są tylko: trajektoria i mapa),

	- narzędzia diagnostyczne do wyrównania wyniku SLAM do Ground Truth (np. affine alignment).

## Katalog FEV_BEV_Transform/ – przygotowanie transformacji BEV
Celem jest wyznaczenie przekształcenia, które pozwala remapować obraz z kamery (FEV) do widoku z góry (BEV). W praktyce pipeline opiera się o ręcznie klikaną siatkę/szachownicę i późniejsze „oczyszczanie” punktów.

Typowy przepływ:

ręczne oznaczenie punktów siatki → oczyszczenie/interpolacja → wyznaczenie map remapowania → maska ROI

Zawartość (skrótowo):
- oznaczanie_naroznikow.py, Ręczne oznaczanie narożników siatki/szachownicy w klatce FEV.
- roi_pix_coord.py oraz oznaczanie_naroznikow_frames.py, Workflow do pracy na powiększonych fragmentach (zoom/ROI) w przypadku, gdy klikanie na pełnym obrazie jest za mało precyzyjne.
- ptab_cleaning.py, Czyszczenie/interpolacja siatki oznaczonych punktów (m.in. PCA) w celu wyrównania błędów ręcznego klikania.
- transformacja_make_maps.py, Buduje mapy remapowania (typowo map.mat i inv_map.mat), które opisują relację BEV ↔ FEV w postaci tablic Xmap, Ymap:

	- Xmap(Hdst, Wdst) – z jakiej kolumny w obrazie źródłowym pobrać piksel,

	- Ymap(Hdst, Wdst) – z jakiego wiersza w obrazie źródłowym pobrać piksel.
- make_roi_mask.py, Generowanie maski ROI do odcięcia obszarów martwych / nieużytecznych po transformacji.

Pliki danych (przykładowo):

- P_tab.csv, Clean_P_tab.csv – surowe i oczyszczone punkty siatki,
- map.mat, map_inv.mat – mapy remapowania,
- roi_mask.mat – maska obszaru użytecznego,
- przykładowe klatki wejściowe frames_*.jpg.


