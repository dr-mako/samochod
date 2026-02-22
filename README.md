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

---

## Katalog FEV_BEV_Transform/ – przygotowanie transformacji BEV
Celem jest wyznaczenie przekształcenia, które pozwala remapować obraz z kamery (FEV) do widoku z góry (BEV). W praktyce pipeline opiera się o ręcznie klikaną siatkę/szachownicę i późniejsze „oczyszczanie” punktów.

Typowy przepływ: ręczne oznaczenie punktów siatki → oczyszczenie/interpolacja → wyznaczenie map remapowania → maska ROI

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

---

## Katalog MAP_FITTING/ – offline map fitting i kalibracja ekstrynsyki

Założenia map fittingu w tym etapie pracy:
- trajektoria pojazdu z odometrii jest traktowana jako „prawda” (zamrożona),
- nie wolno jej korygować,
- estymowane są tylko:
	- pozycje landmarków w jednej mapie,
	- parametry systemowe (np. efektywna ekstrynsyka BEV/kamery względem punktu kinematycznego pojazdu).

To jest celowo ograniczony etap diagnostyczny: jeśli trajektoria jest błędna, mapa zostanie zdeformowana, ale samo to jest informacją o tym, gdzie model ruchu przestaje pasować.

Zawartość (skrótowo)
- sync_frames.py, Synchronizacja logów sterowania (MOTOR/SERVO) z czasami klatek kamery i zapis do CSV (1 wiersz = 1 klatka). Wejście: camera.txt (frame_id, time), out.txt (zdarzenia JSON). Wyjście: synced_raw.csv z uśrednionymi sygnałami sterowania.

- plot_and_traj.py, Konwersja sterowań na sterowanie kinematyczne (np. \(v\), \(\delta\)), integracja modelu Ackermanna, propagacja niepewności \(P\), zapis trajektorii do CSV + wykresy diagnostyczne.
Wyjścia m.in.: synced_with_traj.csv, synced_simple.csv, path.csv.
- segment_lanes_morphology_test.py, Diagnostyka segmentacji pasów ruchu z obrazu (FEV→BEV → normalizacja tła → progowanie → morfologia → filtry). Zawiera presety A–D różniące się agresywnością segmentacji.
- batch_lane_extraction_fixlist_aruco.py, Batch-processing klatek: transformacja do BEV, segmentacja pasów, detekcja ArUco, przeliczenie położenia markerów do lokalnego układu pojazdu i zapis obserwacji do plików analitycznych.
- analyze_and_clean_summary.py, Czyszczenie obserwacji: wykrywanie błędnych detekcji landmarków (outliery), skoków pozycji i problemów z domknięciem pętli. Wynik: summary_clean.csv.
- calibrate_camera_extrinsics_offline.py, Offline estymacja „efektywnego” położenia układu BEV/kamery względem punktu kinematycznego pojazdu oraz ewentualnego yaw-offsetu. To nie jest wyłącznie kalibracja mechaniczna: wynik jest parametrem systemowym kompensującym naraz uproszczenia modelu ruchu, błędy montażowe i niedoskonałość odometrii.
- map_fitting_extrinsics.py (lub wersja zwektoryzowana), Rozwiązuje map fitting: dopasowanie mapy landmarków do zamrożonej trajektorii (SE(2) + pozycje landmarków), bez korekty trajektorii.

---

## Katalog SlamOffline/ – SLAM offline
Ten etap przechodzi od map fittingu do pełnego SLAM: trajektoria nie jest już parametrem stałym wynikającym z odometrii. Położenie trajektorii i landmarków są zmiennymi optymalizacji. Optymalizacja szuka kompromisu między:
- zgodnością kolejnych póz z odometrią (ciągłość ruchu),
- zgodnością obserwacji landmarków z ich stałymi pozycjami w świecie (domykanie pętli).

W praktyce workflow jest dwuetapowy:
1. etap kalibracyjny: estymacja globalnych biasów (np. skala BEV, korekty yaw, prosty parametr poślizgu),
2. etap runtime offline: biasy zamrożone, a korygowane są tylko trajektoria i mapa.

Dodatkowo są narzędzia diagnostyczne do wyrównania wyniku SLAM do GT (np. dopasowanie transformacji afinicznej dwóch chmur punktów).

Katalog Slam_Offline zawiera:

SLAM_stage_runtime_final.py
SLAM_stage_calib_prod.py
SLAM_stage_runtime_from_config.py
slam_output_aligned_affine.py

---

Jak uruchamiać pipeline (wysoki poziom)

1. Przygotuj transformację FEV→BEV oraz maskę ROI w FEV_BEV_Transform/ (wygeneruj map.mat, inv_map.mat, roi_mask.mat).

2. Zsynchronizuj logi sterowania z klatkami kamery i wyznacz trajektorię z odometrii w MAP_FITTING/.

3. Wykonaj batch ekstrakcję obserwacji ArUco (i ewentualnie pasów) oraz wyczyść obserwacje.

4. (Opcjonalnie) skalibruj efektywną ekstrynsykę BEV↔pojazd offline.

5. Uruchom SLAM offline w SlamOffline/:
	- najpierw etap kalibracyjny,

	- potem etap runtime offline,

	- na końcu diagnostyka: porównanie do GT / alignment.
