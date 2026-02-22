# REPOZYTORIUM SAMOCHÓD
# biblioteki instaluj w terminalu z pliku requirements.txt: pip install -r .\requirements.txt


Etap I – Offline Map Fitting
Celem jest estymacja parametrów systemowych oraz budowa spójnej mapy landmarków przy założonej trajektorii robota.
Etap II – Localization Against a Known Map
Etap III – SLAM (opcjonalnie, rozszerzenie)


Etap I – Offline Map Fitting
problem map fitting
✔ jedna pętla
✔ bez loop closure
✔ poznasz:
•	SE(2)
•	residuale
•	uwarunkowanie problemu

Założenia:
•	trajektoria robota jest zamrożona
•	nie wolno jej zmieniać
•	estymujesz tylko:
o	parametry systemowe (ekstrynsyka)
o	pozycje landmarków w jednej mapie
❌ Nie wolno deformować trajektorii

Programy:

sync_frames.py: 

zsynchronizowanie danych z logów sterowania (MOTOR/SERVO) z czasami kolejnych klatek kamery i zapisania wyniku do jednego pliku CSV (po jednym wierszu na klatkę).

    Wejście:

    - LOGI/camera.txt – lista klatek kamery: frame_id;time
    - LOGI/out.txt – log zdarzeń w formacie JSON (1 rekord na linię), m.in.:
        - {"time": ..., "type": "MOTOR", "data": {"id": ..., "spd": ...}}
        - {"time": ..., "type": "SERVO", "data": {"servo_1": ..., "servo_2": ...}}

    Wyjście:
    LOGI/synced_raw.csv – tabela zsynchronizowana do czasów klatek:
    - frame_id
    - time
    - spd_mean_raw
    - servo_mean_deg_raw

    Algorytm:
    1. Wczytuje czasy klatek z camera.txt, sortuje po czasie.
    2. Parsuje out.txt linia-po-linii:
        - ignoruje puste linie i rekordy niebędące poprawnym JSON,
        - wymaga pola time,
        - rozdziela rekordy na dwa strumienie:
            - MOTOR: zbiera time, id (int) i spd (float lub NaN)
            - SERVO: zbiera time, servo_1, servo_2 (float lub NaN)
    3. Próbkuje (dopasowuje) wartości MOTOR i SERVO do czasów klatek t_frame według wybranego trybu:
        - MODE = "zoh" (domyślnie) – zero‑order hold: dla każdej klatki bierze ostatnią znaną wartość sprzed tej klatki,
        - MODE = "linear" – interpolacja liniowa między próbkami (na brzegach zwraca NaN).
    4. Wylicza sygnały “średnie” (jak w istniejącej logice/Matlabie), nadal w surowych jednostkach:
        - SERVO (średnia w stopniach, bez offsetów):
            - servo_mean_deg_raw = (-servo_1_sync + servo_2_sync) / 2
        - MOTORY (średnia z 4 silników, “raw spd units”):
            - osobno synchronizuje spd dla id = 1..4,
            - spd_mean_raw = (-m1 + m2 + m3 - m4) / 4
    5. Zapisuje wynik do synced_raw.csv (UTF‑8), 1 wiersz = 1 klatka.

plot_and_traj.py 

przeliczenie sterowań na trajektorię + niepewność i wykresy
    Skrypt bierze dane zsynchronizowane do klatek z synced_raw.csv, przelicza je na sterowanie robota  \(v i \(\delta\) ), integruje model Ackermanna w czasie, propaguje macierz niepewności \(P\) zapisuje wyniki do CSV oraz rysuje wykresy (sterowania, kroku czasu, niepewności i trajektorii z elipsami błędu).
    - Wejście:
        - LOGI/synced_raw.csv (wymagane kolumny: frame_id, time, spd_mean_raw, servo_mean_deg_raw)
    - Wyjście:
        - LOGI/synced_with_traj.csv – pełny zestaw danych: sterowania, trajektoria, \(P\), sigmy i parametry elipsy
        - LOGI/synced_simple.csv – uproszczone sterowanie na klatkę: frame_id,time,v_mps,delta_rad
        - LOGI/path.csv – format “zgodnie z README” do dalszego użycia: pos_x,pos_y,current_speed,current_angle,timestep

segment_lanes_morphology_test.py 

diagnostyka skuteczności metod wyodrębnienia pasów ruch z obrazu rybie oko. Realizuje logikę: bird’s‑eye → normalizacja tła → progowanie → morfologia → filtr długości → opcjonalny anty‑blik → obrys/szkielet + diagnostyka  → ilustracje. Zawiera 4 presety (metody - do wyboru jedna) sterujące „agresywnością” segmentacji:
 A (najłagodniejszy)
Mniejsze usuwanie tła (bgDiskRadius mniejszy), mniejsze wygładzanie (smoothSigma mniejsze), próg Otsu bez podbijania (otsuScale≈1.0), mniejsze domknięcia i mniejsze progi filtrów (minArea, minLength).
Efekt: łatwiej przepuści cienkie/ciemne pasy, ale też łatwiej przepuści szum.
B (średni)
Mocniejsze usuwanie tła i wygładzanie, wyższy próg (otsuScale>1), bardziej wymagające filtry.
Efekt: mniej szumu, ale ryzyko utraty słabszych fragmentów. 
C (agresywny)
Jeszcze większe usuwanie tła + wygładzanie + najwyższy próg (otsuScale największy), większe wymagania na obiekty (minArea, minLength).
Efekt: zostają głównie „pewne” pasy; łatwo wyciąć delikatne ślady. 
D (C + anty‑blik)
Segmentacja jak w C, ale dodatkowo uruchamiany moduł „anty‑glare” (warunkowo, z bezpiecznikami).
Efekt: jeśli maska wygląda na „przykrytą” przez duży jasny blik, usuwa największy komponent (tylko gdy spełnione są warunki bezpieczeństwa).
UWAGA
W tym skrypcie „mapy” są osobnymi plikami .mat, które opisują transformację z obrazu kamery (FEV – front/normal view) do widoku z góry (BEV – bird’s-eye view) oraz maskę obszaru użytecznego.

batch_lane_extraction_fixlist_aruco.py

Co robi:

1. Transformuje obraz z rybiego oka do BEV (metryczna podłoga).
2. Segmentuje pasy ruchu.
3. Wykrywa markery ArUco.
4. Przekształca ich pozycję do lokalnego układu pojazdu.
5. Zapisuje:
	- pozycję w BEV,
	- ID,
	- dane do dalszej analizy.
Na tym etapie kamera stała się czujnikiem pozycji względem podłogi.


analyze_and_clean_summary.py

wykrywa żle oznaczone landmarki na podstawie analizy względnych pozycji
wykrywa duże skoki położeń landmarków (zamknęcie pętli)
wynik: summary_clean.csv 


calibrate_camera_extrinsics_offline.py

Program służy do estymacji efektywnego położenia punktu G (kamery / układu BEV) względem punktu C (środka modelu kinematycznego pojazdu) oraz ewentualnego błędu montażowego yaw kamery.

Problem:
Model samochodu ma 2 osie skrętne. Zastosowany model ruchu (odometria) operuje na punkcie C (środek pojazdu), a BEV  (widok z lotu ptaka) odnosi się do punktu G (kamera - środek układu współrzędnych).
Punkt G może być ustalony w oparciu o geometrię projektu samochodu. Jego odległość od C powinna wówczas wynosić d = L/2 + R. Jednakże że względu na blędy modelowania (np zastosowany model ruchu, błędy montażowe itd może tak nie być). Wówczas pozycje landmarków rozjeżdzają się. Możemy jednak odszukać "zastęcze" położenie puntu G względem C, wg algolytmu
- zmieniamy d (offset G),
- zmienialiśmy yaw kamery,
- przeliczamy globalne pozycje landmarków,
- minimalizujemy wariancję globalnej pozycji tego samego ID.
Nazywa się to ekstrynsika
UWAGA: Minimum wskaznika oznacza najbardziej spójny układ transformacji. Nie wolno jednak kalibrować układu na danych z zamknięciem pętli. Zastosowany program wykrywa zamknięcie pętli i ogranicza zakres obliczeń. 

Dane wejściowe
	- trajektoria pojazdu wyznaczoną z odometrii (synced_with_traj.csv)
	- globalne obserwacje landmarków z BEV (summary_clean.csv) 
Wielkości wyjściowe:
	- przesunięcia \(D_{G\leftarrow C}\) (wzdłuż osi podłużnej pojazdu),
    - przesunięcia wzdłuż osi Y
	- niewielkiego obrotu kamery (yaw_offset),
przelicza lokalne współrzędne landmarków do układu globalnego.

Wyznaczony offset nie jest czysto geometrycznym położeniem kamery w sensie mechanicznym, lecz:
efektywnym (modelowym) położeniem punktu G, dla którego model ruchu + transformacja BEV minimalizują globalny błąd landmarków.
Innymi słowy:
- Jest to punkt odniesienia, który najlepiej kompensuje:
	- uproszczenia modelu kinematycznego,
	- niedokładności odometrii,
	- niewielkie błędy montażu kamery,
	- offsety sterowania.
To jest parametr systemowy, a nie tylko czysto geometryczny.
Ponieważ BEV jest budowana z interpolacji kalibracyjnej, 
więc pitch i roll są już „wchłonięte” w map.mat i nie muszą być kompensowane.

W tym miejscu rozchodzą się dwie mozliwości:
- SLAM
- map fitting do odometrii

plot_landmarks_mapfit.py

Oblicza i rysuje położenie landmarków z uwzględnina korekcją wyznaczoną w 
optimize_camera_anchor_rms.py:
D_X_FROM_C_M = 0.11562500000000002
D_Y_FROM_C_M = 0.0024999999999999988
YAW_OFFSET_RAD = -2.6249999999999996 * np.pi / 180.0
Poprawione pozycje landmarków zapisuje do pliku landmarks_global.csv
UWAGA: MAM WĄTPLIWOŚĆ CZY TO JEST OK Z PUNKTU WIDZENIA SLAM
"Jeśli poprawiasz model sensora na podstawie mapy,
to przestałeś robić SLAM" to jest droga do map fitting do odometrii


(map_fitting_extrinsics.py)
zastosuj zamiennie map_fitting_extrinsics_wektorized.py

Ten program rozwiązuje:
sztywną transformację SE(2) pomiędzy układem:
C – trajektoria pojazdu
G – mapa landmarków
I to jest czysty map fitting / extrinsic calibration
(bez motion modelu, bez wheel slip, bez 4WS).

To jest zakończone zadanie "map fittingu"



SLAM:

SLAM_stage_runtime_final.py
SLAM_stage_calib_prod.py
SLAM_stage_runtime_from_config.py
slam_output_aligned_affine.py