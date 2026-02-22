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


