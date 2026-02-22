Transformacje_FEV_BE V/
  00_readme.md
  01 oznaczanie_naroznikow.py
  02 roi_pix_coord.py
  03 oznaczanie_naroznikow_frames.py
  04 ptab_cleaning.py
  05 transformacja_make_maps.py
  06 make_roi_mask.py

  data/
    frames/                (klatki wejściowe)
    roi_frames_20/         (Twoje ROI-y do powiększeń)
    map.mat                (Xmap,Ymap)  [po wygenerowaniu]
    roi_mask.mat           (roiMask, roiPos)
    P_tab.csv              (pixel_x,pixel_y,square_x,square_y)
    Clean_P_tab.csv

  01 program do recznego oznaczania narożników szachownicy
  Jeśli oznaczenia szczególnie w narożnikach jest niewystarczająco dokładne to powiększamy obszar roboczy:
  02 Pozwala przygotować ROI‑wycinki (zoom) z pełnej klatki FEV, zachowując wartości współrzędnych  pixeli pełnej klatki - stosujemy w celu "powiekszenia obszaru roboczego" szachownicy
  03 ręczne oznaczanie naroząników na fragmentach powiększonych przez poprzedni program
  04 interpolacja wyrównująca siatkę wykrytych narożników (stosuje PCA)
  05 transformacja FEF do BEV
  06 wycięcie martwych obszarów transformacji za pomoca maski ROI

  Transformacja:
  Z ręcznie klikanych (i później „oczyszczonych”) punktów szachownicy/siatki tworzy mapę remapowania z układu docelowego (BEV, „droga”) na obraz źródłowy (FEV, rybie oko). Wynikiem są dwie macierze:
- Xmap(Hdst, Wdst) – z jakiej kolumny w źródle pobrać piksel,
- Ymap(Hdst, Wdst) – z jakiego wiersza w źródle pobrać piksel.
Potem obraz BEV dostajesz przez:
- Python: cv2.remap(img, Xmap0, Ymap0, ...) (po konwersji na 0-based)

