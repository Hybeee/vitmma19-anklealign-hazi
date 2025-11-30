# vitmma19-anklealign-hazi

## Adatok előkészítése - jegyzet
### Adatok beolvasása
- Az adatok beolvasásánál - data_preparing.py - volt olyan feltöltő, aki cirill karaktereket írt a path-be -> törölni kellett
- Volt olyan, akinél a feltöltött file neve és a json-ben lévő file neve nem egyezett meg
    - H51B9J
- NC1O2T esetén szükség volt a folderek kibontására -> a feltöltött változatban a feltöltött folderen belül volt külön 1-1 mappa az egyes osztályoknak
### Adatok manuális tisztítása
Az adathalmazból összegyűjtöttem azon eseteket, ahol valamilyen watermark/rajz volt a képen. Ezeket a data/images_with_noise.json file-ba gyűjtöttem össze.

### Adatok automatikus tisztítása
Ezt a data_cleaning.py végzi el.