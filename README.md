# DiplomovaPraceAR

Kódy pro praktickou část mé diplomové práce. Cílem této práce je návrh systému bezznačkové rozšířené reality, jehož součástí je vizuální SLAM a zobrazení virtuálního předmětu do skutečného prostředí.

Jsou zde kódy a soubory pro aplikaci SLAM systému a bezznačkové rozšířené reality:

✦ Model jelena, obj i mlt soubor, se nachází ve složce Deer a model vlka pak ve složce Wolf. Modely byly vytvořeny na stránce [Vectary.com](https://www.vectary.com/).

✦ Prvním kódem ke spuštění je kalibrace kamery camera_calib, která počítá parametry kamery používané dál. Je nutné vytisknout vzor šachovnice a nafotit ho vámi používanou kamerou z několika úhlů a vzdáleností. Tyto snímky jsou vstupem algoritmu.

✦ Dále se tam nachází soubor cam_params, do kterého je nutné získané informace o kameře vložit. Jsou využívané v dalším bodě a je nezbytné je přenastavit pro správný chod programu v závislosti na používané kameře.

✦ ORB_slam_class.py počítá SLAM algoritmus a do třídy Frame, která je využívána v hlavním souboru, ukládá všechny potřebné informace. Je založen na kódu autora Yitao Yu https://github.com/yitao-yu/PythonORBSlAM a následně mírně upraven, například doplněn o výpočet homografie.

✦ Objekty ve formátu obj se načítají v souboru load_obj, který je založen na skriptu ze stránek knihovny PyGame (https://www.pygame.org/wiki/OBJFileLoader). 

✦ Hlavní soubor AR_SLAM_main pak všechno zpracovává a jeho celkovým výstupem jsou tři okna zobrazující 3D mapu, proud snímků s vykresleným 3D objektem a nakonec tok snímků s vyznačenými klíčovými body. I tento soubor je založen na kódu https://github.com/yitao-yu/PythonORBSlAM, ale je rozšířen o několik bodů, včetně vložení bodů do obrazu, jejich přepočítávání a vkládání 3D objektu na správnou pozici, kde jsem se inspirovala z https://github.com/juangallostra/augmented-reality.
