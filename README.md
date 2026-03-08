# DSKR Tools
Paket orodij za laboratorijske vaje pri predmetu **Dinamika strojev in konstrukcij (DSKR)**. 
Vključuje numerično analizo lastnih nihanj 2D paličnih konstrukcij z metodo končnih elementov (MKE).

## Modeliranje konstrukcij
Knjižnica omogoča modeliranje dveh osnovnih tipov linijskih konstrukcij:

1. **Truss2D (Paličja):**
   - Vozlišča so modelirana kot **idealni členki**.
   - Elementi prenašajo izključno **osne sile** (nateg/tlak).
   - Vsako vozlišče ima 2 prostostni stopnji (translaciji $u, v$).
   
2. **Frame2D (Okvirji):**
   - Vozlišča so **toga**.
   - Elementi prenašajo **osne sile in upogibne momente (Eulerjeva teorija upogiba nosilcev**.
   - Vsako vozlišče ima 3 prostostne stopnje (translaciji $u, v$ ter rotacijo $\phi$).



## Funkcionalnosti
* **Interaktivni UI:** Vizualno nastavljanje robnih pogojev neposredno na modelu.
* **Diskretizacija (`n_mesh`):** Avtomatska poddelitev elementov na manjše segmente za natančnejši izračun lastnih oblik (ključno pri `Frame2D` za prikaz upogibnih linij).
* **Modalna analiza:** Izračun globalnih matrik $[K]$ in $[M]$ ter reševanje posplošenega problema lastnih vrednosti.
* **Prikaz lastnih nihanj:** Pripravljene metode za izris nihanja konstrukcije v izbranem lastnem načinu.


## Primer uporabe: Truss2D
```
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
from DSKR_tools import Truss2D
%matplotlib qt

# Definiraj vozlišča
nodes = np.array([
    [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], # Spodaj (0-5)
    [1.0, 1.732], [3.0, 1.732], [5.0, 1.732], [7.0, 1.732], [9.0, 1.732]    # Zgoraj (6-10)
])

# Definiraj elemente
elements = np.array([
    # Spodnji pas
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
    # Zgornji pas
    [6, 7], [7, 8], [8, 9], [9, 10],
    # Diagonale (cik-cak vzorec)
    [0, 6], [6, 1], [1, 7], [7, 2], [2, 8], [8, 3], [3, 9], [9, 4], [4, 10], [10, 5]
])

# Ustvari model (A=presek, E=modul elastičnosti, rho=gostota)
model = Truss2D(nodes, elements, A=1e-4, E=210e9, rho=7850)

# Prikažemo paličje
model.display_truss()
```
<img src="images/display_truss.png" width="700">

```
# Nastavimo robne pogoje (omejitve)
model.edit_constraints()
```

<img src="images/edit.gif" width="700">

Z desnim klikom izberemo vozlišče, nato pa s pritiskom tipk (1), (2) in (3) urejamo vozlišča:
- (1) doda nepomično členkasto podporo
- (2) doda pomično členkasto podporo, ki se lahko pomika vzdolž osi, definirane s kotom na drsniku desno zgoraj (kot se zaokroži na cele stopinje)
- (3) odstrani podporo

Omejitve lahko definiramo tudi s podajanjem matrike C pri ustvarjanju objekta Truss2D; enako bodo prikazane v uporabniškem vmesniku.

```
# Zaženi animacijo lastnih oblik
model.animate_mode_shapes(scale=2)
```
<img src="images/animate.gif" width="700">


## Dodatno: analiza konstrukcij s Frame2D
- Pri analizi konstrukcij z elementi tipa Frame so v metodi ```model.edit_constraints()``` na voljo tudi omejitve rotacij v členkih (konzolno vpetje).
- Pri definiciji objekta je na voljo tudi podajanje parametra  ```n_mesh```, s katerim definiramo poddelitev vsakega elementa na n_mesh elementov. S tem izboljšamo natančnost izrisa lastnih oblik, predvsem z vidika zasukov v členkih.
