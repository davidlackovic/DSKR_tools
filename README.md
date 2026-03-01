# DSKR Tools
Paket orodij za laboratorijske vaje pri predmetu **Dinamika strojev in konstrukcij (DSKR)**. 
Vključuje numerično analizo lastnih nihanj 2D paličnih konstrukcij z metodo končnih elementov (MKE).

## Funkcionalnosti
* **Modeliranje paličij:** Definicija vozlišč, elementov in materialnih lastnosti ($A, E, \rho$).
* **Modalna analiza:** Izračun globalnih togostnih ($K$) in masnih ($M$) matrik ter reševanje problema lastnih vrednosti.
* **Upoštevanje robnih pogojev:** Podpora za poljubne omejitve prostostnih stopenj preko matrike omejitev.
* **Interaktivna vizualizacija:** Animacija lastnih oblik z uporabo drsnika za izbiro načina nihanja.

## Primer uporabe
```
import numpy as np
from DSKR_tools import Truss
%matplotlib qt

# Definiraj vozlišča
nodes = np.array([[0,0], 
                    [1,0], 
                    [0, 1],])

# Definiraj elemente
elements = np.array([[0,1],
                    [1,2],
                    [0,2],])
# Definiraj omejitve
phi = np.pi/4
C = np.zeros((3,6))
C[0,0]=1
C[1,1]=1
C[2,2]=np.sin(phi)
C[2,3]=-np.cos(phi)

# Ustvari model (A=presek, E=modul elastičnosti, rho=gostota)
model = Truss(nodes, elements, A=1e-4, E=210e9, rho=7850, constraints=C)

# Zaženi animacijo lastnih oblik
model.animate_mode_shapes(scale=0.2)
```
<img src="images/example.gif" width="700">
