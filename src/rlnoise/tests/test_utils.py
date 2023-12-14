from rlnoise.utils import *
from qibo.models import Circuit
from qibo import gates

circ=Circuit(5)
circ.add(gates.CZ(0,1))
circ.add(gates.RX(0,1))
circ.add(gates.CZ(2,1))
circ.add(gates.RX(2,1))
circ.add(gates.RX(1,1))
circ.add(gates.CZ(0,3))
circ.add(gates.CZ(2,4))
circ.add(gates.RX(3,1))
circ.add(gates.CZ(1,2))
circ.add(gates.RX(0,1))
circ.add(gates.RX(4,1))
circ.add(gates.CZ(0,3))
circ.add(gates.RX(1,1))
circ.add(gates.RX(3,1))
print("BEFORE:")
print(circ.draw())

filled_circ=fill_identity(circ)
print("AFTER:")
print(filled_circ.draw())
print(filled_circ.queue.moments)

circ2=Circuit(1)
circ2.add(gates.RZ(0,0.1))
circ2.add(gates.RZ(0,0.1))
circ2.add(gates.RZ(0,0.1))
print("BEFORE:")
print(circ2.draw())

filled_circ2=fill_identity(circ2)
print("AFTER:")
print(filled_circ2.draw())
print(filled_circ2.queue.moments)