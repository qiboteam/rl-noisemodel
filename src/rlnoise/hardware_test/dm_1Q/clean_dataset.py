import numpy as np
import qibo
from rlnoise.metrics import compute_fidelity

data = np.load("src/rlnoise/hardware_test/dm_1Q/200_circ_set_result (2).npy", allow_pickle=True)
print(data.shape)
circ = data[:,0]
print(circ.shape)
circ_list = []
non_mit = 0.
mit = 0.
for idx,i in enumerate(circ):
    # print(len(i.queue.moments))
    non_mit += compute_fidelity(data[idx,1],data[idx,2])
    mit += compute_fidelity(data[idx,1],data[idx,3])
    # print("non mit:",compute_fidelity(data[idx,1],data[idx,2]))
    # print("mit:", compute_fidelity(data[idx,1],data[idx,3]))
print(f"non mit: {non_mit/idx}, mit: {mit/idx}")
#     qasm_circ = i.to_qasm()
#     qibo_circ = qibo.Circuit.from_qasm(qasm_circ)
#     circ_list.append(qibo_circ)
#     data[idx,0] = qibo_circ

# np.save("src/rlnoise/hardware_test/dm_1Q/200_circ_set_result2.npy", data, allow_pickle=True)