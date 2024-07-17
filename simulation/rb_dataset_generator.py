from rlnoise.randomized_benchmarking import rb_dataset_generator
import json
import qibo

#qibo.set_backend("qibojit",platform="numba")

exp_folder = "simulation/experiments/1q/"

exp_folder = "hardware/experiments/single_qubit/"

config_file = exp_folder + "config_qibolab.json"

with open(config_file) as f:
    config = json.load(f)
if "chip_conf" in config.keys():
    chip_conf = config["chip_conf"]
    if chip_conf["backend"] == "qibolab":
        from rlnoise.utils_hardware import Qibolab_qrc
        backend = Qibolab_qrc(platform=chip_conf["platform"])
    elif chip_conf["backend"] == "QuantumSpain":
        from rlnoise.utils_hardware import QuantumSpain
        from qiboconnection.connection import ConnectionConfiguration
        configuration = ConnectionConfiguration(username = chip_conf["username"],api_key = chip_conf["api_key"])
        backend = QuantumSpain(configuration, device_id=chip_conf["device_id"], nqubits=chip_conf["nqubits"], qubit_map=chip_conf["qubit_map"])
else:
    backend = None
    
rb_dataset_generator(config_file,backend)