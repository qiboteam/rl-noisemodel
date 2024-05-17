from rlnoise.randomized_benchmarking import rb_dataset_generator
import json
import qibo

qibo.set_backend("qibojit",platform="numba")

exp_folder = "simulation/experiments/3q_low_noise/"

#exp_folder = "hardware/experiments/"

config_file = exp_folder + "config.json"

with open(config_file) as f:
    config = json.load(f)
if "chip_conf" in config.keys():
    from rlnoise.utils_hardware import QuantumSpain
    from qiboconnection.connection import ConnectionConfiguration
    chip_conf = config["chip_conf"]
    configuration = ConnectionConfiguration(username = chip_conf["username"],api_key = chip_conf["api_key"])
    backend = QuantumSpain(configuration, device_id=chip_conf["device_id"], nqubits=chip_conf["nqubits"], qubit_map=chip_conf["qubit_map"])
else:
    backend = None
    
rb_dataset_generator(config_file,backend)