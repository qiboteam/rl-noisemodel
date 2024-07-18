from rlnoise.dataset import Dataset
import json
import qibo

qibo.set_backend("qibojit",platform="numba")

#exp_folder = "simulation/experiments/1q/"

exp_folder = "hardware/experiments/single_qubit/"

config_file = exp_folder + "config_qibolab.json"
save_path = exp_folder + "dataset"
eval_path = exp_folder + "eval_dataset"


with open(config_file) as f:
    config = json.load(f)
if "chip_conf" in config.keys():
    chip_conf = config["chip_conf"]
    if chip_conf["backend"] == "qibolab":
        from rlnoise.utils_hardware import Qibolab_qrc
        backend = Qibolab_qrc(platform=chip_conf["platform"], qubit_map=chip_conf["qubit_map"])
    elif chip_conf["backend"] == "QuantumSpain":
        from rlnoise.utils_hardware import QuantumSpain
        from qiboconnection.connection import ConnectionConfiguration
        configuration = ConnectionConfiguration(username = chip_conf["username"],api_key = chip_conf["api_key"])
        backend = QuantumSpain(configuration, device_id=chip_conf["device_id"], nqubits=chip_conf["nqubits"], qubit_map=chip_conf["qubit_map"])
else:
    backend = None

dataset = Dataset(config_file)
dataset.save(save_path)
dataset.generate_eval_dataset(eval_path,backend)
