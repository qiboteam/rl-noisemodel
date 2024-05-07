from rlnoise.dataset import Dataset
from rlnoise.utils_hardware import QuantumSpain
from qiboconnection import ConnectionConfiguration

exp_folder = "simulation/experiments/3q_low_noise_long_dataset/"

config_file = exp_folder + "config.json"
save_path = exp_folder + "dataset"
eval_path = exp_folder + "eval_dataset"

if exp_folder[:10] == "simulation":
    backend = None
    nshots = None
    likelihood = False
    readout_mitigation = False
elif exp_folder[:8] == "hardware":
    chip_conf = config_file["chip_conf"]
    configuration = ConnectionConfiguration(username = chip_conf["username"],api_key = chip_conf["api_key"])
    backend = QuantumSpain(configuration, device_id=chip_conf["device_id"], nqubits=chip_conf["nqubits"], qubit_map=chip_conf["qubit_map"])
    nshots = chip_conf["nshots"]
    likelihood = False
    readout_mitigation = False

dataset = Dataset(config_file)
dataset.save(save_path)
dataset.generate_eval_dataset(eval_path,backend,nshots,likelihood,readout_mitigation)
