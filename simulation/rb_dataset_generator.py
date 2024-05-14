from rlnoise.randomized_benchmarking import rb_dataset_generator

exp_folder = "simulation/experiments/3q_low_noise/"

config_file = exp_folder + "config.json"

if "chip_conf" in config_file.keys():
    from rlnoise.utils_hardware import QuantumSpain
    from qiboconnection import ConnectionConfiguration
    chip_conf = config_file["chip_conf"]
    configuration = ConnectionConfiguration(username = chip_conf["username"],api_key = chip_conf["api_key"])
    backend = QuantumSpain(configuration, device_id=chip_conf["device_id"], nqubits=chip_conf["nqubits"], qubit_map=chip_conf["qubit_map"])
else:
    backend = None
    
rb_dataset_generator(config_file,backend)