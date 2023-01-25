from dataset import Dataset

nqubits = 2
ngates = 10
ncirc = 5

# create dataset
dataset = Dataset(
    n_circuits=ncirc,
    n_gates=ngates,
    n_qubits=nqubits,
    val_split=0.2
)

print('> Training Circuits')
for c in dataset.get_train_loader():
    print(c.draw())

print('> Validation Circuits')
for c in dataset.get_val_loader():
    print(c.draw())
    
print('-------------------------------------')

print('Saving Circuits to dataset.json')
dataset.save_circuits('data/dataset2q.json')
dataset.load_circuits('data/dataset2q.json')
print('Loading Circuits from dataset.json\n')


for i in range(len(dataset)):
    print(dataset[i].draw())
