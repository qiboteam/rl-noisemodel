from rlnoise import utils

print(utils.kld(1,1,1,1))
print(utils.kld(0,1,.001,.001))
print(utils.kld(0,1,.003,.003))
print(utils.kld(0,1,.01,.01))
print(utils.kld(0,1,.05,.05))

print(utils.kld(0,0.2,.01,.01))
print(utils.kld(0,0.2,.005,.005))
print(utils.kld(0,0.2,.001,.001))