from rlnoise import utils


print(utils.moments_matching(0,0.1,.005,.005, alpha=10))
print(utils.moments_matching(0.,0.,.001,.01, alpha=100))


'''
print(utils.kld(0,0.2,.01,.01))
print(utils.kld(0,0.2,.005,.005))
print(utils.kld(0,0.2,.001,.001))
'''