from wepy.walker import Walker
from wepy.resampling.resampler import RandomCloneMergeResampler

n_walkers = 8
init_weight = 1.0 / n_walkers

init_walkers = [Walker(i, init_weight) for i in range(n_walkers)]

resampler = RandomCloneMergeResampler(12312535346)

resampled_walkers = []
resampling_records = []
walkers = init_walkers
for i in range(3):
    cycle_walkers, cycle_records = resampler.resample(walkers)
    resampled_walkers.append(cycle_walkers)
    resampling_records.append(cycle_records)
