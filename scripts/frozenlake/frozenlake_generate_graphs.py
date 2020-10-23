from utils.experiments import evaluate_experiment
from utils.misc import load_v_history, get_oldest_history
import numpy as np
import matplotlib.pyplot as plt
n_experiments = 10
name = "frozenlake_small"
td_n = 4
alpha = 0.01
ord_histories_name = ["mc_ord_{}_run_{}".format(name, i) for i in range(n_experiments)]
weighted_histories_names = ["mc_weighted_{}_run_{}".format(name, i) for i in range(n_experiments)]
td_histories_names = ["td({})_{}_{}_run_{}".format(td_n, name, alpha, i) for i in range(n_experiments)]

ord_histories = [load_v_history(name) for name in ord_histories_name]
weighted_histories= [load_v_history(name) for name in weighted_histories_names]
td_histories = [load_v_history(name) for name in td_histories_names]


baseline_history = load_v_history('mc_{}'.format(name))

baseline = get_oldest_history(baseline_history)



list_of_histories = [ord_histories, weighted_histories, td_histories]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

for histories in list_of_histories:
    rmses = evaluate_experiment(histories, baseline)

    run_lengths = [run_lenght for run_lenght, rmse in rmses]
    rmses = [rmse for run_lenght, rmse in rmses]



    mean = np.mean(rmses, axis=0)
    std = np.std(rmses, axis=0)

    plt.plot(run_lengths[0], mean)
    plt.fill_between(run_lengths[0], mean + std, mean - std, alpha=0.5)

plt.show()
