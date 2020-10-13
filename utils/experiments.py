import numpy as np




def run_experiments(v_evaluator, env, behavior_policy, target_policy, num_episodes, sampling_function,
                    number_of_experiments, save_every, name, **kwargs):
    """
    Runs the experiments, returns a list of histories
    """

    histories = []
    for i in range(number_of_experiments):
        np.random.seed(i)
        _, hist = v_evaluator(env=env, behavior_policy= behavior_policy, target_policy= target_policy, num_episodes=num_episodes, sampling_function=sampling_function, save_every=save_every,
                              name="{}_run_{}".format(name, i), **kwargs)
        histories.append(hist)
    return histories


def evaluate_experiment(histories, baseline):
    rmses = [rmse_hist(history, baseline) for history in histories]
    return rmses


def rmse_hist(history, baseline):
    rmse = []
    run_lens = []
    for run_len, v in history.items():
        run_lens.append(run_len)
        rmse.append(root_mean_squared_error(v, baseline))
    return run_lens, rmse


def root_mean_squared_error(predictions, baseline):
    '''
    Has as input two default dicts, uses the keys of the baseline to know which entries to use
    '''
    results = np.sqrt( np.mean([  (baseline[key] - predictions[key])**2  for key in baseline.keys()]))
    return results


