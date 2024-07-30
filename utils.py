
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import seaborn as sns

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# -_-_-_-_-_-_-_-_-_                       HELPER                                   _-_-_-_-_-_-_-_-_-_
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


def calc_execution_time(device, task, core, dvfs):
    if device['id'] == "cloud":
        return task["computationalLoad"] / device["voltages_frequencies"][0]
    else:
        return task["computationalLoad"] / device["voltages_frequencies"][core][dvfs][0]


def calc_power_consumption(device, task, core, dvfs):
    if device['id'] == "cloud":
        return 13.85
    return (device["capacitance"][core] * (device["voltages_frequencies"][core][dvfs][1] ** 2) * device["voltages_frequencies"][core][dvfs][0])


def calc_energy(device, task, core, dvfs):
    return calc_execution_time(device, task, core, dvfs) * calc_power_consumption(device, task, core, dvfs)


def calc_total(device, task, core, dvfs):
    timeTransMec = 0
    timeTransCC = 0
    baseTime = 0
    baseEnergy = 0
    totalEnergy = 0
    totalTime = 0

    transferRate5g = 1e9
    latency5g = 5e-3
    transferRateFiber = 1e10
    latencyFiber = 1e-3

    timeDownMec = task["returnDataSize"] / transferRate5g
    timeDownMec += latency5g
    timeUpMec = task["dataEntrySize"] / transferRate5g
    timeUpMec += latency5g

    alpha = 52e-5
    beta = 3.86412
    powerMec = alpha * 1e9 / 1e6 + beta

    timeDownCC = task["returnDataSize"] / transferRateFiber
    timeDownCC += latencyFiber
    timeUpCC = task["dataEntrySize"] / transferRateFiber
    timeUpCC += latencyFiber

    powerCC = 3.65

    if device["id"].startswith("mec"):
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec
        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime + timeTransMec
        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy + energyTransMec

    elif device['id'].startswith("cloud"):
        timeTransMec = timeUpMec + timeDownMec
        energyTransMec = powerMec * timeTransMec

        timeTransCC = timeUpCC+timeDownCC
        energyTransCC = powerCC * timeTransCC

        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime + timeTransMec + timeTransCC

        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy + energyTransMec + energyTransCC

    elif device['id'].startswith("iot"):
        baseTime = calc_execution_time(device, task, core, dvfs)
        totalTime = baseTime
        baseEnergy = calc_energy(device, task, core, dvfs)
        totalEnergy = baseEnergy

    return totalTime, totalEnergy


def checkIfSuitable(state, device):
    safeFail = 0
    taskFail = 0
    if state['safe'] and not device["handleSafeTask"]:
        safeFail = 1
    if state['kind'] not in device["acceptableTasks"]:
        taskFail = 1
    return taskFail, safeFail


def getSetup(e, t, setup, alpha=1, beta=1):
    match setup:
        case 1:
            return -1 * (alpha * e + beta * t)
        case 2:
            return 1 / (alpha * e + beta * t)
        case 3:
            return -np.exp(alpha * e) - np.exp(beta * t)
        case 4:
            return -np.exp(alpha * e + beta * t)
        case 5:
            return np.exp(-1 * (alpha * e + beta * t))
        case 6:
            return -np.log(alpha * e + beta * t)
        case 7:
            return -((alpha * e + beta * t) ** 2)

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
# -_-_-_-_-_-_-_-_-_                       UTILITY                                   _-_-_-_-_-_-_-_-_-_
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


def save_results(result_path, rewardSetup, punish, avg_loss_history, avg_fail_history, avg_time_history, avg_energy_history, avg_reward_history, num_epoch):
    half_num_epoch = num_epoch//2
    new_epoch_data = {
        "Setup": rewardSetup,
        "Punishment": punish,

        "Average Loss":  sum(avg_loss_history)/num_epoch,
        "Last Epoch Loss": avg_loss_history[-1],

        "Task Converge": int(np.argmax(np.flip(avg_fail_history[:, 1]) != 0)),
        "Task Fail Percentage": np.count_nonzero(avg_fail_history[:, 1])/len(avg_fail_history[:, 1]),
        "Safe Converge": int(np.argmax(np.flip(avg_fail_history[:, 2]) != 0)),
        "Safe Fail Percentage": np.count_nonzero(avg_fail_history[:, 2])/len(avg_fail_history[:, 2]),

        "Average Time": sum(avg_time_history)/num_epoch,
        "Last Epoch Time": avg_time_history[-1],

        "Average Energy": sum(avg_energy_history)/num_epoch,
        "Last Epoch Energy":  avg_energy_history[-1],

        "Average Reward":  sum(avg_reward_history)/num_epoch,
        "Last Epoch Reward": avg_reward_history[-1],

        "First 10 Avg Time": np.mean(avg_time_history[:10]),
        "Mid 10 Avg Time": np.mean(avg_time_history[half_num_epoch:half_num_epoch + 10]),
        "Last 10 Avg Time": np.mean(avg_time_history[:-10]),

        "First 10 Avg Energy": np.mean(avg_energy_history[:10]),
        "Mid 10 Avg Energy": np.mean(avg_energy_history[half_num_epoch:half_num_epoch + 10]),
        "Last 10 Avg Energy": np.mean(avg_energy_history[:-10]),

        "First 10 Avg Reward": np.mean(avg_reward_history[:10]),
        "Mid 10 Avg Reward": np.mean(avg_reward_history[half_num_epoch:half_num_epoch + 10]),
        "Last 10 Avg Reward": np.mean(avg_reward_history[:-10]),


        "First 10 Avg Loss": np.mean(avg_loss_history[:10]),
        "Mid 10 Avg Loss": np.mean(avg_loss_history[half_num_epoch:half_num_epoch + 10]),
        "Last 10 Avg Loss": np.mean(avg_loss_history[:-10]),

        "First 10 (total, task, safe) Fail": str(np.mean(avg_fail_history[:10], axis=0)),
        "Mid 10 (total, task, safe) Fail":  str(np.mean(avg_fail_history[half_num_epoch:half_num_epoch + 10], axis=0)),
        "Last 10 (total, task, safe) Fail": str(np.mean(avg_fail_history[:-10], axis=0)),
    }
    new_epoch_data_list = [new_epoch_data]

    df = None
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        new_df = pd.DataFrame(new_epoch_data_list)
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(new_epoch_data_list)
    # df.to_csv(result_path, index=False)


def plot_histories(rSetup, init_punish, punish, epsilon, init_explore_rate, explore_rate, exp_counter, lossHistory, avg_time_history, avg_energy_history, avg_fail_history, iot_usage, mec_usage, cc_usage, path_history):
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    plt.suptitle(
        f"Training History with setup {rSetup}, initial punish: {init_punish}, final punish: {punish}", fontsize=16, fontweight='bold')

    loss_values = lossHistory
    axs[0, 0].plot(loss_values, label='Average Loss',
                   color='blue', marker='o')  # Add markers for clarity
    axs[0, 0].set_title('Average Loss History')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot for average time history
    time_values = np.array(avg_time_history)  # Ensure data is in numpy array
    axs[0, 1].plot(time_values, label='Average Time', color='red', marker='o')
    axs[0, 1].set_title('Average Time History')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    time_lower_bound = 0.00625
    time_middle_bound = 0.0267
    time_upper_bound = 1
    # axs[0, 1].axhline(y=time_lower_bound, color='blue',
    #                   linestyle='--', label='Lower Bound (0.00625)')
    # axs[0, 1].axhline(y=time_middle_bound, color='green',
    #                   linestyle='--', label='Middle Bound (0.0267)')
    # axs[0, 1].axhline(y=time_upper_bound, color='red',
    #                   linestyle='--', label='Upper Bound (1)')
    axs[0, 1].legend()

    # Plot for average energy history
    energy_values = np.array(avg_energy_history)
    axs[1, 0].plot(energy_values, label='Average Energy',
                   color='green', marker='o')
    axs[1, 0].set_title('Average Energy History')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Energy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    energy_lower_bound = 0.0000405
    energy_middle_bound = 0.100746
    energy_upper_bound = 1.2
    # axs[1, 0].axhline(y=energy_lower_bound, color='blue',
    #                   linestyle='--', label='Lower Bound (0.0000405)')
    # axs[1, 0].axhline(y=energy_middle_bound, color='green',
    #                   linestyle='--', label='Middle Bound (0.100746)')
    # axs[1, 0].axhline(y=energy_upper_bound, color='red',
    #                   linestyle='--', label='Upper Bound (1.2)')
    axs[1, 0].legend()

    # Plot for average fail history
    fail_values = np.array(avg_fail_history)
    axs[1, 1].plot(fail_values, label='Average Fail',
                   color='purple', marker='o')
    axs[1, 1].set_title('Average Fail History')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Fail Count')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot for devices usage history
    axs[2, 0].plot(iot_usage, label='IoT Usage', color='blue', marker='o')
    axs[2, 0].plot(mec_usage, label='MEC Usage', color='orange', marker='x')
    axs[2, 0].plot(cc_usage, label='Cloud Usage', color='green', marker='s')
    axs[2, 0].set_title('Devices Usage History')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Usage')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Heatmap for path history
    output_classes = ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
    path_counts = np.zeros((len(path_history), len(output_classes)))

    for epoch in range(len(path_history)):
        epoch_paths = path_history[epoch]

        for path in epoch_paths:
            path_index = output_classes.index(path)
            path_counts[epoch, path_index] += 1

    sns.heatmap(path_counts, cmap="YlGnBu",
                xticklabels=output_classes, ax=axs[2, 1])
    axs[2, 1].set_title(
        f'Path History Heatmap - All Epochs\n(r: {rSetup}, p: {init_punish}, ep: {epsilon}, exp_rate: {init_explore_rate:.5f} - {explore_rate:.5f}, exp_times: {exp_counter})')
    axs[2, 1].set_xlabel('Output Classes')
    axs[2, 1].set_ylabel('Epochs')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f"./results/Energy Figs/r{rSetup}_p{init_punish}")
    plt.show()
