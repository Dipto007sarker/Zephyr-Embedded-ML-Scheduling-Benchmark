# Zephyr RTOS Embedded ML Scheduling Benchmark

This repository contains an experimental benchmark for studying how Zephyr RTOS scheduling configurations behave when real-time (RT) tasks execute alongside embedded machine learning (ML) inference workloads.

The project evaluates whether different scheduler configurations can preserve RT task timing under increasing ML workload pressure. The experiments are designed around Zephyr RTOS, TensorFlow Lite Micro (TFLM), and QEMU/native simulation environments.

## Project Goal

Embedded devices increasingly run ML inference beside time-sensitive control tasks. However, ML inference can be computationally bursty and may interfere with periodic RT tasks. This project studies that interaction by measuring RT lateness, RT deadline misses, RT activation gaps, and ML execution cycles under different scheduler configurations.

The main research question is:

> How does Zephyr RTOS scheduling behavior change when embedded ML workload realism increases?

## Experimental Design

The benchmark uses three concurrent Zephyr threads:

| Thread | Purpose |
|---|---|
| ML thread | Runs synthetic computation or TFLM inference workload |
| RT thread | Periodic real-time task with timing measurements |
| Background thread | Adds CPU contention through a low-priority loop |

The RT task uses a fixed period of 200 ms. A deadline miss is counted when RT lateness exceeds one full period.

## Scheduling Configurations

Three scheduling configurations are evaluated.

| Configuration | RT Priority | ML Priority | Deadline Scheduling |
|---|---:|---:|---|
| `PRIO-NODL` | 2 | 4 | Disabled |
| `EQ-NODL` | 4 | 4 | Disabled |
| `EQ-DL` | 4 | 4 | Enabled, RT = 10, ML = 30 |

In Zephyr, a lower numerical priority value means a higher scheduling priority. Therefore, in `PRIO-NODL`, the RT task has higher priority than the ML task.

In `EQ-DL`, both RT and ML tasks have equal static priority, but Zephyr deadline-aware scheduling is enabled. The RT task is assigned an earlier deadline value than the ML task.

## Workload Phases

The full study uses multiple workload phases:

| Phase | Workload |
|---|---|
| Phase 1 | Synthetic floating-point ML-like workload |
| Phase 2 | TensorFlow Lite Micro Hello World inference |
| Phase 3 | Repeated Hello World inference |
| Phase 4 | TensorFlow Lite Micro Magic Wand inference |
| Phase 5 | Adaptive runtime protection mechanism |

This README focuses on the setup and Phase 1 workflow. Later phases follow the same structure with different ML workload implementations.

## Environment

The project was developed using:

- Windows with WSL Ubuntu
- Zephyr RTOS
- Zephyr SDK
- QEMU
- Python virtual environment
- West build system

## Setup Instructions

### 1. Open WSL

Open Windows Terminal as Administrator and run:

```bash
wsl -d Ubuntu
```

Go to the home directory:

```bash
cd ~
```

Check the current user and directory:

```bash
whoami
pwd
```

## Install Required System Packages

```bash
sudo apt update

sudo apt install -y git cmake ninja-build gperf \
  ccache dfu-util device-tree-compiler wget \
  python3-dev python3-pip python3-setuptools python3-tk python3-venv \
  xz-utils file make gcc gcc-multilib g++-multilib libsdl2-dev
```

## Create Zephyr Workspace

```bash
mkdir -p ~/zephyrproject
cd ~/zephyrproject
```

## Create Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install west
```

Verify that Python and West point inside the virtual environment:

```bash
which python
which west
```

Expected paths should look similar to:

```text
/home/<user>/zephyrproject/.venv/bin/python
/home/<user>/zephyrproject/.venv/bin/west
```

## Initialize Zephyr and Fetch Sources

```bash
cd ~/zephyrproject
west init -m https://github.com/zephyrproject-rtos/zephyr.git
west update
west zephyr-export
pip install -r zephyr/scripts/requirements.txt
```

## Install Zephyr SDK

The Zephyr SDK version must match the installed Zephyr tree. If SDK `0.17.2` causes a version mismatch, use SDK `1.0.1`.

```bash
cd ~

wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v1.0.1/zephyr-sdk-1.0.1_linux-x86_64_gnu.tar.xz

tar xvf zephyr-sdk-1.0.1_linux-x86_64_gnu.tar.xz

cd zephyr-sdk-1.0.1
./setup.sh
```

Point Zephyr to the SDK:

```bash
export ZEPHYR_SDK_INSTALL_DIR=$HOME/zephyr-sdk-1.0.1
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
```

To make this persistent:

```bash
echo 'export ZEPHYR_SDK_INSTALL_DIR=$HOME/zephyr-sdk-1.0.1' >> ~/.bashrc
echo 'export ZEPHYR_TOOLCHAIN_VARIANT=zephyr' >> ~/.bashrc
source ~/.bashrc
```

## Test Zephyr Installation

Build and run the Zephyr Hello World sample:

```bash
cd ~/zephyrproject
source .venv/bin/activate
export ZEPHYR_SDK_INSTALL_DIR=$HOME/zephyr-sdk-1.0.1
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr

cd ~/zephyrproject/zephyr
west build -b native_sim samples/hello_world -p always
west build -t run
```

Expected output:

```text
*** Booting Zephyr OS build ...
Hello World! native_sim/native
```

`native_sim` keeps running until interrupted. Exit with:

```text
Ctrl+C
```

## TensorFlow Lite Micro Setup

Enable the TFLM module in the West manifest:

```bash
cd ~/zephyrproject
west config manifest.project-filter -- +tflite-micro
west update
```

The TFLM module may be fetched into:

```text
~/zephyrproject/optional/modules/lib/tflite-micro
```

Verify it exists:

```bash
ls ~/zephyrproject/optional/modules/lib/tflite-micro
```

Build and run the TFLM Hello World sample:

```bash
cd ~/zephyrproject/zephyr
west build -b qemu_x86 samples/modules/tflite-micro/hello_world -p always
west build -t run
```

Expected output should include values such as:

```text
x_value: ...
y_value: ...
```

Exit QEMU using:

```text
Ctrl+A then X
```

## Project Directory Structure

Create the custom experiment app:

```bash
cd ~/zephyrproject
mkdir -p app/zephyr_ml_sched/phase1/src
cd app/zephyr_ml_sched/phase1
```

Recommended repository structure:

```text
zephyr_ml_sched/
├── phase1/
│   ├── CMakeLists.txt
│   ├── prj.conf
│   ├── src/
│   │   └── main.c
│   └── results/
│       ├── configuration_1/
│       ├── configuration_2/
│       └── configuration_3/
├── scripts/
│   └── parse_zephyr_logs.py
└── README.md
```

## Phase 1: Synthetic Workload

Phase 1 uses a synthetic floating-point loop to simulate ML-like computation pressure.

### `CMakeLists.txt`

Create `CMakeLists.txt` inside `phase1`:

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(zephyr_ml_sched)

target_sources(app PRIVATE src/main.c)
```

### `prj.conf`

Create `prj.conf` inside `phase1`:

```conf
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_HEAP_MEM_POOL_SIZE=16384

CONFIG_STDOUT_CONSOLE=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
CONFIG_SERIAL=y

CONFIG_TIMESLICING=y
CONFIG_TIMESLICE_SIZE=2

CONFIG_SCHED_DEADLINE=y
CONFIG_TIMING_FUNCTIONS=y

CONFIG_THREAD_NAME=y
CONFIG_INIT_STACKS=y
```

### `src/main.c`

Create `src/main.c` and paste the benchmark source code.

The benchmark creates:

- `ml_task`
- `rt_task`
- `bg_task`

The RT task logs:

```text
[RT] run=<n> exec_ms=<ms> lateness_ms=<ms> gap_ms=<ms> misses=<n>
```

The ML task logs:

```text
[ML] run=<n> cycles=<cycles>
```

The summary log prints:

```text
[SUMMARY] ml_runs=<n> rt_runs=<n> rt_misses=<n>
```

## Running Phase 1 Experiments

Activate the Python virtual environment and set Zephyr paths:

```bash
source ~/zephyrproject/.venv/bin/activate
export ZEPHYR_BASE=~/zephyrproject/zephyr
export ZEPHYR_SDK_INSTALL_DIR=$HOME/zephyr-sdk-1.0.1
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
```

Go to the Phase 1 app:

```bash
cd ~/zephyrproject/app/zephyr_ml_sched/phase1
```

Create result folders:

```bash
mkdir -p results/configuration_1
mkdir -p results/configuration_2
mkdir -p results/configuration_3
```

## Configuration 1: Priority Baseline, No Deadline

Settings in `main.c`:

```c
#define ML_PRIO 4
#define RT_PRIO 2
#define BG_PRIO 6
```

Deadline calls should be disabled:

```c
// k_thread_deadline_set(ml_tid, 30);
// k_thread_deadline_set(rt_tid, 10);
```

Build:

```bash
west build -b qemu_x86 . -p always
```

Run five trials:

```bash
timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_1/run_priority_baseline_no_deadline_1.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_1/run_priority_baseline_no_deadline_2.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_1/run_priority_baseline_no_deadline_3.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_1/run_priority_baseline_no_deadline_4.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_1/run_priority_baseline_no_deadline_5.txt'
```

## Configuration 2: Equal Priority, No Deadline

Settings in `main.c`:

```c
#define ML_PRIO 4
#define RT_PRIO 4
#define BG_PRIO 6
```

Deadline calls should remain disabled:

```c
// k_thread_deadline_set(ml_tid, 30);
// k_thread_deadline_set(rt_tid, 10);
```

Rebuild:

```bash
west build -b qemu_x86 . -p always
```

Run five trials:

```bash
timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_2/run_equal_priority_baseline_no_deadline_1.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_2/run_equal_priority_baseline_no_deadline_2.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_2/run_equal_priority_baseline_no_deadline_3.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_2/run_equal_priority_baseline_no_deadline_4.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_2/run_equal_priority_baseline_no_deadline_5.txt'
```

## Configuration 3: Equal Priority, With Deadline

Settings in `main.c`:

```c
#define ML_PRIO 4
#define RT_PRIO 4
#define BG_PRIO 6
```

Deadline calls should be enabled:

```c
k_thread_deadline_set(ml_tid, 30);
k_thread_deadline_set(rt_tid, 10);
```

Rebuild:

```bash
west build -b qemu_x86 . -p always
```

Run five trials:

```bash
timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_3/run_equal_priority_baseline_with_deadline_1.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_3/run_equal_priority_baseline_with_deadline_2.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_3/run_equal_priority_baseline_with_deadline_3.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_3/run_equal_priority_baseline_with_deadline_4.txt'

timeout -k 2s 15s bash -c 'west build -t run | tee results/configuration_3/run_equal_priority_baseline_with_deadline_5.txt'
```

## Parsing Logs

Install Python dependencies:

```bash
pip install pandas
```

Run the parser for each configuration folder:

```bash
cd ~/zephyrproject/app/zephyr_ml_sched/phase1/results/configuration_1
python3 ../../../scripts/parse_zephyr_logs.py --input-dir . --pattern "run_*.txt"

cd ../configuration_2
python3 ../../../scripts/parse_zephyr_logs.py --input-dir . --pattern "run_*.txt"

cd ../configuration_3
python3 ../../../scripts/parse_zephyr_logs.py --input-dir . --pattern "run_*.txt"
```

Each run produces:

```text
zephyr_trial_summary.csv
zephyr_config_summary.csv
```

## Merging Results

After parsing each configuration, merge the per-trial and per-config CSV files into combined summary files.

Example:

```bash
cd ~/zephyrproject/app/zephyr_ml_sched/phase1/results

python3 - <<'PY'
import pandas as pd
from pathlib import Path

configs = ["configuration_1", "configuration_2", "configuration_3"]

trial_dfs = []
config_dfs = []

for c in configs:
    trial_path = Path(c) / "zephyr_trial_summary.csv"
    config_path = Path(c) / "zephyr_config_summary.csv"

    if trial_path.exists():
        df = pd.read_csv(trial_path)
        df["folder"] = c
        trial_dfs.append(df)

    if config_path.exists():
        df = pd.read_csv(config_path)
        df["folder"] = c
        config_dfs.append(df)

pd.concat(trial_dfs, ignore_index=True).to_csv("combined_trial_summary.csv", index=False)
pd.concat(config_dfs, ignore_index=True).to_csv("combined_config_summary.csv", index=False)

print("Wrote combined_trial_summary.csv")
print("Wrote combined_config_summary.csv")
PY
```

## Output Files

Typical output files include:

```text
run_*.txt
zephyr_trial_summary.csv
zephyr_config_summary.csv
combined_trial_summary.csv
combined_config_summary.csv
```

The main metrics are:

| Metric | Meaning |
|---|---|
| `final_ml_runs` | Number of completed ML task iterations |
| `final_rt_runs` | Number of completed RT task activations |
| `final_rt_misses` | Number of RT deadline misses |
| `avg_rt_exec_ms` | Average RT task execution time |
| `avg_rt_lateness_ms` | Average RT task lateness |
| `max_rt_lateness_ms` | Maximum RT task lateness |
| `avg_rt_gap_ms` | Average time gap between RT activations |
| `max_rt_gap_ms` | Maximum time gap between RT activations |
| `avg_ml_cycles` | Average ML task cycle count |
| `max_ml_cycles` | Maximum ML task cycle count |

## Notes

- QEMU runs continuously until manually stopped or killed by `timeout`.
- Exit QEMU manually with `Ctrl+A`, then `X`.
- If `native_sim` appears stuck, it is usually still running normally.
- If SDK version mismatch occurs, install a Zephyr SDK version compatible with the checked-out Zephyr tree.
- The TFLM module may be located under `optional/modules/lib/tflite-micro`, not `modules/lib/tflite-micro`.

## Reproducibility

For each scheduling configuration:

1. Update priority and deadline settings in `main.c`.
2. Rebuild with `west build -b qemu_x86 . -p always`.
3. Run five 15-second trials using `timeout`.
4. Save logs under the corresponding result folder.
5. Parse logs using `parse_zephyr_logs.py`.
6. Merge trial and configuration summaries.
7. Use the resulting CSV files for analysis and plotting.

## Citation / Acknowledgment

This project was developed for a model analysis and simulation study on Zephyr RTOS scheduling behavior under embedded ML workload pressure.
