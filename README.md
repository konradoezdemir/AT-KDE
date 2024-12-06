# README: AT-KDE

Code Base for Paper "A Divide-and-Conquer Approach for Modeling Arrival Times in Dynamic Environments" by L. Kirchdorfer and K. Özdemir et al. (2024)

## Introduction

**Abstract**

Simulation is a critical tool for analyzing, improving, and redesigning organizational processes and their supporting information systems. A key component of simulation is the case-arrival model, which determines the pattern of new case entries into a process. Accurate case-arrival modeling is essential for reliable simulations, yet existing approaches often rely on oversimplified static distributions of inter-arrival times. These approaches fail to capture the dynamic and temporal complexities inherent in organizational environments, leading to less accurate and reliable outcomes. To address this limitation, we propose \emph{Auto Time Kernel Density Estimation} (AT-KDE), a divide-and-conquer approach that models arrival times by incorporating global dynamics, day-of-week variations, and intraday distributional changes, ensuring both precision and scalability. Experiments conducted across 20 diverse processes demonstrate that AT-KDE is considerably more accurate and robust than existing approaches while maintaining runtime efficiency.

**Key features:**

- **AT-KDE (Auto Time Kernel Density Estimation):** A divide-and-conquer method that identifies and models multiple temporal scales (global, weekly, and intraday) to produce highly accurate and scalable arrival time simulations.
- **Improved Modeling Techniques:** In addition to AT-KDE, the code supports other baseline methods, including mean, exponential, Prophet-based, and standard KDE models. This allows for easy benchmarking and comparison.
- **Robust and Efficient:** Tested on 20 diverse processes, AT-KDE significantly improves accuracy and reliability compared to existing approaches, without compromising computational efficiency.

**Process Data Overview:**
| Event log           | Traces  | Time range (d.) | Min Arrivals per day | Mean Arrivals per day | Max Arrivals per day |
|---------------------|---------|-----------------|----------------------|-----------------------|----------------------|
| BPIC12              | 13087   | 151             | 17                   | 86.1                  | 233                  |
| BPIC12CW            | 9658    | 152             | 0                    | 63.1                  | 155                  |
| BPIC12O             | 5015    | 165             | 0                    | 30.21                 | 78                   |
| BPIC12W             | 9658    | 151             | 13                   | 63.5                  | 154                  |
| BPIC13C             | 1487    | 2332            | 0                    | 0.6                   | 15                   |
| BPIC17W             | 30276   | 369             | 0                    | 81.8                  | 288                  |
| BPIC19              | 251734  | 25923           | 0                    | 9.7                   | 1538                 |
| BPIC20D             | 10500   | 719             | 0                    | 14.6                  | 73                   |
| BPIC20I             | 6449    | 800             | 0                    | 8.1                   | 56                   |
| BPIC20Permit        | 7065    | 816             | 0                    | 8.7                   | 43                   |
| Env.permit          | 1434    | 478             | 0                    | 3.0                   | 28                   |
| Helpdesk            | 4580    | 1415            | 0                    | 3.2                   | 22                   |
| Hospital            | 99999   | 1130            | 3                    | 88.4                  | 132                  |
| Sepsis              | 1049    | 476             | 0                    | 2.2                   | 11                   |
| P2P                 | 608     | 285             | 0                    | 2.2                   | 12                   |
| CVS                 | 10000   | 63              | 0                    | 156.25                | 253                  |
| Confidential 1000   | 1000    | 164             | 0                    | 6.1                   | 18                   |
| Confidential 2000   | 2000    | 358             | 0                    | 5.6                   | 16                   |
| ACR                 | 954     | 148             | 0                    | 6.4                   | 26                   |
| Production          | 225     | 88              | 0                    | 2.5                   | 11                   |

All the event logs can be found in this [Google Drive folder](https://drive.google.com/file/d/1abPg1txA6P9jyNfmhJel2kZNCP0FdmAW/view?usp=sharing).

## Repository Structure

- **`generate_arrivals.py`**  
  The main script for simulating arrival times. Users can configure different methods for estimating inter-arrival times, including AT-KDE.
  
- **`source/iat_approaches/`**  
  Contains various inter-arrival time modeling strategies, such as the AT-KDE generator and other baseline methods.
  
- **`utils/`**  
  Utility functions for data handling, timestamp transformations, and other common operations.
  
- **`data/event_logs/`**  
  Directory for storing input datasets/event logs. Place your event logs or synthetic datasets here before running simulations.

- **`event_log_simulations/`**  
  Stores simulation results based on the input event logs.

- **`kde_core/`**  
  Core components related to kernel density estimation, which are integral to the AT-KDE methodology.

- **`run_shells/`**  
  Shell scripts for automating the simulation runs.

- **`diagnostics/`**  
  Contains tools for analyzing and visualizing the output of simulations, aiding in the evaluation of model performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/konradoezdemir/AT-KDE.git

2. Create and activate a virtual environment and install dependencies:
   ```bash
    python -m venv atkde_env
    source atkde_env/bin/activate  #for linux/mac
    .\atkde_env\Scripts\activate  #for windows 

    pip install -r requirements.txt
## Usage Example: Single Data Simulation
This command will use the AT-KDE approach to model inter-arrival times based on the provided event log, creating a highly dynamic and precise simulation of case arrivals.
```bash
python generate_arrivals.py --input_type event_log --dataset BPI_Challenge_2012CW --method kde --run 1
```
## Usage Example: Multi/Comparison Data Simulation
This command will run all benchmark models and AT-KDE on a selection of datasets in the data directory. Results are stored as an .xlsx reporting root-cadd scores and a jpeg with arrival count plot comparisons.
```bash
cd run_shells
./run_all.ps1 #for windows 
sh run_all.sh #for linux/mac
```
