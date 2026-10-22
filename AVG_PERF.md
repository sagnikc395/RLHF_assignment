AVG PERF RLHF POLICY 
cumulative return 200.0
cumulative return 140.0
cumulative return 188.0
cumulative return 153.0
cumulative return 134.0
average return 163.0


AVG PERF SYNTHETIC POLICY 
cumulative return 115.0
cumulative return 60.0
cumulative return 17.0
cumulative return 68.0
cumulative return 86.0
average return 69.2

Code for the avg plot across all 

```shell
uv run rollout_policy.py --plot_dir ./rlhf --env CartPole-v1 --num_rollouts 5
```
