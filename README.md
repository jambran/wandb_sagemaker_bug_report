# wandb-sagemaker-issue

Minimal code to reproduce an issue with distributed training on 
sagemaker and `wandb`.

## Summary
I've been following the 
[documenation](https://docs.wandb.ai/guides/track/advanced/distributed-training) 
regarding parallel training. I'd like to follow method 2: `wandb.init()` 
on all processes.

I'm trying to train on sagemaker, but I can't get a 
successful training job to complete. 
I can remove the `wandb` logging code, replacing 
the `report_to` to `tensorboard`, and the training job completes successfully.

The error message coming from `wandb` is a bit uninformative.
```
[1,3]<stderr>:  File "train.py", line 75, in <module>
[1,3]<stderr>:    group="DDP",
[1,3]<stderr>:  File "/opt/conda/lib/python3.6/site-packages/wandb/sdk/wandb_init.py", line 846, in init
[1,3]<stderr>:    six.raise_from(Exception("problem"), error_seen)
[1,3]<stderr>:  File "<string>", line 3, in raise_from
[1,3]<stderr>:Exception: problem
```

## Steps to recreate

1. Install the environment
   ```
   python3 -m venv venv_wandb_sagemaker_issue
   source venv_wandb_sagemaker_issue/bin/activate
   pip install -r requirements.txt
   ```
2. Configure your sagemaker environment
3. run `train_sagemaker.py`
4. The output of this script can be found in `error_message.log`