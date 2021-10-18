import logging
import sagemaker
from getpass import getuser
from sagemaker.huggingface.estimator import HuggingFace

"""
users running this script will need to update the following variables
in a file named `config.py`
`role`
`s3_output_location`

"""
import config

if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler()],
                        level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        )

    sess = sagemaker.Session()
    exp_name = 'debug-wandb-sagemaker-distributed-training'
    role = config.role

    logging.info(f"sagemaker role arn: {role}")
    logging.info(f"sagemaker bucket: {sess.default_bucket()}")
    logging.info(f"sagemaker session region: {sess.boto_region_name}")

    s3_output_location = config.s3_output_location

    hyperparameters = {
        'epochs': 1,
        'train_batch_size': 8,
        'eval_batch_size': 2,
        'warmup_steps': 500,
        'model_id': 'bert-base-uncased',
        'learning_rate': 5e-5,
        'fp16': True,
        'dataset': './data/dataset',
        'output_data_dir': s3_output_location,
        'model_dir': '/opt/ml/model',
        'max_grad_norm': 0,
    }
    mpi_options = {
        "enabled": True,
        "processes_per_host": 4,
    }
    smp_options = {
        "enabled": True,
        "parameters": {
            "microbatches": 4,
            "placement_strategy": "spread",
            "pipeline": "interleaved",
            "optimize": "speed",
            "partitions": 4,
            "ddp": True,
        }
    }
    distribution = {
        "smdistributed": {"modelparallel": smp_options},
        "mpi": mpi_options
    }

    logging.info('setting up the estimator')
    # tag the job with the appropriate info for easier resource analysis later
    user = getuser()
    tag_list = {
        "App": "core-infra",
        "Owner": f"whiq@workhuman.com",
        "Team": "whiq",
        "Person": user,
    }

    tags = [{'Key': key,
             'Value': value}
            for key, value in tag_list.items()]

    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        base_job_name=exp_name,
        instance_type='ml.g4dn.12xlarge',
        instance_count=1,
        volume_size=30,
        image_uri=config.image_uri,
        py_version='py36',
        role=role,
        sagemaker_session=sess,
        hyperparameters=hyperparameters,
        dependencies=['requirements.txt',
                      'data',
                      'secrets.env',
                      ],
        output_path=s3_output_location,
        code_location=s3_output_location,
        tags=tags,
        distribution=distribution,
    )
    logging.info('estimator instantiated')

    logging.info('fitting the estimator')
    huggingface_estimator.fit()
    logging.info('estimator fit')