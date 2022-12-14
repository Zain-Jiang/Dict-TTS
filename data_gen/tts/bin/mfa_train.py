import subprocess
from utils.hparams import hparams, set_hparams
import os


def mfa_train():
    CORPUS = hparams['processed_data_dir'].split("/")[-1]
    print(f"| Run MFA for {CORPUS}.")
    NUM_JOB = int(os.getenv('N_PROC', os.cpu_count()))
    subprocess.check_call(
        f'CORPUS={CORPUS} NUM_JOB={NUM_JOB} MFA_VERSION={hparams["mfa_version"]} '
        f'bash scripts/run_mfa_train.sh',
        shell=True)


if __name__ == '__main__':
    set_hparams(print_hparams=False)
    mfa_train()
