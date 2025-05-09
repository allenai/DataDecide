DATA_DECIDE_MODEL_NAMES = [
    'baseline-150M-5xC-2', 
    'baseline-1B-5xC-2', 
    'baseline-300M-5xC-2', 
    'baseline-530M-5xC-2', 
    'c4-150M-5xC-2', 
    'c4-1B-5xC-2', 
    'c4-20M-5xC', 
    'c4-300M-5xC-2', 
    'c4-4M-5xC', 
    'c4-530M-5xC-2', 
    'c4-60M-5xC', 
    'c4-750M-5xC-2', 
    'c4-90M-5xC', 
    'dclm_ft7percentile_fw2-150M-5xC-2', 
    'dclm_ft7percentile_fw2-1B-5xC-2', 
    'dclm_ft7percentile_fw2-20M-5xC', 
    'dclm_ft7percentile_fw2-300M-5xC-2', 
    'dclm_ft7percentile_fw2-4M-5xC', 
    'dclm_ft7percentile_fw2-530M-5xC-2', 
    'dclm_ft7percentile_fw2-60M-5xC', 
    'dclm_ft7percentile_fw2-750M-5xC-2', 
    'dclm_ft7percentile_fw2-90M-5xC', 
    'dclm_ft7percentile_fw3-150M-5xC-2', 
    'dclm_ft7percentile_fw3-1B-5xC-2', 
    'dclm_ft7percentile_fw3-20M-5xC', 
    'dclm_ft7percentile_fw3-300M-5xC-2', 
    'dclm_ft7percentile_fw3-4M-5xC', 
    'dclm_ft7percentile_fw3-530M-5xC-2', 
    'dclm_ft7percentile_fw3-60M-5xC', 
    'dclm_ft7percentile_fw3-750M-5xC-2', 
    'dclm_ft7percentile_fw3-90M-5xC', 
    'dclm_fw_top10-150M-5xC', 
    'dclm_fw_top10-1B-5xC', 
    'dclm_fw_top10-20M-5xC', 
    'dclm_fw_top10-300M-5xC', 
    'dclm_fw_top10-4M-5xC', 
    'dclm_fw_top10-530M-5xC', 
    'dclm_fw_top10-60M-5xC', 
    'dclm_fw_top10-750M-5xC', 
    'dclm_fw_top10-90M-5xC', 
    'dclm_fw_top3-150M-5xC-2', 
    'dclm_fw_top3-1B-5xC', 
    'dclm_fw_top3-20M-5xC', 
    'dclm_fw_top3-300M-5xC-2', 
    'dclm_fw_top3-4M-5xC', 
    'dclm_fw_top3-530M-5xC-2', 
    'dclm_fw_top3-60M-5xC', 
    'dclm_fw_top3-750M-5xC-2', 
    'dclm_fw_top3-90M-5xC', 
    'DCLM-baseline-150M-5xC-2', 
    'DCLM-baseline-1B-5xC-2', 
    'DCLM-baseline-20M-5xC', 
    'DCLM-baseline-300M-5xC-2', 
    'DCLM-baseline-4M-5xC', 
    'DCLM-baseline-530M-5xC-2', 
    'DCLM-baseline-60M-5xC', 
    'DCLM-baseline-750M-5xC-2', 
    'DCLM-baseline-90M-5xC', 
    'dolma-v1-6-and-sources-baseline-150M-5xC-2', 
    'dolma-v1-6-and-sources-baseline-1B-5xC-2', 
    'dolma-v1-6-and-sources-baseline-20M-5xC', 
    'dolma-v1-6-and-sources-baseline-300M-5xC-2', 
    'dolma-v1-6-and-sources-baseline-4M-5xC', 
    'dolma-v1-6-and-sources-baseline-530M-5xC-2', 
    'dolma-v1-6-and-sources-baseline-60M-5xC', 
    'dolma-v1-6-and-sources-baseline-750M-5xC-2', 
    'dolma-v1-6-and-sources-baseline-90M-5xC', 
    'dolma17-20M-5xC', 
    'dolma17-25p-DCLM-baseline-75p-150M-5xC-2', 
    'dolma17-25p-DCLM-baseline-75p-1B-5xC-2', 
    'dolma17-25p-DCLM-baseline-75p-20M-5xC', 
    'dolma17-25p-DCLM-baseline-75p-300M-5xC-2', 
    'dolma17-25p-DCLM-baseline-75p-4M-5xC', 
    'dolma17-25p-DCLM-baseline-75p-530M-5xC-2', 
    'dolma17-25p-DCLM-baseline-75p-60M-5xC', 
    'dolma17-25p-DCLM-baseline-75p-750M-5xC-2', 
    'dolma17-25p-DCLM-baseline-75p-90M-5xC', 
    'dolma17-4M-5xC', 
    'dolma17-50p-DCLM-baseline-50p-150M-5xC-2', 
    'dolma17-50p-DCLM-baseline-50p-1B-5xC-2', 
    'dolma17-50p-DCLM-baseline-50p-20M-5xC', 
    'dolma17-50p-DCLM-baseline-50p-300M-5xC-2', 
    'dolma17-50p-DCLM-baseline-50p-4M-5xC', 
    'dolma17-50p-DCLM-baseline-50p-530M-5xC-2', 
    'dolma17-50p-DCLM-baseline-50p-60M-5xC', 
    'dolma17-50p-DCLM-baseline-50p-750M-5xC-2', 
    'dolma17-50p-DCLM-baseline-50p-90M-5xC', 
    'dolma17-60M-5xC', 
    'dolma17-750M-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-150M-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-1B-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-20M-5xC', 
    'dolma17-75p-DCLM-baseline-25p-300M-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-4M-5xC', 
    'dolma17-75p-DCLM-baseline-25p-530M-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-60M-5xC', 
    'dolma17-75p-DCLM-baseline-25p-750M-5xC-2', 
    'dolma17-75p-DCLM-baseline-25p-90M-5xC', 
    'dolma17-90M-5xC', 
    'falcon_and_cc_eli5_oh_top10p-150M-5xC-2', 
    'falcon_and_cc_eli5_oh_top10p-1B-5xC-2', 
    'falcon_and_cc_eli5_oh_top10p-20M-5xC', 
    'falcon_and_cc_eli5_oh_top10p-300M-5xC-2', 
    'falcon_and_cc_eli5_oh_top10p-4M-5xC', 
    'falcon_and_cc_eli5_oh_top10p-530M-5xC-2', 
    'falcon_and_cc_eli5_oh_top10p-60M-5xC', 
    'falcon_and_cc_eli5_oh_top10p-750M-5xC-2', 
    'falcon_and_cc_eli5_oh_top10p-90M-5xC', 
    'falcon_and_cc_eli5_oh_top20p-150M-5xC-2', 
    'falcon_and_cc_eli5_oh_top20p-1B-5xC-2', 
    'falcon_and_cc_eli5_oh_top20p-20M-5xC', 
    'falcon_and_cc_eli5_oh_top20p-300M-5xC-2', 
    'falcon_and_cc_eli5_oh_top20p-4M-5xC', 
    'falcon_and_cc_eli5_oh_top20p-530M-5xC-2', 
    'falcon_and_cc_eli5_oh_top20p-60M-5xC', 
    'falcon_and_cc_eli5_oh_top20p-750M-5xC-2', 
    'falcon_and_cc_eli5_oh_top20p-90M-5xC', 
    'falcon_and_cc_og_eli5_oh_top10p-150M-5xC-2', 
    'falcon_and_cc_og_eli5_oh_top10p-1B-5xC-2', 
    'falcon_and_cc_og_eli5_oh_top10p-20M-5xC', 
    'falcon_and_cc_og_eli5_oh_top10p-300M-5xC-2', 
    'falcon_and_cc_og_eli5_oh_top10p-4M-5xC', 
    'falcon_and_cc_og_eli5_oh_top10p-530M-5xC-2', 
    'falcon_and_cc_og_eli5_oh_top10p-60M-5xC', 
    'falcon_and_cc_og_eli5_oh_top10p-750M-5xC-2', 
    'falcon_and_cc_og_eli5_oh_top10p-90M-5xC', 
    'falcon_and_cc_tulu_qc_top10-150M-5xC-2', 
    'falcon_and_cc_tulu_qc_top10-1B-5xC-2', 
    'falcon_and_cc_tulu_qc_top10-20M-5xC', 
    'falcon_and_cc_tulu_qc_top10-300M-5xC-2', 
    'falcon_and_cc_tulu_qc_top10-4M-5xC', 
    'falcon_and_cc_tulu_qc_top10-530M-5xC-2', 
    'falcon_and_cc_tulu_qc_top10-60M-5xC', 
    'falcon_and_cc_tulu_qc_top10-750M-5xC-2', 
    'falcon_and_cc_tulu_qc_top10-90M-5xC', 
    'falcon_and_cc-150M-5xC-2', 
    'falcon_and_cc-1B-5xC-2', 
    'falcon_and_cc-20M-5xC', 
    'falcon_and_cc-300M-5xC-2', 
    'falcon_and_cc-4M-5xC', 
    'falcon_and_cc-530M-5xC-2', 
    'falcon_and_cc-60M-5xC', 
    'falcon_and_cc-750M-5xC-2', 
    'falcon_and_cc-90M-5xC', 
    'falcon-150M-5xC-2', 
    'falcon-1B-5xC-2', 
    'falcon-20M-5xC', 
    'falcon-300M-5xC-2', 
    'falcon-4M-5xC', 
    'falcon-530M-5xC-2', 
    'falcon-60M-5xC', 
    'falcon-750M-5xC-2', 
    'falcon-90M-5xC', 
    'fineweb_edu_dedup-150M-5xC-2', 
    'fineweb_edu_dedup-1B-5xC-2', 
    'fineweb_edu_dedup-20M-5xC', 
    'fineweb_edu_dedup-300M-5xC-2', 
    'fineweb_edu_dedup-4M-5xC', 
    'fineweb_edu_dedup-530M-5xC-2', 
    'fineweb_edu_dedup-60M-5xC', 
    'fineweb_edu_dedup-750M-5xC-2', 
    'fineweb_edu_dedup-90M-5xC', 
    'no_code-150M-5xC-2', 
    'no_code-1B-5xC-2', 
    'no_code-20M-5xC', 
    'no_code-300M-5xC-2', 
    'no_code-4M-5xC', 
    'no_code-530M-5xC-2', 
    'no_code-60M-5xC', 
    'no_code-750M-5xC-2', 
    'no_code-90M-5xC', 
    'no_flan-150M-5xC-2', 
    'no_flan-1B-5xC-2', 
    'no_flan-20M-5xC', 
    'no_flan-300M-5xC-2', 
    'no_flan-4M-5xC', 
    'no_flan-530M-5xC-2', 
    'no_flan-60M-5xC', 
    'no_flan-750M-5xC-2', 
    'no_flan-90M-5xC', 
    'no_math_no_code-150M-5xC-2', 
    'no_math_no_code-1B-5xC-2', 
    'no_math_no_code-20M-5xC', 
    'no_math_no_code-300M-5xC-2', 
    'no_math_no_code-4M-5xC', 
    'no_math_no_code-530M-5xC-2', 
    'no_math_no_code-60M-5xC', 
    'no_math_no_code-750M-5xC-2', 
    'no_math_no_code-90M-5xC', 
    'no_reddit-150M-5xC-2', 
    'no_reddit-1B-5xC-2', 
    'no_reddit-20M-5xC', 
    'no_reddit-300M-5xC-2', 
    'no_reddit-4M-5xC', 
    'no_reddit-530M-5xC-2', 
    'no_reddit-60M-5xC', 
    'no_reddit-750M-5xC-2', 
    'no_reddit-90M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-150M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-1B-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-20M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-300M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-4M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-530M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-60M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-750M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p-90M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-150M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-1B-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-20M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-300M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-4M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-530M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-60M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-750M-5xC', 
    'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p-90M-5xC', 
    'prox_fineweb_pro-150M-5xC-2', 
    'prox_fineweb_pro-1B-5xC-2', 
    'prox_fineweb_pro-20M-5xC', 
    'prox_fineweb_pro-300M-5xC-2', 
    'prox_fineweb_pro-4M-5xC', 
    'prox_fineweb_pro-530M-5xC-2', 
    'prox_fineweb_pro-60M-5xC', 
    'prox_fineweb_pro-750M-5xC-2', 
    'prox_fineweb_pro-90M-5xC'
]

DATA_DECIDE_SIZES = ['4M', '6M', '8M', '10M', '14M', '16M', '20M', '60M', '90M', '150M', '300M', '530M', '750M', '1B']

FULL_SCHEDULE = {
    '4M': 5725,
    '6M': 9182,
    '8M': 13039,
    '10M': 15117,
    '14M': 21953,
    '16M': 24432,
    '20M': 14584,
    '60M': 29042,
    '90M': 29901,
    '150M': 38157,
    '300M': 45787,
    '530M': 57786,
    '750M': 63589,
    '1B': 69369,
}

MODEL_TO_BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1_176_832_000
}

def get_compute(scale):
    return 2048 * 6 * MODEL_TO_BATCH[scale] * MODEL_TO_PARAMETERS[scale] * FULL_SCHEDULE[scale]

DATA_DECIDE_COMPUTE_SIZES = tuple(get_compute(size) for size in DATA_DECIDE_SIZES)
