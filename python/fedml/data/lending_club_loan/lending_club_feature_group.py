qualification_feat = [
    "grade",
    # 'sub_grade',
    # 'emp_title',
    "emp_length",
    "home_ownership",
    "annual_inc_comp",
    # 'annual_inc',
    # 'annual_inc_joint',
    "verification_status",
    # 'verification_status_joint',
    # 'issue_d',
    # 'earliest_cr_line',
    "total_rev_hi_lim",
    "tot_hi_cred_lim",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

loan_feat = [
    "loan_amnt",
    "term",
    "initial_list_status",
    "purpose",
    "application_type",
    "disbursement_method",
]

debt_feat = [
    "int_rate",
    "installment",
    "revol_bal",
    "revol_util",
    "out_prncp",
    "recoveries",
    "dti",
    "dti_joint",
    "tot_coll_amt",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "max_bal_bc",
    "all_util",
    "bc_util",
    "total_bal_ex_mort",
    "revol_bal_joint",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mort_acc",
    "num_rev_tl_bal_gt_0",
    "percent_bc_gt_75",
]

repayment_feat = [
    "num_sats",
    "num_bc_sats",
    "pct_tl_nvr_dlq",
    "bc_open_to_buy",
    #  'last_pymnt_d',
    "last_pymnt_amnt",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "tot_cur_bal",
    "avg_cur_bal",
]

multi_acc_feat = [
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_actv_rev_tl",
    "num_tl_op_past_12m",
    "open_rv_12m",
    "open_rv_24m",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "total_acc",
    "inq_last_6mths",
    "open_acc",
    "inq_fi",
    "inq_last_12m",
    "acc_open_past_24mths",
]

mal_behavior_feat = [
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "pub_rec_bankruptcies",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "mths_since_recent_bc_dlq",
    "chargeoff_within_12_mths",
    # 'collection_recovery_fee',
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "pub_rec",
    "mths_since_last_delinq",
    "delinq_2yrs",
    "delinq_amnt",
    "tax_liens",
]

all_feature_list = (
    qualification_feat
    + loan_feat
    + debt_feat
    + repayment_feat
    + multi_acc_feat
    + mal_behavior_feat
)
