#!/usr/bin/env python3
import io
import logging
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.distributed as dist
from cuml.feature_extraction.text import TfidfVectorizer as CuTfidfVectorizer
from cuml.linear_model import MBSGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import cupy as cp
import cudf
from optuna.integration import lightgbm as lgb_optuna


SEED = 42
EXTRA_SEEDS = [123, 2029]
SEED_LIST = [SEED] + EXTRA_SEEDS

BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
N_SPLITS = 5
NUM_EPOCHS = 3
STACK_BASE_MODELS = ["deberta", "lgb", "sgd"]

DEBERTA_LR = 1.5e-5
DEBERTA_WEIGHT_DECAY = 0.05
DEBERTA_WARMUP_RATIO = 0.1
LABEL_SMOOTHING = 0.05
USE_GRADIENT_CHECKPOINTING = True

DEBERTA_TRAIN_BATCH = 128
DEBERTA_EVAL_BATCH = 64
DEBERTA_GRAD_ACCUM = 1
DEBERTA_NUM_WORKERS = 16

DATA_PATH = None
TEST_PATH_OVERRIDE = None

LOG1P_THRESHOLD = 1000.0
GROUP_COLUMN = None
LGB_TIME_BUDGET = 0
CPU_TRIM_CHARS = 1200
CPU_MAX_FEATURES = 300000
TFIDF_LOGREG = True

TEXT_COL = "text"
LABEL_COL = "label"
ALT_TEXT_COLS = ("text_content",)
META_PREFIXES = ("src_", "lang_", "model_", "ds_")

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "output_training.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tee_logs(log_path: Path = DEFAULT_LOG_PATH):
    class Tee(io.TextIOBase):
        def __init__(self, stream, log_file):
            self.stream = stream
            self.log_file = log_file

        def write(self, data):
            self.stream.write(data)
            self.log_file.write(data)
            self.log_file.flush()
            self.stream.flush()

        def flush(self):
            self.stream.flush()
            self.log_file.flush()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging to %s", log_path)
    return log_file


def seed_everything(seed: int = SEED):
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def discover_paths(data_path=None, test_override=None):
    cwd = Path.cwd()
    data_override = Path(data_path).expanduser() if data_path else None
    candidates = [
        data_override,
        cwd / "merged_ai_human_multisocial_features_cleaned_train.csv",
        cwd / "src/ai_vs_human/merged_ai_human_multisocial_features_cleaned_train.csv",
        cwd / "merged_ai_human_multisocial_features_train.csv",
        cwd / "src/ai_vs_human/merged_ai_human_multisocial_features_train.csv",
        cwd / "merged_ai_human_multisocial_features_cleaned.csv",
        cwd / "src/ai_vs_human/merged_ai_human_multisocial_features_cleaned.csv",
        cwd / "merged_ai_human_multisocial_features.csv",
        cwd / "src/ai_vs_human/merged_ai_human_multisocial_features.csv",
        cwd / "ai_human_content_detection_dataset.csv",
        cwd / "src/ai_vs_human/ai_human_content_detection_dataset.csv",
    ]
    candidates = [p for p in candidates if p is not None]
    train_path = next((p for p in candidates if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError(
            "No training data file found. Set DATA_PATH or place merged_ai_human_multisocial_features_train.csv "
            "(or merged_ai_human_multisocial_features.csv / ai_human_content_detection_dataset.csv) in the repo."
        )

    name = train_path.name
    paired_test = None
    if train_path.suffix:
        if name.endswith("_train" + train_path.suffix):
            paired_test = train_path.with_name(name.replace("_train", "_test"))
        else:
            paired_test = train_path.with_name(
                train_path.stem + "_test" + train_path.suffix
            )

    test_override_path = Path(test_override).expanduser() if test_override else None
    test_candidates = [
        test_override_path,
        paired_test,
        train_path.parent / "merged_ai_human_multisocial_features_test.csv",
        train_path.parent / "ai_human_content_detection_test.csv",
        train_path.parent / "ai_human_content_detection_dataset.csv",
        train_path,
    ]
    test_candidates = [p for p in test_candidates if p is not None]
    test_path = next((p for p in test_candidates if p.exists()), test_candidates[0])
    return train_path, test_path


def prepare_dirs(work_dir: Path):
    dirs = {
        "deberta": work_dir / "models" / "deberta_v3_base",
        "lgb": work_dir / "models" / "lightgbm",
        "lgb_legacy": work_dir / "models" / "lightgbm_numeric",
        "sgd": work_dir / "models" / "tfidf_sgd",
        "stack": work_dir / "models" / "stack_meta",
        "oof": work_dir / "oof",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def rename_text_column(df: pd.DataFrame, text_col=TEXT_COL, alt_cols=ALT_TEXT_COLS):
    if text_col in df.columns:
        return df
    for alt_col in alt_cols:
        if alt_col in df.columns:
            return df.rename(columns={alt_col: text_col})
    raise ValueError(f"Data missing `{text_col}` column.")


def prepare_data(train_path: Path, test_path: Path):
    train_df = rename_text_column(pd.read_csv(train_path))
    test_df = rename_text_column(pd.read_csv(test_path)) if test_path.exists() else None

    initial_rows = len(train_df)
    train_df = train_df.drop_duplicates(subset=[TEXT_COL, LABEL_COL]).reset_index(
        drop=True
    )
    if len(train_df) != initial_rows:
        logging.info(
            "Dropped %d duplicate rows (text+label)", initial_rows - len(train_df)
        )

    drop_numeric = {"grammar_errors", "length"}
    num_cols = [
        c
        for c in train_df.columns
        if c not in [TEXT_COL, LABEL_COL]
        and c not in drop_numeric
        and pd.api.types.is_numeric_dtype(train_df[c])
        and not any(c.startswith(pref) for pref in META_PREFIXES)
    ]

    train_df[num_cols] = train_df[num_cols].apply(pd.to_numeric, errors="coerce")
    num_medians = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(num_medians)
    if test_df is not None:
        missing_num_cols = [c for c in num_cols if c not in test_df.columns]
        for col in missing_num_cols:
            test_df[col] = num_medians[col]
        if missing_num_cols:
            logging.info(
                "Filled missing numeric columns in test set: %d (using train medians)",
                len(missing_num_cols),
            )
        test_df[num_cols] = test_df[num_cols].apply(pd.to_numeric, errors="coerce")
        test_df[num_cols] = test_df[num_cols].fillna(num_medians)

    log1p_cols = [
        c
        for c in num_cols
        if (train_df[c] > 0).all() and train_df[c].max() > LOG1P_THRESHOLD
    ]
    if log1p_cols:
        train_df[log1p_cols] = np.log1p(train_df[log1p_cols])
        if test_df is not None:
            test_df[log1p_cols] = np.log1p(test_df[log1p_cols])
        preview = log1p_cols[:5]
        logging.info(
            "Applied log1p to skewed columns: %s%s",
            preview,
            "..." if len(log1p_cols) > 5 else "",
        )

    logging.info("Train shape: %s", train_df.shape)
    logging.info(
        "Numeric features (%d): %s%s",
        len(num_cols),
        num_cols[:10],
        "..." if len(num_cols) > 10 else "",
    )
    logging.info("\n%s", train_df[[TEXT_COL, LABEL_COL]].head(2))
    logging.info("\n%s", train_df[num_cols].describe().T.head())

    corr = (
        train_df[num_cols + [LABEL_COL]]
        .corr()[LABEL_COL]
        .drop(LABEL_COL)
        .sort_values(key=np.abs, ascending=False)
    )
    logging.info("Top correlated numeric features with label:\n%s", corr.head(10))
    return train_df, test_df, num_cols


def detect_group_column(df: pd.DataFrame):
    if GROUP_COLUMN and GROUP_COLUMN in df.columns:
        return GROUP_COLUMN
    for col in df.columns:
        if any(col.startswith(pref) for pref in META_PREFIXES):
            nunique = df[col].nunique()
            if 1 < nunique < len(df):
                return col
    return None


def build_folds(df: pd.DataFrame, y: np.ndarray):
    group_col = detect_group_column(df)
    if group_col:
        groups = df[group_col]
        n_groups = groups.nunique()
        if n_groups >= N_SPLITS:
            splitter = GroupKFold(n_splits=N_SPLITS)
            folds = list(splitter.split(df[TEXT_COL], y, groups))
            logging.info("Using GroupKFold on `%s` (n_groups=%d)", group_col, n_groups)
        elif n_groups > 1:
            splitter = GroupKFold(n_splits=n_groups)
            folds = list(splitter.split(df[TEXT_COL], y, groups))
            logging.info(
                "Using GroupKFold on `%s` with reduced splits=%d", group_col, n_groups
            )
        else:
            splitter = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED
            )
            folds = list(splitter.split(df[TEXT_COL], y))
            logging.info(
                "Group column `%s` has <=1 group; falling back to StratifiedKFold",
                group_col,
            )
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(df[TEXT_COL], y))
        logging.info("Using StratifiedKFold")
    return folds


class HFTextDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, label_col=None, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        text = str(self.df.iloc[idx][self.text_col])
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False
        )
        if self.label_col is not None:
            enc["labels"] = int(self.df.iloc[idx][self.label_col])
        return enc


def softmax_logits(logits):
    logits = torch.tensor(logits)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs[:, 1]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax_logits(logits)
    return {"roc_auc": roc_auc_score(labels, probs)}


def _train_deberta_fold(job):
    fold = job["fold"]
    train_slice = job["train_df"]
    val_slice = job["val_df"]
    val_idx = job["val_idx"]
    fold_dir = job["fold_dir"]

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(max(local_rank, 0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    collator_local = DataCollatorWithPadding(tokenizer=tokenizer_local, padding=True)

    train_ds = HFTextDataset(
        train_slice, tokenizer_local, TEXT_COL, LABEL_COL, MAX_LENGTH
    )
    val_ds = HFTextDataset(val_slice, tokenizer_local, TEXT_COL, LABEL_COL, MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    if USE_GRADIENT_CHECKPOINTING:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()
        if getattr(model, "config", None) is not None and hasattr(
            model.config, "use_cache"
        ):
            model.config.use_cache = False

    training_kwargs = dict(
        output_dir=str(fold_dir),
        per_device_train_batch_size=DEBERTA_TRAIN_BATCH,
        per_device_eval_batch_size=DEBERTA_EVAL_BATCH,
        gradient_accumulation_steps=DEBERTA_GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=DEBERTA_LR,
        weight_decay=DEBERTA_WEIGHT_DECAY,
        warmup_ratio=DEBERTA_WARMUP_RATIO,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=True,  # Vital: acelera transferencia RAM -> VRAM
        dataloader_num_workers=8,
        group_by_length=True,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        report_to="none",
        lr_scheduler_type="cosine",
        label_smoothing_factor=LABEL_SMOOTHING,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )

    try:
        args = TrainingArguments(
            **training_kwargs, evaluation_strategy="epoch", save_strategy="epoch"
        )
    except TypeError:
        try:
            args = TrainingArguments(
                **training_kwargs,
                evaluate_during_training=True,
                eval_steps=500,
                save_steps=500,
            )
        except TypeError:
            fallback_kwargs = training_kwargs.copy()
            for key in (
                "load_best_model_at_end",
                "metric_for_best_model",
                "greater_is_better",
                "lr_scheduler_type",
                "label_smoothing_factor",
                "gradient_checkpointing",
                "ddp_find_unused_parameters",
            ):
                fallback_kwargs.pop(key, None)
            args = TrainingArguments(**fallback_kwargs)

    args.load_best_model_at_end = True
    if getattr(args, "metric_for_best_model", None) is None:
        args.metric_for_best_model = "roc_auc"
    if getattr(args, "greater_is_better", None) is None:
        args.greater_is_better = True
    if getattr(args, "evaluation_strategy", None) in (None, "no", "none"):
        args.evaluation_strategy = "epoch"
    if getattr(args, "eval_strategy", None) in (None, "no", "none"):
        args.eval_strategy = getattr(args, "evaluation_strategy", "epoch")
    if getattr(args, "save_strategy", None) in (None, "no", "none"):
        args.save_strategy = getattr(args, "evaluation_strategy", "epoch")
    if getattr(args, "load_best_model_at_end", None) is None:
        args.load_best_model_at_end = True

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer_local,
        data_collator=collator_local,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    preds = trainer.predict(val_ds).predictions
    oof_preds = softmax_logits(preds)

    best_dir = trainer.state.best_model_checkpoint or str(fold_dir / "best")
    if trainer.state.best_model_checkpoint is None:
        trainer.save_model(best_dir)

    torch.cuda.empty_cache()
    return fold, val_idx, oof_preds, best_dir


def train_deberta(train_df, folds):
    logging.info("Device: %s", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    oof_deberta = np.zeros(len(train_df))
    model_paths = [None] * len(folds)

    # All ranks must process the full fold list sequentially to keep DDP in sync.
    folds_to_process = list(range(len(folds)))
    logging.info("[Model A] Folds to process (all ranks): %s", folds_to_process)

    for fold in folds_to_process:
        train_idx, val_idx = folds[fold]
        logging.info("[Model A] Preparing fold %d/%d", fold + 1, len(folds))
        train_slice = train_df.iloc[train_idx][[TEXT_COL, LABEL_COL]].reset_index(
            drop=True
        )
        val_slice = train_df.iloc[val_idx][[TEXT_COL, LABEL_COL]].reset_index(drop=True)
        fold_dir = MODEL_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        job = {
            "fold": fold,
            "train_df": train_slice,
            "val_df": val_slice,
            "val_idx": val_idx,
            "fold_dir": fold_dir,
        }

        fold_id, val_idx, preds, best_dir = _train_deberta_fold(job)
        logging.info("[Model A] Fold %d/%d complete", fold_id + 1, len(folds))
        oof_deberta[val_idx] = preds
        model_paths[fold_id] = best_dir

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    model_paths = [p for p in model_paths if p is not None]
    oof_auc = roc_auc_score(train_df[LABEL_COL], oof_deberta)
    logging.info("Model A OOF ROC-AUC: %.5f", oof_auc)
    pd.DataFrame({"oof_deberta": oof_deberta}).to_csv(
        OOF_DIR / "oof_deberta.csv", index=False
    )
    return oof_deberta, model_paths


def tune_lgbm(train_df, num_cols, y, folds):
    time_budget = int(LGB_TIME_BUDGET)
    train_data = lgb.Dataset(train_df[num_cols], label=y)
    tuner_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.015,
        "verbosity": -1,
        "feature_pre_filter": False,
        "scale_pos_weight": (len(y) - y.sum()) / y.sum(),
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "random_state": SEED,
        "bagging_freq": 1,
        "reg_lambda": 1.5,
        "reg_alpha": 0.5,
        "min_split_gain": 0.0,
    }
    tuner_kwargs = dict(
        params=tuner_params,
        train_set=train_data,
        folds=folds,
        num_boost_round=2500,
        callbacks=[lgb.early_stopping(100, verbose=False)],
        return_cvbooster=True,
        study=optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
        ),
    )
    if time_budget > 0:
        tuner_kwargs["time_budget"] = time_budget

    tuner = lgb_optuna.LightGBMTunerCV(**tuner_kwargs)
    tuner.run()

    best_params = tuner.best_params
    best_cvbooster = None
    if hasattr(tuner, "get_best_booster"):
        try:
            best_cvbooster = tuner.get_best_booster()
        except Exception:
            best_cvbooster = None
    best_iter = (
        getattr(best_cvbooster, "best_iteration", None)
        if best_cvbooster is not None
        else None
    )
    if best_iter is None:
        best_iter = (
            best_params.get("num_boost_round")
            or best_params.get("n_estimators")
            or 2500
        )
    best_params.pop("metric", None)
    best_params.pop("feature_pre_filter", None)
    best_params.pop("num_boost_round", None)
    best_params.update({"n_estimators": best_iter, "random_state": SEED, "n_jobs": -1})
    logging.info("[Model B] Mejores parametros: %s", best_params)
    return best_params


def train_lgb(train_df, num_cols, y, folds):
    best_params = tune_lgbm(train_df, num_cols, y, folds)
    oof_lgb = np.zeros(len(train_df))
    lgb_models = []
    lgb_model_paths = []
    lgb_fold_auc = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        logging.info("[Model B] Fold %d/%d", fold + 1, N_SPLITS)
        X_train = train_df.iloc[train_idx][num_cols]
        y_train = y[train_idx]
        X_val = train_df.iloc[val_idx][num_cols]
        y_val = y[val_idx]

        fold_dir = LGB_MODEL_DIR / f"fold_{fold}"
        legacy_fold_dir = LGB_MODEL_DIR_LEGACY / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        legacy_fold_dir.mkdir(parents=True, exist_ok=True)

        model = lgb.LGBMClassifier(**best_params, random_state=SEED + fold)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        val_preds = model.predict_proba(X_val, num_iteration=model.best_iteration_)[
            :, 1
        ]
        fold_auc = roc_auc_score(y_val, val_preds)
        logging.info(
            "[Model B] Fold %d AUC: %.5f (best_iter=%s)",
            fold + 1,
            fold_auc,
            model.best_iteration_,
        )

        oof_lgb[val_idx] = val_preds
        lgb_models.append(model)
        model.booster_.save_model(
            str(fold_dir / "best.txt"), num_iteration=model.best_iteration_
        )
        model.booster_.save_model(
            str(legacy_fold_dir / "best.txt"), num_iteration=model.best_iteration_
        )
        lgb_model_paths.append(fold_dir / "best.txt")
        lgb_fold_auc.append(fold_auc)

    lgb_oof_auc = roc_auc_score(y, oof_lgb)
    logging.info("Model B OOF ROC-AUC: %.5f", lgb_oof_auc)
    pd.DataFrame({"oof_lgb": oof_lgb}).to_csv(OOF_DIR / "oof_lgb.csv", index=False)
    pd.DataFrame({"fold": np.arange(N_SPLITS), "fold_auc": lgb_fold_auc}).to_csv(
        OOF_DIR / "oof_lgb_folds.csv", index=False
    )
    return oof_lgb, lgb_models, lgb_model_paths


def train_tfidf_sgd(train_df, y, folds):
    force_cpu_sgd = True
    use_gpu_sgd = torch.cuda.is_available() and not force_cpu_sgd
    gpu_trim_chars = 400
    gpu_ngram_range = (3, 5)
    gpu_min_df = 2
    gpu_max_features = 80000
    cpu_trim_chars = int(CPU_TRIM_CHARS)
    cpu_ngram_range = (3, 5)
    cpu_min_df = 2
    cpu_max_features = int(CPU_MAX_FEATURES)
    cpu_n_jobs = -1
    use_logreg_tfidf = bool(TFIDF_LOGREG)
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        logging.info("[Model C] GPU not usable (CuPy/CUDA check failed): %s", exc)
        use_gpu_sgd = False

    oof_sgd = np.zeros(len(train_df))
    sgd_models = []
    sgd_model_paths = []
    sgd_fold_auc = []

    logging.info("====================================")
    logging.info("[Model C] Training on GPU with RAPIDS cuML")
    logging.info("====================================")

    if use_gpu_sgd:
        try:
            for fold, (train_idx, val_idx) in enumerate(folds):
                logging.info("[Model C] Fold %d/%d", fold + 1, N_SPLITS)
                train_text = train_df.iloc[train_idx][TEXT_COL].astype(str)
                val_text = train_df.iloc[val_idx][TEXT_COL].astype(str)
                if gpu_trim_chars is not None:
                    train_text = train_text.str.slice(stop=gpu_trim_chars)
                    val_text = val_text.str.slice(stop=gpu_trim_chars)

                X_train_gpu = cudf.Series(train_text.values)
                X_val_gpu = cudf.Series(val_text.values)
                y_train_gpu = cp.array(y[train_idx], dtype=cp.float32)
                y_val = y[val_idx]

                fold_dir = SGD_MODEL_DIR / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                tfidf_gpu = CuTfidfVectorizer(
                    analyzer="char",
                    ngram_range=gpu_ngram_range,
                    min_df=gpu_min_df,
                    max_features=gpu_max_features,
                )
                X_train_tfidf = tfidf_gpu.fit_transform(X_train_gpu)
                X_val_tfidf = tfidf_gpu.transform(X_val_gpu)

                clf = MBSGDClassifier(
                    loss="log",
                    penalty="l2",
                    alpha=1e-4,
                    epochs=2000,
                    tol=1e-3,
                    learning_rate="adaptive",
                )
                clf.fit(X_train_tfidf, y_train_gpu)

                val_probs_gpu = clf.predict_proba(X_val_tfidf)
                val_preds = (
                    val_probs_gpu.values[:, 1].get()
                    if hasattr(val_probs_gpu, "values")
                    else val_probs_gpu[:, 1].get()
                )

                fold_auc = roc_auc_score(y_val, val_preds)
                logging.info("[Model C] Fold %d AUC: %.5f", fold + 1, fold_auc)

                oof_sgd[val_idx] = val_preds
                sgd_models.append((tfidf_gpu, clf))
                joblib.dump((tfidf_gpu, clf), fold_dir / "best.joblib")
                sgd_model_paths.append(fold_dir / "best.joblib")
                sgd_fold_auc.append(fold_auc)

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as exc:
            logging.info(
                "[Model C] GPU training failed, falling back to CPU TF-IDF+SGD. Error: %s",
                exc,
            )
            oof_sgd = np.zeros(len(train_df))
            sgd_models = []
            sgd_model_paths = []
            sgd_fold_auc = []
            use_gpu_sgd = False

    if not use_gpu_sgd:
        logging.info("====================================")
        logging.info(
            "[Model C] Training on CPU with sklearn TF-IDF + SGDClassifier/LogReg"
        )
        logging.info("====================================")
        sgd_param_grid = [
            {"alpha": 1e-4, "eta0": 0.1},
            {"alpha": 3e-4, "eta0": 0.05},
            {"alpha": 1e-5, "eta0": 0.2},
        ]

        def train_cpu_fold(fold, train_idx, val_idx):
            logging.info("[Model C-CPU] Fold %d/%d", fold + 1, N_SPLITS)
            X_train_cpu = train_df.iloc[train_idx][TEXT_COL].astype(str)
            X_val_cpu = train_df.iloc[val_idx][TEXT_COL].astype(str)
            if cpu_trim_chars is not None:
                X_train_cpu = X_train_cpu.str.slice(stop=cpu_trim_chars)
                X_val_cpu = X_val_cpu.str.slice(stop=cpu_trim_chars)
            y_val = y[val_idx]

            tfidf_cpu = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=cpu_ngram_range,
                min_df=cpu_min_df,
                max_features=cpu_max_features,
                sublinear_tf=True,
                strip_accents="unicode",
            )
            best_model = None
            best_auc = -np.inf
            best_preds = None

            if use_logreg_tfidf:
                pipeline = Pipeline(
                    [
                        ("tfidf", tfidf_cpu),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=800,
                                n_jobs=-1,
                                C=1.0,
                                solver="lbfgs",
                                class_weight="balanced",
                            ),
                        ),
                    ]
                )
                pipeline.fit(X_train_cpu, y[train_idx])
                val_preds = pipeline.predict_proba(X_val_cpu)[:, 1]
                best_model, best_preds = pipeline, val_preds
                best_auc = roc_auc_score(y_val, val_preds)
            else:
                for params in sgd_param_grid:
                    pipeline = Pipeline(
                        [
                            ("tfidf", tfidf_cpu),
                            (
                                "clf",
                                SGDClassifier(
                                    loss="log_loss",
                                    penalty="l2",
                                    alpha=params["alpha"],
                                    max_iter=2000,
                                    tol=1e-3,
                                    random_state=SEED + fold,
                                    learning_rate="adaptive",
                                    eta0=params["eta0"],
                                    class_weight="balanced",
                                ),
                            ),
                        ]
                    )
                    pipeline.fit(X_train_cpu, y[train_idx])
                    val_preds = pipeline.predict_proba(X_val_cpu)[:, 1]
                    fold_auc = roc_auc_score(y_val, val_preds)
                    if fold_auc > best_auc:
                        best_auc = fold_auc
                        best_model = pipeline
                        best_preds = val_preds

            fold_auc = roc_auc_score(y_val, best_preds)
            logging.info("[Model C-CPU] Fold %d AUC: %.5f", fold + 1, fold_auc)

            fold_dir = SGD_MODEL_DIR / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, fold_dir / "best.joblib")

            return {
                "fold": fold,
                "val_idx": val_idx,
                "val_preds": best_preds,
                "clf": best_model,
                "fold_auc": fold_auc,
                "model_path": fold_dir / "best.joblib",
            }

        cpu_results = joblib.Parallel(n_jobs=cpu_n_jobs, backend="loky")(
            joblib.delayed(train_cpu_fold)(fold, train_idx, val_idx)
            for fold, (train_idx, val_idx) in enumerate(folds)
        )

        for res in sorted(cpu_results, key=lambda r: r["fold"]):
            oof_sgd[res["val_idx"]] = res["val_preds"]
            sgd_models.append(res["clf"])
            sgd_model_paths.append(res["model_path"])
            sgd_fold_auc.append(res["fold_auc"])

    sgd_oof_auc = roc_auc_score(y, oof_sgd)
    logging.info("Model C OOF ROC-AUC: %.5f", sgd_oof_auc)
    pd.DataFrame({"oof_sgd": oof_sgd}).to_csv(OOF_DIR / "oof_sgd.csv", index=False)
    fold_ids = list(range(len(sgd_fold_auc)))
    fold_auc_values = list(sgd_fold_auc)
    if len(fold_auc_values) != N_SPLITS:
        logging.info(
            "[Model C] Warning: expected %d fold AUCs, got %d; padding with NaN",
            N_SPLITS,
            len(fold_auc_values),
        )
        for missing_fold in range(len(fold_auc_values), N_SPLITS):
            fold_ids.append(missing_fold)
            fold_auc_values.append(np.nan)

    pd.DataFrame({"fold": fold_ids, "fold_auc": fold_auc_values}).to_csv(
        OOF_DIR / "oof_sgd_folds.csv", index=False
    )
    return oof_sgd, sgd_models, sgd_model_paths


def load_oof_preds(name, y, cache=None):
    cache = cache or {}
    existing = cache.get(name)
    if existing is not None and len(existing):
        return np.asarray(existing)
    path = OOF_DIR / f"oof_{name}.csv"
    if path.exists():
        col = f"oof_{name}"
        df = pd.read_csv(path)
        if col in df:
            logging.info("[Meta] Loaded %s from %s", col, path)
            return df[col].values
    return None


def train_meta_learner(y, stack_base_models, cache=None):
    oof_sources = {}
    missing_oof = []
    length_mismatch = []
    for key in stack_base_models:
        preds = load_oof_preds(key, y, cache)
        if preds is None:
            missing_oof.append(key)
            continue
        if len(preds) != len(y):
            length_mismatch.append((key, len(preds)))
            continue
        oof_sources[key] = preds

    if missing_oof:
        raise RuntimeError(
            f"Missing OOF predictions for: {', '.join(missing_oof)}. "
            "Run the corresponding training steps once to cache them to disk."
        )
    if length_mismatch:
        mismatch_msg = "; ".join(f"{k} has {n} rows" for k, n in length_mismatch)
        raise RuntimeError(
            f"OOF size mismatch ({len(y)} rows expected): {mismatch_msg}"
        )

    stack_train = np.column_stack([oof_sources[k] for k in stack_base_models])
    meta_candidates = [0.01, 0.1, 1.0, 10]
    meta_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    meta_learner = LogisticRegressionCV(
        Cs=meta_candidates,
        cv=meta_skf,
        max_iter=2000,
        n_jobs=-1,
        solver="lbfgs",
        scoring="roc_auc",
        class_weight="balanced",
    )
    meta_learner.fit(stack_train, y)
    meta_learner.base_model_order = stack_base_models
    best_c = float(np.ravel(meta_learner.C_)[0])
    stack_auc = roc_auc_score(y, meta_learner.predict_proba(stack_train)[:, 1])
    logging.info("Meta-learner OOF ROC-AUC: %.5f (best C=%.4f)", stack_auc, best_c)

    stack_model_path = STACK_MODEL_DIR / "meta_learner.joblib"
    joblib.dump(meta_learner, stack_model_path)
    stack_oof = {f"oof_{k}": oof_sources[k] for k in stack_base_models}
    stack_oof["oof_stack"] = meta_learner.predict_proba(stack_train)[:, 1]
    stack_oof[LABEL_COL] = y
    pd.DataFrame(stack_oof).to_csv(OOF_DIR / "oof_stack.csv", index=False)
    return meta_learner


def _dedupe_paths(paths):
    seen = set()
    unique = []
    for p in paths:
        p = Path(p)
        if p.exists() and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def predict_deberta(df):
    tok = globals().get("tokenizer") or AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=False
    )
    coll = globals().get("collator") or DataCollatorWithPadding(
        tokenizer=tok, padding=True
    )
    globals()["tokenizer"] = tok
    globals()["collator"] = coll
    test_ds = HFTextDataset(df, tok, TEXT_COL, None, MAX_LENGTH)
    fold_paths = _dedupe_paths(
        [MODEL_DIR / f"fold_{f}" / "best" for f in range(N_SPLITS)]
        + list(globals().get("deberta_model_paths", []))
    )
    if not fold_paths:
        raise RuntimeError("No DeBERTa checkpoints found; run Model A training first.")
    fold_preds = []
    for path in fold_paths:
        model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        infer_trainer = Trainer(model=model, tokenizer=tok, data_collator=coll)
        preds = infer_trainer.predict(test_ds).predictions
        fold_preds.append(softmax_logits(preds))
        torch.cuda.empty_cache()
    return np.mean(fold_preds, axis=0)


def predict_lgb(df, num_cols):
    feats = df[num_cols]
    lgb_dirs = [LGB_MODEL_DIR]
    if "LGB_MODEL_DIR_LEGACY" in globals():
        lgb_dirs.append(LGB_MODEL_DIR_LEGACY)
    candidate_paths = _dedupe_paths(
        [d / f"fold_{f}" / "best.txt" for d in lgb_dirs for f in range(N_SPLITS)]
        + list(globals().get("lgb_model_paths", []))
    )
    if candidate_paths:
        models = [lgb.Booster(model_file=str(p)) for p in candidate_paths]
    else:
        models = globals().get("lgb_models")
    if not models:
        raise RuntimeError("No LightGBM models found; run Model B training first.")
    fold_preds = []
    for m in models:
        if isinstance(m, lgb.Booster):
            fold_preds.append(m.predict(feats))
        else:
            fold_preds.append(
                m.predict_proba(
                    feats, num_iteration=getattr(m, "best_iteration_", None)
                )[:, 1]
            )
    return np.mean(fold_preds, axis=0)


def predict_sgd(df):
    texts = df[TEXT_COL].astype(str)
    candidate_paths = _dedupe_paths(
        [SGD_MODEL_DIR / f"fold_{f}" / "best.joblib" for f in range(N_SPLITS)]
        + list(globals().get("sgd_model_paths", []))
    )
    if candidate_paths:
        models = [joblib.load(p) for p in candidate_paths]
    else:
        models = globals().get("sgd_models")
    if not models:
        raise RuntimeError("No TF-IDF+SGD models found; run Model C training first.")
    fold_preds = [
        (
            m.predict_proba(texts)[:, 1]
            if hasattr(m, "predict_proba")
            else m[1].predict_proba(m[0].transform(texts))[:, 1]
        )
        for m in models
    ]
    return np.mean(fold_preds, axis=0)


def load_meta_model():
    saved_path = STACK_MODEL_DIR / "meta_learner.joblib"
    if saved_path.exists():
        return joblib.load(saved_path)
    fallback = globals().get("meta_learner")
    if fallback is None:
        raise RuntimeError("Meta-learner not trained yet; run the stacking step.")
    return fallback


def run_inference(test_df, num_cols):
    if test_df is None:
        logging.info("No test file found; set TEST_PATH to run inference.")
        return
    logging.info("Running inference on test set...")
    base_preds = {
        "deberta": predict_deberta(test_df),
        "lgb": predict_lgb(test_df, num_cols),
        "sgd": predict_sgd(test_df),
    }
    meta_for_inference = load_meta_model()
    base_order = getattr(meta_for_inference, "base_model_order", STACK_BASE_MODELS)
    stack_test = np.column_stack([base_preds[name] for name in base_order])
    test_pred = meta_for_inference.predict_proba(stack_test)[:, 1]
    submission = pd.DataFrame({"id": test_df.index, "prediction": test_pred})
    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    logging.info("Saved submission.csv")


def diagnostics(y, folds):
    def load_oof(name):
        in_memory = globals().get(f"oof_{name}")
        if in_memory is not None:
            return np.asarray(in_memory)
        path = OOF_DIR / f"oof_{name}.csv"
        if path.exists():
            col = f"oof_{name}"
            df = pd.read_csv(path)
            if col in df:
                return df[col].values
        return None

    def fold_scores(preds):
        scores = []
        for fold, (_, val_idx) in enumerate(folds):
            scores.append(roc_auc_score(y[val_idx], preds[val_idx]))
        return scores

    summary_rows = []
    for label, key in [
        ("Model A: DeBERTa-v3", "deberta"),
        ("Model B: LightGBM numeric", "lgb"),
        ("Model C: TF-IDF + SGD", "sgd"),
        ("Meta-learner (stack)", "stack"),
    ]:
        preds = load_oof(key)
        if preds is None:
            logging.info("Skipping %s: no OOF predictions found", label)
            continue
        overall = roc_auc_score(y, preds)
        folds_auc = fold_scores(preds) if len(preds) == len(y) else None
        summary_rows.append(
            OrderedDict(
                model=label,
                overall_auc=overall,
                fold_mean=np.mean(folds_auc) if folds_auc else None,
                fold_std=np.std(folds_auc) if folds_auc else None,
                min_fold=np.min(folds_auc) if folds_auc else None,
                max_fold=np.max(folds_auc) if folds_auc else None,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        logging.info("\n%s", summary_df.sort_values("overall_auc", ascending=False))
    else:
        logging.info("No OOF data available to summarize.")

    meta_model = None
    try:
        meta_model = load_meta_model()
    except Exception:
        meta_model = globals().get("meta_learner")

    if meta_model is not None and hasattr(meta_model, "coef_"):
        coef = meta_model.coef_.ravel()
        bases = ["deberta", "lgb", "sgd"][: len(coef)]
        logging.info(
            "Meta-learner coefficients (positive -> higher AI probability):\n%s",
            pd.Series(coef, index=bases),
        )
        if hasattr(meta_model, "C"):
            logging.info("Meta-learner C: %s", getattr(meta_model, "C", None))

    base_preds = {name: load_oof(name) for name in ["deberta", "lgb", "sgd"]}
    if all(v is not None for v in base_preds.values()):
        corr_df = pd.DataFrame(base_preds)
        logging.info("Correlation of base model OOF predictions:\n%s", corr_df.corr())

    pos_rate = y.mean()
    logging.info("Positive rate in training: %.4f (n=%d)", pos_rate, len(y))


def main():
    tee_logs()
    seed_everything(SEED)
    logging.info("Device: %s", DEVICE)

    global DATA_PATH, TEST_PATH, WORK_DIR, MODEL_DIR, LGB_MODEL_DIR, LGB_MODEL_DIR_LEGACY, SGD_MODEL_DIR, STACK_MODEL_DIR, OOF_DIR

    DATA_PATH, TEST_PATH = discover_paths(DATA_PATH, TEST_PATH_OVERRIDE)
    logging.info("Using training file: %s", DATA_PATH)
    logging.info("Using test file: %s", TEST_PATH)
    if TEST_PATH == DATA_PATH:
        logging.info(
            "TEST_PATH not provided; using training data as a smoke-test for inference."
        )

    WORK_DIR = DATA_PATH.parent
    dirs = prepare_dirs(WORK_DIR)
    MODEL_DIR = dirs["deberta"]
    LGB_MODEL_DIR = dirs["lgb"]
    LGB_MODEL_DIR_LEGACY = dirs["lgb_legacy"]
    SGD_MODEL_DIR = dirs["sgd"]
    STACK_MODEL_DIR = dirs["stack"]
    OOF_DIR = dirs["oof"]

    train_df, test_df, num_cols = prepare_data(DATA_PATH, TEST_PATH)
    y = train_df[LABEL_COL].astype(int).values
    folds = build_folds(train_df, y)

    global oof_deberta, deberta_model_paths, oof_lgb, lgb_models, lgb_model_paths, oof_sgd, sgd_models, sgd_model_paths, meta_learner

    oof_deberta, deberta_model_paths = train_deberta(train_df, folds)
    oof_lgb, lgb_models, lgb_model_paths = train_lgb(train_df, num_cols, y, folds)
    oof_sgd, sgd_models, sgd_model_paths = train_tfidf_sgd(train_df, y, folds)
    meta_learner = train_meta_learner(
        y, STACK_BASE_MODELS, {"deberta": oof_deberta, "lgb": oof_lgb, "sgd": oof_sgd}
    )

    run_inference(test_df, num_cols)
    diagnostics(y, folds)


if __name__ == "__main__":
    main()
