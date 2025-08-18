import os
import torch
from shutil import rmtree
import gc
import json

import hydra
from pathlib import Path
from typing import Union
import logging
from atgen.utils.main_decorator import main_decorator
from atgen.utils.constants import (
    DEFAULT_CONFIG_NAME,
    UNLABELED_DATA_SPLIT_DEFAULT_NAME,
    TEST_DATA_SPLIT_DEFAULT_NAME,
    NUM_PROCS_FOR_DATASETS,
)

log = logging.getLogger()


@main_decorator
def run_subset_selection(config, workdir: Union[str, Path]):
    from transformers import set_seed
    from datasets import concatenate_datasets, Dataset

    from atgen.metrics.compute_metrics import compute_metrics
    from atgen.utils.data import (
        load_data,
        prepare_conversational_data,
        maybe_get_few_shot_examples,
    )
    from atgen.utils.load_model_tokenizer import load_model_tokenizer
    from atgen.utils.prepare_model_for_training import prepare_model_for_training
    from atgen.utils.training_utils import get_trainer
    from atgen.strategies.get_strategy import get_strategy
    from atgen.labellers import get_labeller
    from atgen.utils.generate import generate
    from atgen.utils.check_required_performance import check_required_performance
    from atgen.utils.save_labeled_data import save_labeled_data
    from atgen.utils.save_log_iter_results import save_log_iter_results
    from atgen.strategies.base_strategy import BaseStrategy
    from atgen.labellers.base_labeller import BaseLabeler
    from atgen.utils.check_performance_metrics import (
        check_performance_against_requirements,
    )
    from atgen.utils.evaluate_bfcl import evaluate_bfcl

    seed = config.seed
    cache_dir = config.cache_dir
    dev_split_size = config.training.dev_split_size
    output_column_name_train = config.data.train_output_column_name
    output_column_name_test = config.data.test_output_column_name

    model_name = config.model.checkpoint

    num_al_iterations = config.al.num_iterations
    budget = config.al.budget
    if budget is None:
        budget = 1e10

    has_test = (
        config.data.test_split_name is not None and config.data.test_split_name != ""
    )

    print(
        f"""Running Active Learning...
AL Strategy: {config.al.strategy}
Num Iterations: {num_al_iterations}
Query Size: {config.al.query_size}
Dataset: {config.data.dataset if isinstance(config.data.dataset, str) else 'custom'}
Seed: {seed}
Model: {model_name}
Config: {config.name}
Prompt:\n{config.data.system_prompt}
"""
    )

    if isinstance(workdir, str):
        workdir = Path(workdir)
    train_output_dir = workdir / "tmp"
    save_dir = workdir / "tmp_best"

    print("Loading data.")
    unlabeled_data = load_data(
        data_config=config.data,
        split=UNLABELED_DATA_SPLIT_DEFAULT_NAME,
        cache_dir=config.cache_dir,
        seed=seed,
    )
    if has_test and not config.data.use_test_benchmark:
        test_data = load_data(
            data_config=config.data,
            split=TEST_DATA_SPLIT_DEFAULT_NAME,
            cache_dir=config.cache_dir,
            seed=seed,
        )
    # TODO: make better. Current workaround for multi-choice QA.
    if config.data.get("processed_input_column_name", None) is not None:
        config.data.input_column_name = config.data.processed_input_column_name
    input_column_name = config.data.input_column_name
    # After loading data, need to calculate the query size if it is proportional to the dataset size
    if config.al.init_query_size is None:
        config.al.init_query_size = int(len(unlabeled_data) * config.al.init_query_size)
        log.info(f"Setting init query size to {config.al.init_query_size}")
    if isinstance(config.al.query_size, float):
        config.al.query_size = int(len(unlabeled_data) * config.al.query_size)
        log.info(f"Setting query size to {config.al.query_size}")
    al_query_size = config.al.query_size

    print("Initial iteration: loading model & tokenizer...")
    model, tokenizer = load_model_tokenizer(
        checkpoint=model_name, model_config=config.model, cache_dir=cache_dir
    )

    if config.al.query_ids_path:
        with open(config.al.query_ids_path, "r") as f:
            labeled_ids = json.load(f)[:al_query_size]
        log.info(f"Loaded {len(labeled_ids)} labeled ids from {config.al.query_ids_path}")
    else:
        print("Loading subset selection strategy...")
        ss_strategy: BaseStrategy = get_strategy(
            strategy_name=config.al.strategy,
            subsample_size=config.al.subsample_size,
            unlabeled_pool=unlabeled_data[input_column_name],
            model=model,
            tokenizer=tokenizer,
            inference_config=config.inference,  # for hadas
            model_config=config.model,  # for hadas
            data_config=config.data,  # for hadas
            cache_dir=cache_dir,  # for hadas, huds, graph_cut
            seed=seed,
            **config.al.strategy_kwargs,
        )

        print("Loading labeller...")
        # TODO: unsure whether need to log here since may be confusing for a human labeller
        labeller: BaseLabeler = get_labeller(
            config.labeller,
            output_column_name=output_column_name_train,
            cache_dir=cache_dir,
            budget=budget,
            workdir=workdir,  # if labeller is a human
            data_config=config.data,  # if labeller is a custom LLM on transformers
            model_config=config.model,  # if labeller is a custom LLM on transformers
        )
        print("Calculating query_ids")
        labeled_ids: list[int] = ss_strategy(
            model=model,
            tokenizer=tokenizer,
            unlabeled_pool=unlabeled_data.remove_columns(output_column_name_train),
            labeled_pool=None,
            num_to_label=al_query_size,
            batch_size=config.inference.batch_size,
            max_new_tokens=config.inference.max_new_tokens,
        )

    query: Dataset = unlabeled_data.filter(lambda x: x["id"] in labeled_ids)
    labeled_data: Dataset = labeller(query)
    if labeller.is_out_of_budget:
        labeled_data = labeled_data.filter(
            lambda x: x[labeller.output_column_name] != "",
            batched=False,
            num_proc=NUM_PROCS_FOR_DATASETS,
        )
        print(f"Labeler ran out of budget at iteration 0.")

    # Get the few-shot examples
    few_shot_examples, labeled_data = maybe_get_few_shot_examples(
        config=config, labeled_data=labeled_data, workdir=workdir
    )

    iter_dir = workdir / "iter_0"
    iter_dir.mkdir(exist_ok=True)

    print(f"Saving labeled data...")
    save_labeled_data(
        labeled_data=labeled_data,
        labeled_query=labeled_data,
        workdir=workdir,
        iter_dir=iter_dir,
        labeled_ids=labeled_ids,
        query_ids=labeled_ids,
    )

    if has_test and not config.data.use_test_benchmark:
        print("Preparing test data")
        test_data: Dataset = prepare_conversational_data(
            dataset=test_data,
            data_config=config.data,
            split="test",
            few_shot_examples=few_shot_examples,
            model_name=model_name,
        )
    # Evaluate the initial model before any training
    if config.al.eval_zero_iteration:
        if not config.data.use_test_benchmark:
            generations: list[str] = generate(
                config.inference,
                data=test_data,
                model=model,
                tokenizer=tokenizer,
                save_dir=save_dir,
                data_config=config.data,
                model_config=config.model,
            )
            if os.path.exists(save_dir):
                rmtree(save_dir)

            metrics: dict[str, float] = compute_metrics(
                generated_texts=generations,
                reference_texts=test_data[output_column_name_test],
                original_texts=test_data[input_column_name],
                task=config.data.task,
                config=config.evaluation,
                cache_dir=cache_dir,
            )
        else:
            if "bfcl" in config.data.test_split_name:
                test_split_name = config.data.test_split_name.split("bfcl_")[1]
            else:
                raise NotImplementedError(
                    f"Test split name {config.data.test_split_name} is not supported"
                )
            generations, metrics = evaluate_bfcl(
                model_name=model_name,
                bfcl_results_dir=iter_dir,
                test_category=test_split_name,
                num_threads=config.inference.num_threads_for_bfcl,
            )
        save_log_iter_results(
            config=config,
            workdir=workdir,
            iter_dir=iter_dir,
            metrics=metrics,
            generations=generations,
            al_iter=0,
            train_result={},
            model=None,
            tokenizer=None,
        )

    # Start AL cycle. Use `num_al_iterations + 2` because we do not label data
    # but want to train the model on the last iteration.

    al_iter = 1 if config.al.eval_zero_iteration else 0
    iter_dir = workdir / ("iter_" + str(al_iter))
    iter_dir.mkdir(exist_ok=True)

    if not config.data.is_in_conversational_format:
        train_eval_data = prepare_conversational_data(
            dataset=labeled_data,
            data_config=config.data,
            split="train",
            few_shot_examples=few_shot_examples,
            model_name=model_name,
        )
    else:
        train_eval_data = labeled_data

    if dev_split_size > 0 and len(train_eval_data) > 1:
        train_eval_data = train_eval_data.train_test_split(
            test_size=dev_split_size, shuffle=True, seed=seed
        )
        train_data = train_eval_data["train"]
        eval_data = train_eval_data["test"]
    else:
        train_data = train_eval_data
        eval_data = None

    model = prepare_model_for_training(model, config.model.peft)

    # Set seed for reproducibility
    set_seed(seed)
    trainer = get_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=train_output_dir,
        seed=seed,
    )

    # Launch training
    if len(train_data) > 0:
        train_result = trainer.train()
        print(f"Training completed with {len(train_data)} examples")
    else:
        log.warning(
            "No labeled training data available. Skipping training for this iteration."
        )
        train_result = {"training_loss": 0.0, "skipped": True}

    model = model.cpu()
    if config.model.save_in_fp_32:
        model = model.to(torch.float32)
    model = model.eval().merge_and_unload()
    # Free up memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    rmtree(train_output_dir)

    if not has_test:
        if dev_split_size > 0:
            test_data = eval_data
    else:
        if not config.data.use_test_benchmark:
            generations: list[str] = generate(
                config.inference,
                data=test_data,
                model=model,
                tokenizer=tokenizer,
                save_dir=save_dir,
                data_config=config.data,
                model_config=config.model,
            )
            if os.path.exists(save_dir):
                rmtree(save_dir)

            metrics: dict[str, float] = compute_metrics(
                generated_texts=generations,
                reference_texts=test_data[output_column_name_test],
                original_texts=test_data[input_column_name],
                task=config.data.task,
                config=config.evaluation,
                cache_dir=cache_dir,
            )
        else:
            if "bfcl" in config.data.test_split_name:
                test_split_name = config.data.test_split_name.split("bfcl_")[1]
            else:
                raise NotImplementedError(
                    f"Test split name {config.data.test_split_name} is not supported"
                )
            generations, metrics = evaluate_bfcl(
                model_name=model_name,
                bfcl_results_dir=iter_dir,
                model=model,
                tokenizer=tokenizer,
                test_category=test_split_name,
                num_threads=config.inference.num_threads_for_bfcl,
            )
        save_log_iter_results(
            config=config,
            workdir=workdir,
            iter_dir=iter_dir,
            metrics=metrics,
            generations=generations,
            al_iter=al_iter,
            train_result=train_result,
            model=model,
            tokenizer=tokenizer,
        )

    print("Subset selection is done.")


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", os.getcwd() + "/configs/"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME", DEFAULT_CONFIG_NAME),
    version_base="1.1",
)
def main(config):
    if getattr(config, "debug", True):
        try:
            run_subset_selection(config)
        except Exception as e:
            print(e)
            import pdb
            import sys

            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
    else:
        run_subset_selection(config)


if __name__ == "__main__":
    main()
