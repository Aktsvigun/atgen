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
    OUTPUT_FIELD_PURPOSE_TRAIN,
    OUTPUT_FIELD_PURPOSE_TEST,
)

log = logging.getLogger()


@main_decorator
def run_active_learning(config, workdir: Union[str, Path]):
    from transformers import set_seed
    from datasets import concatenate_datasets, Dataset

    from atgen.metrics.compute_metrics import compute_metrics
    from atgen.utils.data import (
        load_data,
        prepare_conversational_data,
        maybe_get_few_shot_examples,
        get_output_column_name_for_phase,
    )
    from atgen.utils.installers import install_spacy, install_nltk
    from atgen.utils.load_model_tokenizer import load_model_tokenizer
    from atgen.utils.prepare_model_for_training import prepare_model_for_training
    from atgen.utils.training_utils import get_trainer
    from atgen.strategies.get_strategy import get_strategy
    from atgen.labellers import get_labeller
    from atgen.utils.generate import generate
    from atgen.utils.check_required_performance import check_required_performance
    from atgen.utils.save_labeled_data import save_labeled_data
    from atgen.utils.save_log_iter_results import save_log_iter_results
    from atgen.utils.get_initial_labeled_data import (
        get_initial_labeled_data_with_few_shot,
    )
    from atgen.strategies.base_strategy import BaseStrategy
    from atgen.labellers.base_labeller import BaseLabeler
    from atgen.utils.check_performance_metrics import (
        check_performance_against_requirements,
    )

    # TODO Figure out how to stop downloading it every time
    install_spacy()
    install_nltk()

    seed = config.seed
    cache_dir = config.cache_dir
    input_column_name = config.data.input_column_name
    dev_split_size = config.training.dev_split_size
    output_column_name_train = config.data.train_output_column_name
    output_column_name_test = config.data.test_output_column_name

    model_name = config.model.checkpoint

    num_al_iterations = config.al.num_iterations
    al_query_size = config.al.query_size
    required_performance_dict = check_required_performance(
        config.al.required_performance
    )
    budget = config.al.budget
    if budget is None:
        budget = 1e10

    # Stopping criteria due to reaching required performance
    is_performance_reached = False

    # Initialize variables for tracking available metrics
    available_metrics = {}
    is_metrics_availability_checked = False

    has_test = (
        config.data.test_split_name is not None and config.data.test_split_name != ""
    )

    log.info(
        f"""Running Active Learning...
AL Strategy: {config.al.strategy}
Num Iterations: {num_al_iterations}
Query Size: {al_query_size}
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

    log.info("Loading data.")
    unlabeled_data = load_data(
        data_config=config.data,
        split=UNLABELED_DATA_SPLIT_DEFAULT_NAME,
        cache_dir=config.cache_dir,
        seed=seed,
    )
    if has_test:
        test_data = load_data(
            data_config=config.data,
            split=TEST_DATA_SPLIT_DEFAULT_NAME,
            cache_dir=config.cache_dir,
            seed=seed,
        )

    log.info("Initial iteration: loading model & tokenizer.")
    model, tokenizer = load_model_tokenizer(
        checkpoint=model_name, model_config=config.model, cache_dir=cache_dir
    )

    log.info("Loading AL strategy.")
    al_strategy: BaseStrategy = get_strategy(
        config.al.strategy,
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

    init_query_size = config.al.init_query_size + config.data.few_shot.count
    init_query_size_is_positive = init_query_size > 0
    if init_query_size_is_positive:
        iter_dir = workdir / "iter_0"
        iter_dir.mkdir(exist_ok=True)

        if al_strategy.random_init:
            labeled_data, labeled_ids = get_initial_labeled_data_with_few_shot(
                config=config,
                init_query_size=init_query_size,
                unlabeled_data=unlabeled_data,
                labeller=labeller,
                model_name=model_name,
            )
            unlabeled_data = unlabeled_data.filter(
                lambda x: x["id"] not in set(labeled_ids)
            )
        else:
            query_ids: list[str] = al_strategy(
                model=model,
                tokenizer=tokenizer,
                unlabeled_pool=unlabeled_data.remove_columns(output_column_name_train),
                labeled_pool=None,
                num_to_label=al_query_size,
                batch_size=config.inference.batch_size,
                max_new_tokens=config.inference.max_new_tokens,
            )

            query = unlabeled_data.filter(lambda x: x["id"] in query_ids)
            unlabeled_data = unlabeled_data.filter(lambda x: x["id"] not in query_ids)
            labeled_data = labeller(query)
            if labeller.is_out_of_budget:
                log.info(f"Labeler ran out of budget at iteration 0.")
            labeled_ids = query_ids

        # Get the few-shot examples
        few_shot_examples, labeled_data = maybe_get_few_shot_examples(
            config=config, labeled_data=labeled_data, workdir=workdir
        )

        log.info(f"Saving labeled data at iteration 0.")
        save_labeled_data(
            labeled_data=labeled_data,
            labeled_query=labeled_data,
            workdir=workdir,
            iter_dir=iter_dir,
            labeled_ids=labeled_ids,
            query_ids=labeled_ids,
        )
    else:
        labeled_data = unlabeled_data.select(range(0, 0))
        labeled_ids = []

    unlabeled_data: Dataset = prepare_conversational_data(
        dataset=unlabeled_data,
        data_config=config.data,
        split="test",
        few_shot_examples=few_shot_examples,
        model_name=model_name,
    )

    if has_test:
        if not config.data.use_test_benchmark:
            test_data: Dataset = prepare_conversational_data(
                dataset=test_data,
                data_config=config.data,
                split="test",
                few_shot_examples=few_shot_examples,
                model_name=model_name,
            )
        # Evaluate the initial model before any training
        if init_query_size_is_positive and config.al.evaluate_zero_iteration:
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
                config=config.evaluation,
                cache_dir=cache_dir,
            )

            # Check required performance metrics
            is_performance_reached, is_metrics_availability_checked, available_metrics = (
                check_performance_against_requirements(
                    metrics=metrics,
                    required_performance_dict=required_performance_dict,
                    is_metrics_availability_checked=is_metrics_availability_checked,
                    available_metrics=available_metrics,
                )
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
            )

    # Start AL cycle. Use `num_al_iterations + 2` because we do not label data
    # but want to train the model on the last iteration.
    start_iter = 1 if init_query_size_is_positive else 0
    for al_iter in range(start_iter, num_al_iterations + 1 + start_iter):
        log.info(f"Starting AL iteration #{al_iter}.")

        iter_dir = workdir / ("iter_" + str(al_iter))
        iter_dir.mkdir(exist_ok=True)

        log.info(f"Iteration {al_iter}: model loading started...")
        if al_iter != 1:
            model, tokenizer = load_model_tokenizer(
                checkpoint=model_name, model_config=config.model, cache_dir=cache_dir
            )
        log.info(f"Iteration {al_iter}: model loading done.")

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
        trainer: SFTTrainer = get_trainer(
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
            log.info(f"Training completed with {len(train_data)} examples")
        else:
            log.warning(
                "No labeled training data available. Skipping training for this iteration."
            )
            train_result = {"training_loss": 0.0, "skipped": True}
        del trainer
        rmtree(train_output_dir)

        if config.model.save_in_fp_32:
            model = model.to(torch.float32)
        model = model.eval().merge_and_unload()

        if not has_test:
            if dev_split_size > 0:
                test_data = eval_data
        else:
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
                config=config.evaluation,
                cache_dir=cache_dir,
            )

            # Check required performance metrics
            is_performance_reached, is_metrics_availability_checked, available_metrics = (
                check_performance_against_requirements(
                    metrics=metrics,
                    required_performance_dict=required_performance_dict,
                    is_metrics_availability_checked=is_metrics_availability_checked,
                    available_metrics=available_metrics,
                )
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
        )

        # Make AL query for the next round if we have not run out of iterations
        if al_iter != num_al_iterations + 1:
            log.info(f"Making AL query at iteration {al_iter}.")
            query_ids: list[str] = al_strategy(
                model=model,
                tokenizer=tokenizer,
                unlabeled_pool=unlabeled_data.remove_columns(output_column_name_train),
                labeled_pool=labeled_data,
                num_to_label=al_query_size,
                batch_size=config.inference.batch_size,
                max_new_tokens=config.inference.max_new_tokens,
            )

            query: Dataset = unlabeled_data.filter(lambda x: x["id"] in query_ids)
            unlabeled_data: Dataset = unlabeled_data.filter(lambda x: x["id"] not in query_ids)
            labeled_query: Dataset = labeller(query)
            if labeller.is_out_of_budget:
                log.info(f"Labeler ran out of budget at iteration {al_iter}.")
            labeled_data: Dataset = concatenate_datasets([labeled_data, labeled_query])
            labeled_ids += query_ids

            log.info(f"Saving labeled data at iteration #{al_iter}.")
            save_labeled_data(
                labeled_data=labeled_data,
                labeled_query=labeled_query,
                workdir=workdir,
                iter_dir=iter_dir,
                labeled_ids=labeled_ids,
                query_ids=query_ids,
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if labeller.is_out_of_budget:
            log.info("Labeler ran out of budget. Finishing active learning.")
            return
        if is_performance_reached:
            log.info("Stopping AL since the required performance is reached.")
            return

    log.info("Active learning is done.")


@hydra.main(
    config_path=os.environ.get("HYDRA_CONFIG_PATH", os.getcwd() + "/configs/"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME", DEFAULT_CONFIG_NAME),
    version_base="1.1",
)
def main(config):
    if getattr(config, "debug", True):
        try:
            run_active_learning(config)
        except Exception as e:
            print(e)
            import pdb
            import sys

            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
    else:
        run_active_learning(config)


if __name__ == "__main__":
    main()
