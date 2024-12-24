import argparse
import os

from rich.progress import Progress
from slickconf import instantiate, load_config, summarize
from slickconf.loader import apply_overrides
import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader

from halite.data.dataset import MapDataset
from halite.data.preprocess import Collator
from halite.logging import get_logger
from halite.parallel import ParallelDims
from halite.transformers.infer import InferenceEngine, ModelConfig
from halite.projects.eval.config import EvalTaskConfig, Model, Task, EvalTask
from halite.utils import get_torch_dtype


def run_task(engine: InferenceEngine, task: Task, eval: EvalTask, logger):
    dset = instantiate(task.dataset)
    eval_fn = instantiate(task.evaluate_fn)

    preprocess_ops = []
    if task.preprocess is not None:
        for op in task.preprocess:
            preprocess_ops.append(instantiate(op))
    dset = MapDataset(
        [instantiate(task.dataset)], [task.name], operations=preprocess_ops
    )
    collator = Collator(())
    loader = DataLoader(
        dset, batch_size=eval.batch_size, collate_fn=collator, num_workers=4
    )

    fewshot_sampler = None
    if task.fewshot is not None:
        fewshot_sampler = instantiate(task.fewshot.sampler)(
            instantiate(task.fewshot.samples)()
        )

    prefix_fn = None
    if task.prefix is not None:
        prefix_fn = instantiate(task.prefix)

    max_context_len = engine.model_config.context_len
    tokenizer = engine.tokenizer

    metric_accum = None

    with Progress() as progress:
        task_progress = progress.add_task(task.name, total=len(loader))

        for batch in loader:
            requests = []
            for prompt in batch.prompt:
                prefix = ""
                if prefix_fn is not None:
                    fewshot = None
                    if fewshot_sampler is not None:
                        fewshot = fewshot_sampler()

                    prefix = prefix_fn(fewshot=fewshot)

                prompt = prefix + prompt

                prefix_len = len(tokenizer.encode(prompt))
                sampling_params = task.sampling_params.build(
                    prefix_len, max_context_len
                )

                requests.append((prompt, sampling_params))

            result = engine.infer_batch(requests)

            for batch_id, (prompt, response) in enumerate(result):
                response = tokenizer.decode(response)

                record = batch.slice(batch_id)
                metric = eval_fn(record, [response])

                if metric["exact_match"] != 0:
                    logger.info(record.prompt)
                    logger.info(record.solution)
                    logger.info(tokenizer.decode(prompt))
                    logger.info(response)
                    logger.info(record.answer)

                if metric_accum is None:
                    metric_accum = metric
                    metric_accum["n_samples"] = 1

                else:
                    for k, v in metric.items():
                        metric_accum[k] += v

                    metric_accum["n_samples"] += 1

            format_description = [f"{k}: {v}" for k, v in metric_accum.items()]
            format_description = "; ".join(format_description)
            format_description = f"{task.name}: {format_description}"

            progress.update(task_progress, advance=1, description=format_description)


def main():
    conf = parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    pdims = ParallelDims(dp_replicate=world_size, dp_shard=1, tp=1, pp=1)
    mesh = pdims.build_mesh("cuda")
    logger = get_logger(mesh)

    logger.info(summarize(conf))
    logger.info(f"dp replicate: {pdims.dp_replicate}")

    device = torch.device("cuda")

    logger.info("building model")

    with torch.device("meta"):
        model = instantiate(conf.model.model_infer)

    if conf.model.wrapper is not None:
        logger.info("applying wrapper")
        model = instantiate(conf.model.wrapper)(
            model=model, mesh=mesh, parallel_dims=pdims
        )
        logger.info(str(model))

    model.to_empty(device=device)
    model.eval()

    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=conf.model.checkpoint_path)

    tokenizer = instantiate(conf.model.tokenizer)

    device = "cuda"
    model_conf = conf.model.model_conf
    engine = InferenceEngine(
        model.to(
            device=device,
            dtype=get_torch_dtype(model_conf.dtype),
        ),
        tokenizer,
        ModelConfig(
            n_heads=model_conf.n_heads,
            n_key_value_heads=model_conf.n_key_value_heads,
            head_dim=model_conf.head_dim,
            n_layers=model_conf.n_layers,
            context_len=model_conf.context_len,
        ),
    )

    for task in conf.tasks:
        run_task(engine, task, conf.eval_task, logger)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    model = Model(checkpoint_path=args.model)
    tasks = []

    for task in args.tasks:
        task = load_config(task)

        for subtask in task.tasks:
            tasks.append(Task(**subtask))

    conf = dict(model=model, tasks=tasks)
    eval_task = {}
    apply_overrides(eval_task, args.opts)
    conf["eval_task"] = eval_task
    conf = EvalTaskConfig(**conf)

    return conf


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
