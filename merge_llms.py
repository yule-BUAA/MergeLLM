import argparse
import sys
import os
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, align_tokenizers_and_embeddings
from utils.load_config import cache_dir


def merge_and_save_models(args: argparse.Namespace, finetuned_model_names: list, models_to_merge: list, finetuned_tokenizers: list,
                          finetuned_configs: list, logger: logging.Logger, merging_method: MergingMethod):
    """
    merge models by merging method with merging_method_name and save them
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param models_to_merge: list, individual models that need to be merged
    :param finetuned_tokenizers: list of finetuned tokenizers
    :param finetuned_configs: list of finetuned configs
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :return:
    """
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name), device_map="cpu")
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))
    pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))

    # align the tokens of pretrained and finetuned tokenizer
    align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer,
                                    pretrained_config=pretrained_config, finetuned_models=models_to_merge,
                                    finetuned_tokenizers=finetuned_tokenizers, finetuned_configs=finetuned_configs, logger=logger)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    merged_model = merging_method.get_merged_model(merged_model=pretrained_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[],
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   slerp_t=args.slerp_t,
                                                   dot_threshold=args.dot_threshold,
                                                   param_density=args.param_density,
                                                   param_value_mask_rate=args.param_value_mask_rate,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   above_average_value_ratio=args.above_average_value_ratio,
                                                   score_calibration_value=args.score_calibration_value)

    # save the merged model parameters and pretrained tokenizer
    save_model_path = f"./save_merge_llms/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/{args.save_model_name}"
    logger.info(f"Saving merged models at {save_model_path}...")
    merged_model.save_pretrained(save_directory=save_model_path)
    pretrained_tokenizer.save_pretrained(save_directory=save_model_path)

    # each each finetuned tokenizer
    for index, finetuned_model_name in enumerate(finetuned_model_names):
        save_tokenizer_path = os.path.join(save_model_path, finetuned_model_name)
        logger.info(f"Saving each finetuned model's tokenizer at {save_tokenizer_path}...")
        finetuned_tokenizers[index].save_pretrained(save_directory=save_tokenizer_path)

    logger.info(f"Merging of {'_'.join(finetuned_model_names)} with method {args.merging_method_name} is completed.")


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument('--models_to_merge', nargs='+', required=True, help='list of models that need to be merged')
parser.add_argument("--pretrained_model_name", type=str, required=True, help="name of the pretrained model")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "slerp_merging", "stock_merging", "breadcrumbs_merging", "ties_merging", "widen_merging", "mask_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--slerp_t", type=float, default=0.5, help="hyperparameter t for slerp merging")
parser.add_argument("--dot_threshold", type=float, default=0.9995, help="threshold for considering the two vectors as colinear")
parser.add_argument("--param_density", type=float, default=0.9, help="density of retained parameters, used for breadcrumbs merging")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "slerp_merging", "stock_merging", "breadcrumbs_merging", "ties_merging", "widen_merging"])
parser.add_argument("--above_average_value_ratio", type=float, default=1.0, help="the ratio above average value")
parser.add_argument("--score_calibration_value", type=float, default=1.0, help="value for score calibration")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()


if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    finetuned_model_names = args.models_to_merge
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    elif args.merging_method_name == "slerp_merging":
        args.save_model_name = f"{args.merging_method_name}_slerp_t_{args.slerp_t}_dot_threshold_{args.dot_threshold}"
    elif args.merging_method_name == "stock_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "breadcrumbs_merging":
        args.save_model_name = f"{args.merging_method_name}_param_density_{args.param_density}_param_value_mask_rate_{args.param_value_mask_rate}_scaling_coefficient_{args.scaling_coefficient}"
    elif args.merging_method_name == "ties_merging":
        args.save_model_name = f"{args.merging_method_name}_param_value_mask_rate_{args.param_value_mask_rate}_scaling_coefficient_{args.scaling_coefficient}"
    elif args.merging_method_name == "widen_merging":
        args.save_model_name = f"{args.merging_method_name}_above_avg_{args.above_average_value_ratio}_score_calibration_{args.score_calibration_value}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        elif args.mask_apply_method == "task_arithmetic":
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        elif args.mask_apply_method == "slerp_merging":
            mask_apply_method_name = f"{args.mask_apply_method}_slerp_t_{args.slerp_t}_dot_threshold_{args.dot_threshold}"
        elif args.mask_apply_method == "stock_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        elif args.mask_apply_method == "breadcrumbs_merging":
            mask_apply_method_name = f"{args.mask_apply_method}_param_density_{args.param_density}_param_value_mask_rate_{args.param_value_mask_rate}_scaling_coefficient_{args.scaling_coefficient}"
        elif args.mask_apply_method == "ties_merging":
            mask_apply_method_name = f"{args.mask_apply_method}_param_value_mask_rate_{args.param_value_mask_rate}_scaling_coefficient_{args.scaling_coefficient}"
        else:
            assert args.mask_apply_method == "widen_merging"
            mask_apply_method_name = f"{args.mask_apply_method}_above_avg_{args.above_average_value_ratio}_score_calibration_{args.score_calibration_value}"
        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}"

    save_merge_log_path = f"./save_merge_llm_logs/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"Configuration is {args}.")

    models_to_merge, finetuned_tokenizers, finetuned_configs = [], [], []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
        finetuned_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)
        finetuned_configs.append(finetuned_config)

    merge_and_save_models(args=args, finetuned_model_names=finetuned_model_names, models_to_merge=models_to_merge, finetuned_tokenizers=finetuned_tokenizers,
                          finetuned_configs=finetuned_configs, logger=logger, merging_method=merging_method)

    sys.exit()
