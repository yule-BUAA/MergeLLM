import argparse
import sys
import logging
import os
import time
from vllm import LLM

from utils.evaluate_llms_utils import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference of merged LLMs")
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
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--evaluate_source_model_name", type=str, required=True, help="evaluate source model name, used for loading tokenizer")
    parser.add_argument("--evaluate_task", type=str, help="evaluate task", default="instruct", choices=["instruct", "math", "code"])
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

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
    elif args.merging_method_name == "difference_merging":
        args.save_model_name = f"{args.merging_method_name}_range_{args.comparison_range}_norm_diff_{args.normalize_param_difference}_above_avg_{args.above_average_value_ratio}_manual_score_{args.manual_important_score_value}"
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

    load_model_path = f"./save_merge_llms/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/{args.save_model_name}"
    llm = LLM(model=load_model_path, tokenizer=os.path.join(load_model_path, args.evaluate_source_model_name), tensor_parallel_size=args.tensor_parallel_size)

    if args.evaluate_task == "instruct":
        logger.info(f"Evaluating merged model on instruct task...")
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/alpaca_eval/{args.save_model_name}"
        test_alpaca_eval(llm=llm, generator_model_name=load_model_path, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_gen_results_folder=save_gen_results_folder)
    elif args.evaluate_task == "math":
        logger.info(f"Evaluating merged model on math task...")
        save_gen_results_folder = f"./save_gen_mathematics_results/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/gsm8k/{args.save_model_name}"
        test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                   start_index=args.start_index, end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
        save_gen_results_folder = f"./save_gen_mathematics_results/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/MATH/{args.save_model_name}"
        test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    else:
        assert args.evaluate_task == "code"
        logger.info(f"Evaluating merged model on code task...")
        save_gen_results_folder = f"./save_gen_codes_results/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/human_eval/{args.save_model_name}"
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_gen_results_folder=save_gen_results_folder)
        save_gen_results_folder = f"./save_gen_codes_results/{args.pretrained_model_name}/{'_'.join(finetuned_model_names)}/mbpp/{args.save_model_name}"
        test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                  save_gen_results_folder=save_gen_results_folder)

    logger.info(f"Inference of merged model {'_'.join(finetuned_model_names)} on {args.evaluate_task} task is completed.")

    sys.exit()
