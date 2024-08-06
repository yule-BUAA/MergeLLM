import argparse
import sys
import logging
import os
import time
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.evaluate_llms_utils import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp
from utils.utils import set_random_seed, align_tokenizers_and_embeddings
from utils.load_config import cache_dir


def create_llm(finetuned_model_path, finetuned_model_name, pretrained_model_name, args, logger: logging.Logger, tensor_parallel_size=1, save_model_path=None):
    if finetuned_model_path is not None:
        llm = LLM(model=finetuned_model_path, tokenizer=finetuned_model_path, tensor_parallel_size=args.tensor_parallel_size)
    elif args.weight_mask_rate == 0.0:
        llm = LLM(model=os.path.join(cache_dir, finetuned_model_name), tokenizer=os.path.join(cache_dir, finetuned_model_name), tensor_parallel_size=tensor_parallel_size)
    else:
        assert save_model_path is not None
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name), device_map="cpu")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
        pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, pretrained_model_name))
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))
        finetuned_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name))

        # align the tokens of pretrained and finetuned tokenizer
        align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer, pretrained_config=pretrained_config,
                                        finetuned_models=[finetuned_model], finetuned_tokenizers=[finetuned_tokenizer], finetuned_configs=[finetuned_config],
                                        logger=logger)

        # set random seed to guarantee reproducibility
        set_random_seed(seed=0)
        masked_param_dict = mask_model_weights(finetuned_model=finetuned_model, pretrained_model=pretrained_model,
                                               exclude_param_names_regex=[], weight_format=args.weight_format,
                                               weight_mask_rate=args.weight_mask_rate,
                                               use_weight_rescale=args.use_weight_rescale, mask_strategy=args.mask_strategy)
        # copy the masked parameters to the original model
        for param_name, param_value in finetuned_model.named_parameters():
            if param_name in masked_param_dict:
                param_value.data.copy_(masked_param_dict[param_name])

        logger.info(f"Saving model at {save_model_path}...")
        os.makedirs(save_model_path, exist_ok=True)
        finetuned_model.save_pretrained(save_directory=save_model_path)
        finetuned_tokenizer.save_pretrained(save_directory=save_model_path)
        if args.dataset_name in ["alpaca_eval", "gsm8k", "MATH", "human_eval", "mbpp"]:
            logger.info(f"Model is saved, creating LLM for inference...")
            llm = LLM(model=save_model_path, tokenizer=save_model_path, tensor_parallel_size=tensor_parallel_size)
        else:
            llm = None
    return llm


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference of LLMs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--finetuned_model_path", type=str, help="path of the finetuned language model")
    group.add_argument("--finetuned_model_name", type=str, help="name of the finetuned language model")
    parser.add_argument("--pretrained_model_name", type=str, help="name of the pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset to be used")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    # inference of llm with the given absolute path
    if args.finetuned_model_path is not None:
        assert args.finetuned_model_name is None and args.pretrained_model_name is None
        save_model_name = args.finetuned_model_path.split("/")[-1]
        save_model_path = None
    else:
        # inference of llm by masking delta parameters
        assert args.finetuned_model_name is not None
        if args.weight_mask_rate == 0.0:
            save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}"
        else:
            assert args.pretrained_model_name is not None
            save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}_rescale_{args.use_weight_rescale}"
            if args.mask_strategy == "magnitude":
                save_model_name = f"{save_model_name}_strategy_{args.mask_strategy}"
            if args.weight_format == "finetuned_weight":
                save_model_name = f"{save_model_name}_weight_format_{args.weight_format}"
        save_model_path = f"./save_llms/{args.dataset_name}/{save_model_name}"

    if args.dataset_name == "alpaca_eval":
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["gsm8k", "MATH"]:
        save_gen_results_folder = f"./save_gen_mathematics_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["human_eval", "mbpp"]:
        save_gen_results_folder = f"./save_gen_codes_results/{args.dataset_name}/{save_model_name}"
    else:
        save_gen_results_folder = None

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./save_llm_logs/{args.dataset_name}/{save_model_name}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./save_llm_logs/{args.dataset_name}/{save_model_name}/{str(time.time())}.log")
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

    llm = create_llm(finetuned_model_path=args.finetuned_model_path, finetuned_model_name=args.finetuned_model_name,
                     pretrained_model_name=args.pretrained_model_name, args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                     save_model_path=save_model_path)

    use_other_datasets = False
    if args.dataset_name == "alpaca_eval":
        test_alpaca_eval(llm=llm, generator_model_name=save_model_name, logger=logger, start_index=args.start_index,
                         end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "gsm8k":
        args.test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                   end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "MATH":
        args.test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                            end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "human_eval":
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_gen_results_folder=save_gen_results_folder)
    elif args.dataset_name == "mbpp":
        args.test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger, start_index=args.start_index,
                  end_index=args.end_index, save_gen_results_folder=save_gen_results_folder)
    else:
        use_other_datasets = True
        logger.info(f"Dataset {args.dataset_name} is not supported, just save the model.")

    if not use_other_datasets:
        if args.finetuned_model_path is not None:
            logger.info(f"Inference of {args.finetuned_model_path} on dataset {args.dataset_name} is completed.")
        else:
            logger.info(f"Inference of {args.finetuned_model_name} on dataset {args.dataset_name} is completed.")

    sys.exit()
