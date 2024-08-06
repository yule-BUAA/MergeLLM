import re
import os
import random
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerState, PreTrainedTokenizer, PreTrainedModel, PretrainedConfig


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_and_model_for_hf_trainer(trainer: Trainer):
    """
    save the state and model for trainer
    :param trainer: transformers.Trainer to be saved
    :return:
    """
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    # save model at output_dir
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def load_state_and_model_for_hf_trainer(model: nn.Module, load_model_dir: str, map_location: str = None):
    """
    load the state and model for trainer
    :param model: nn.Module, the model to be loaded
    :param load_model_dir: str, the path where the state and model to be loaded
    :param map_location: str, how to remap the storage locations
    :return:
    """
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(os.path.join(load_model_dir, "pytorch_model.bin"), map_location=map_location))
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(os.path.join(load_model_dir, "trainer_state.json"))
    return model, trainer_state


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def align_tokenizers_and_embeddings(pretrained_model: PreTrainedModel, pretrained_tokenizer: PreTrainedTokenizer, pretrained_config: PretrainedConfig,
                                    finetuned_models: list[PreTrainedModel], finetuned_tokenizers: list[PreTrainedTokenizer],
                                    finetuned_configs: list[PretrainedConfig], logger: logging.Logger):
    """
    resize the tokenizer and token embedding, take the union of all the added pad tokens and resize the token embeddings to accommodate all the added pad tokens
    :param pretrained_model: PreTrainedModel, pretrained model
    :param pretrained_tokenizer: PreTrainedTokenizer, pretrained tokenizer
    :param pretrained_config: PretrainedConfig, pretrained config
    :param finetuned_models: list of PreTrainedModel, list of finetuned models
    :param finetuned_tokenizers: list of PreTrainedTokenizer, list of finetuned tokenizers
    :param finetuned_configs: list of PretrainedConfig, list of finetuned configs
    :param logger: Logger, logger
    :return:
    """
    pretrained_vocab_size = pretrained_config.vocab_size
    try:
        # examine the pretrained tokenizer
        models_vocab_size = [pretrained_vocab_size]
        logger.info(f"Vocab size of pretrained model is {pretrained_vocab_size}.")
        pretrained_token_dict = json.loads(pretrained_tokenizer._tokenizer.to_str())
        pretrained_added_pad_tokens = [token_dict for token_dict in pretrained_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
        assert pretrained_added_pad_tokens == []
        models_added_pad_tokens_list = [(True, pretrained_added_pad_tokens)]

        # append the added pad token of finetuned tokenizers into a set
        added_pad_tokens_set = set()
        for index, (finetuned_tokenizer, finetuned_config) in enumerate(zip(finetuned_tokenizers, finetuned_configs)):
            finetuned_vocab_size = finetuned_config.vocab_size
            models_vocab_size.append(finetuned_vocab_size)
            finetuned_token_dict = json.loads(finetuned_tokenizer._tokenizer.to_str())
            finetuned_added_pad_tokens = [token_dict for token_dict in finetuned_token_dict["added_tokens"] if token_dict["id"] >= pretrained_vocab_size]
            logger.info(f"Vocab size of index {index} finetuned model is {finetuned_vocab_size}.")
            logger.info(f"Added pad tokens of index {index} finetuned model is {finetuned_added_pad_tokens}.")
            # the tokens are added in tokenizer config but the corresponding embeddings are missing
            if finetuned_vocab_size - pretrained_vocab_size < len(finetuned_added_pad_tokens):
                logger.warning(f"Vocab size in index {index} finetuned model's config mismatches (less than) number of added tokens.")
                logger.warning(f"Before removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                for _ in range(len(finetuned_added_pad_tokens) - (finetuned_vocab_size - pretrained_vocab_size)):
                    removed_pad_token = finetuned_token_dict['added_tokens'].pop()
                    logger.warning(f"Remove pad token {removed_pad_token}.")
                    assert removed_pad_token["content"] in [token_dict["content"] for token_dict in finetuned_added_pad_tokens]
                finetuned_tokenizer._tokenizer = finetuned_tokenizer._tokenizer.from_str(json.dumps(finetuned_token_dict))
                logger.warning(f"After removing pad tokens, the added tokens are {json.loads(finetuned_tokenizer._tokenizer.to_str())['added_tokens']}.")
                is_matched = False
            else:
                assert finetuned_vocab_size - pretrained_vocab_size == len(finetuned_added_pad_tokens)
                is_matched = True
            for token_dict in finetuned_added_pad_tokens:
                added_pad_tokens_set.add(token_dict["content"])
            models_added_pad_tokens_list.append((is_matched, [token_dict["content"] for token_dict in finetuned_added_pad_tokens]))
        logger.info(f"All added pad tokens of finetuned models are {added_pad_tokens_set}.")

        # align the tokenizers
        aligned_models_vocab_size_set = set()
        for index, (model, tokenizer, model_vocab_size) in enumerate(zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers, models_vocab_size)):
            is_matched = models_added_pad_tokens_list[index][0]
            model_added_pad_tokens_list = models_added_pad_tokens_list[index][1]
            for added_pad_token in added_pad_tokens_set:
                # deal with models like llama-2-13b-code-alpaca, whose finetuned_token_dict['added_tokens'] contains pad tokens and token embeddings are added,
                # but tokenizer.add_special_tokens({"pad_token": "<pad>"}) returns 1 instead of 0 (this model does not have tokenizer.json file)
                if is_matched and added_pad_token in model_added_pad_tokens_list:
                    logger.info(f"Skip added pad token {added_pad_token} of index {index} model since its original added pad tokens and token embeddings are matched.")
                    continue
                num_new_tokens = tokenizer.add_special_tokens({"pad_token": added_pad_token})
                if num_new_tokens > 0:
                    assert num_new_tokens == 1
                    model_vocab_size = model_vocab_size + num_new_tokens

                    model.resize_token_embeddings(new_num_tokens=model_vocab_size)

                    # shape (new_num_tokens, embed_dim)
                    input_embeddings = model.get_input_embeddings().weight.data
                    output_embeddings = model.get_output_embeddings().weight.data

                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg

            logger.info(f"Aligned index {index} model: input token embedding shape {model.get_input_embeddings().weight.shape}, "
                        f"output token embedding shape {model.get_output_embeddings().weight.shape}, "
                        f"tokenizer added tokens {json.loads(tokenizer._tokenizer.to_str())['added_tokens']}.")
            aligned_models_vocab_size_set.add(model.model.embed_tokens.weight.shape)
        assert len(aligned_models_vocab_size_set) == 1
    except Exception as e:
        logger.error(e)
        logger.warning(f"Unable to align tokenizers by default function, using alternative smart_tokenizer_and_embedding_resize function.")
        for model, tokenizer in zip([pretrained_model] + finetuned_models, [pretrained_tokenizer] + finetuned_tokenizers):
            smart_tokenizer_and_embedding_resize(special_tokens_dict={"pad_token": "<special_pad>"},
                                                 tokenizer=tokenizer, model=model, pretrained_vocab_size=pretrained_vocab_size)


def smart_tokenizer_and_embedding_resize(special_tokens_dict: dict, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, pretrained_vocab_size: int):
    """
    alternative function for resizing tokenizer and embedding
    :param special_tokens_dict: dict, dictionary of special tokens
    :param tokenizer: PreTrainedTokenizer, pretrained tokenizer
    :param model: PreTrainedModel, model
    :param model: pretrained_vocab_size, int, vocabulary size of pretrained model
    :return:
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(pretrained_vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
