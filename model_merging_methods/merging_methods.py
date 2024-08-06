from collections import defaultdict
from tqdm import tqdm
import re
import torch
import torch.nn as nn

from model_merging_methods.task_vector import TaskVector
from utils.utils import get_param_names_to_merge
from model_merging_methods.mask_weights_utils import mask_model_weights


class MergingMethod:
    def __init__(self, merging_method_name: str):
        """
        Methods for model merging.
        :param merging_method_name: str, name of the merging method, can be "average_merging", "task_arithmetic", "ties_merging", "magnitude_direction_merging"
        :return:
        """
        self.merging_method_name = merging_method_name

    def copy_params_to_model(self, params: dict, model: nn.Module):
        """
        copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def average_merging(self, models_to_merge: list, exclude_param_names_regex: list):
        """
        average merging method
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return:
        """
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)
        # iterate each individual model that needs to be merged
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

        with torch.no_grad():
            # average merging of individual models' parameters
            merged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(dim=0) for param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return merged_params

    def task_arithmetic(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]

            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

    def slerp_merging(self, models_to_merge: list, exclude_param_names_regex: list, slerp_t: float = 0.5, dot_threshold: float = 0.9995):
        """
        spherical linear interpolation method
        slerp(p, q, t) = \frac{sin((1-t) \theta) \cdot p + sin(t \theta) \cdot q}{sin(\theta)}
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param slerp_t: float, hyperparameter t for slerp merging, where slerp_t=0.0 returns the first model, slerp_t=1.0 returns the second model
        :param dot_threshold: float, threshold for considering the two vectors as colinear
        :return:
        """
        def normalize(input_tensor: torch.Tensor, eps: float = 1e-10):
            """
            normalize the input tensor
            :param input_tensor: Tensor, input tensor to be normalized
            :param eps: float, minimal normalization value
            """
            norm_input_tensor = torch.norm(input_tensor)
            if norm_input_tensor > eps:
                input_tensor = input_tensor / norm_input_tensor
            return input_tensor

        def lerp(v0: torch.Tensor, v1: torch.Tensor, slerp_t: float = 0.5):
            """
            linear interpolation
            :param v0: Tensor, input tensor v0
            :param v1: Tensor, input tensor v1
            :param slerp_t: float, hyperparameter t for slerp merging, where slerp_t=0.0 returns the first model, slerp_t=1.0 returns the first model
            """
            return (1 - slerp_t) * v0 + slerp_t * v1

        assert len(models_to_merge) == 2, "slerp merging expects exactly two models!"

        finetuned_param_dict_list = [{param_name: param_value for param_name, param_value in model_to_merge.named_parameters()} for model_to_merge in models_to_merge]
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(finetuned_param_dict_list[0].keys()), exclude_param_names_regex=exclude_param_names_regex)

        # dict, dictionary of merged model parameters
        merged_params = {}
        with torch.no_grad():
            for param_name in param_names_to_merge:
                # Tensor, shape (2, param_shape)
                stacked_param = torch.stack([finetuned_param_dict[param_name] for finetuned_param_dict in finetuned_param_dict_list], dim=0)
                # Tensor, shape (param_shape, )
                v0, v1 = stacked_param[0], stacked_param[1]

                # Tensor, copy the vectors for reusing
                v0_copy = v0.clone().detach()
                v1_copy = v1.clone().detach()

                # Tensor, normalize the input tensors
                v0 = normalize(input_tensor=v0)
                v1 = normalize(input_tensor=v1)

                # dot product with the normalized vectors
                dot = torch.sum(v0 * v1)

                # if absolute value of dot product larger than dot_threshold (almost 1), vectors are colinear, so use lerp
                if torch.abs(dot) > dot_threshold:
                    merged_params[param_name] = lerp(v0=v0_copy, v1=v1_copy, slerp_t=slerp_t)
                    continue

                # slerp(p, q, t) = \frac{sin((1-t) \theta) \cdot p + sin(t \theta) \cdot q}{sin(\theta)}
                # calculate initial angle between v0 and v1
                theta_0 = torch.arccos(dot)
                sin_theta_0 = torch.sin(theta_0)

                # angle for hyperparameter slerp_t
                theta_t = theta_0 * slerp_t
                sin_theta_t = torch.sin(theta_t)

                # finish the slerp algorithm
                s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                merged_params[param_name] = s0 * v0_copy + s1 * v1_copy

        return merged_params

    def stock_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list):
        """
        stock merging method
        w = t \cdot w_{avg}^N + (1 - t) \cdot w_0
        t = \frac{N cos(\theta)}{1 + (N - 1) \cdot cos(\theta)}
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
        finetuned_param_dict_list = [{param_name: param_value for param_name, param_value in model_to_merge.named_parameters()} for model_to_merge in models_to_merge]
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(finetuned_param_dict_list[0].keys()), exclude_param_names_regex=exclude_param_names_regex)

        # dict, dictionary of merged model parameters
        merged_params = {}
        with torch.no_grad():
            for param_name in param_names_to_merge:
                # Tensor, shape (param_shape, )
                pretrained_param = pretrained_param_dict[param_name]
                # list, each element is a Tensor with shape (param_shape, )
                finetuned_params = [finetuned_param_dict[param_name] for finetuned_param_dict in finetuned_param_dict_list]

                out_shape = pretrained_param.shape
                pretrained_param = pretrained_param.view(-1)
                finetuned_params = [finetuned_param.view(-1) for finetuned_param in finetuned_params]

                # follow the mergekit to implement stock merging
                delta_params = [finetuned_param - pretrained_param for finetuned_param in finetuned_params]

                cos_thetas = []
                for i, delta_param_i in enumerate(delta_params):
                    for j in range(i + 1, len(delta_params)):
                        delta_param_j = delta_params[j]

                        norm_product = torch.norm(delta_param_i, dim=-1) * torch.norm(delta_param_j, dim=-1)
                        cos_theta = ((delta_param_i * delta_param_j).sum(dim=-1) / norm_product.clamp(min=1e-6)).clamp(min=-1, max=1)
                        cos_thetas.append(cos_theta)

                # Tensor, shape (1, )
                cos_theta = torch.stack(cos_thetas, dim=0).mean(dim=0).unsqueeze(dim=-1)
                N = len(finetuned_params)
                # Tensor, shape (1, )
                t = (N * cos_theta) / (1 + (N - 1) * cos_theta)

                param_avg = sum(finetuned_params) / len(finetuned_params)
                merged_param = t * param_avg + (1 - t) * pretrained_param
                merged_params[param_name] = merged_param.reshape(out_shape)

        return merged_params

    def breadcrumbs_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_density: float = 0.9,
                            param_value_mask_rate: float = 0.09, scaling_coefficient: float = 1.0):
        """
        breadcrumbs merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_density: float, density of retained parameters
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        def mask_smallest_largest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_density: float = 0.9, param_value_mask_rate: float = 0.8):
            """
            mask the smallest- and largest-magnitude parameter values (set to zeros) based on param_density and param_value_mask_rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
            :param param_density: float, density of retained parameters
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
            # num_models_to_merge, num_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * (1 - param_density))
            num_mask_smallest_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
            num_mask_largest_params = num_mask_params - num_mask_smallest_params

            assert num_mask_smallest_params >= 0 and num_mask_largest_params >= 0

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_smallest_params-th smallest magnitude element of all the parameters in each individual model
            kth_smallest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_smallest_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
            smallest_mask = flattened_models_to_merge_param.abs() >= kth_smallest_values

            # Tensor, shape (num_models_to_merge, 1), find the (flattened_models_to_merge_param.shape[1] - num_mask_largest_params)-th smallest magnitude element of all the parameters in each individual model
            kth_largest_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=flattened_models_to_merge_param.shape[1] - num_mask_largest_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
            largest_mask = flattened_models_to_merge_param.abs() <= kth_largest_values

            # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve finally
            mask = smallest_mask & largest_mask

            return flattened_models_to_merge_param * mask

        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # dict, dictionary of model parameters
        merged_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict):
                # Tensor, original shape
                param_original_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
                # Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
                flattened_models_to_merge_param = torch.vstack([task_vector.task_vector_param_dict[param_name].flatten() for task_vector in models_to_merge_task_vectors])

                # Tensor, shape (num_models_to_merge, num_params), mask the smallest-magnitude parameter values using param_value_mask_rate
                flattened_models_to_merge_param = mask_smallest_largest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_density=param_density,
                                                                                               param_value_mask_rate=param_value_mask_rate)

                # Tensor, shape (num_params, )
                merged_flattened_param = torch.sum(flattened_models_to_merge_param, dim=0)

                merged_task_vector_param_dict[param_name] = merged_flattened_param.reshape(param_original_shape)

            merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

    def ties_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        """
        ties merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
            """
            mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
            # num_models_to_merge, num_params = flattened_models_to_merge_param.shape
            num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            """
            get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
            """
            disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
            :param param_signs: Tensor, shape (num_params, ), the signs of parameters aggregated across individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_models_to_merge, num_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_params), the preserved parameters
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

            # Tensor, shape (num_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
            # Tensor, shape (num_params, ), the averaged flattened parameters
            merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

            return merged_flattened_param

        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # dict, dictionary of model parameters
        merged_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in tqdm(models_to_merge_task_vectors[0].task_vector_param_dict):
                # Tensor, original shape
                param_original_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
                # Tensor, shape (num_models_to_merge, num_params), flattened parameters of individual models that need to be merged
                flattened_models_to_merge_param = torch.vstack([task_vector.task_vector_param_dict[param_name].flatten() for task_vector in models_to_merge_task_vectors])

                # Tensor, shape (num_models_to_merge, num_params), mask the smallest-magnitude parameter values using param_value_mask_rate
                flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

                # Tensor, shape (num_params, ), get the signs for each parameter in flattened_models_to_merge_param
                param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

                # Tensor, shape (num_params, ), disjoint merge
                merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

                merged_task_vector_param_dict[param_name] = merged_flattened_param.reshape(param_original_shape)

            merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

    def widen_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                      above_average_value_ratio: float = 1.0, score_calibration_value: float = 1.0):
        """
        widen merging method based on weight disentanglement
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        :return:
        """
        def transpose_token_embeddings(param_dict: dict):
            """
            transpose token embeddings
            :param param_dict: dict, dictionary of model parameters
            """
            for param_name in param_dict:
                if param_name == "model.embed_tokens.weight":
                    param_dict[param_name] = param_dict[param_name].transpose(dim0=0, dim1=1)

        def compute_param_magnitude_direction(param_dict: dict, module_dict: dict):
            """
            compute magnitude vector and direction matrix for parameters in param_dict
            :param param_dict: dict, dictionary of model parameters
            :param module_dict: dict, dictionary of model modules
            :return:
            """
            param_magnitude_dict, param_direction_dict = {}, {}
            for param_name in tqdm(param_dict, desc=f"computing parameter magnitude vector and direction matrix"):
                param_last_name = param_name.split(".")[-1]
                module_name = param_name[:-len(f".{param_last_name}")]
                if param_dict[param_name].dim() == 1:
                    # skip trainable vectors for norm or bias
                    continue
                else:
                    assert param_dict[param_name].dim() == 2
                    assert isinstance(module_dict[module_name], nn.Linear) or isinstance(module_dict[module_name], nn.Embedding)
                    # the weight shape is (out_dim, in_dim) for both Linear and transposed nn.Embedding modules
                    # Tensor, shape (in_dim, )
                    magnitude_vector = torch.norm(param_dict[param_name], p=2, dim=0)
                    # Tensor, shape (out_dim, in_dim)
                    direction_matrix = param_dict[param_name] / magnitude_vector
                    param_magnitude_dict[param_name] = magnitude_vector
                    param_direction_dict[param_name] = direction_matrix

            return param_magnitude_dict, param_direction_dict

        def compute_param_magnitude_direction_differences(pretrained_param_magnitude_dict: dict, pretrained_param_direction_dict: dict,
                                                          finetuned_param_magnitude_dict: dict, finetuned_param_direction_dict: dict, module_dict: dict):
            """
            compute the difference of magnitude and direction vectors between pretrained and finetuned models
            :param pretrained_param_magnitude_dict: dict, dictionary of pretrained parameter magnitudes
            :param pretrained_param_direction_dict: dict, dictionary of pretrained parameter directions
            :param finetuned_param_magnitude_dict: dict, dictionary of finetuned parameter magnitudes
            :param finetuned_param_direction_dict: dict, dictionary of finetuned parameter directions
            :param module_dict: dict, dictionary of model modules
            """
            param_magnitude_diff_dict, param_direction_diff_dict = {}, {}
            for param_name in tqdm(pretrained_param_magnitude_dict, desc=f"computing parameter magnitude direction differences"):
                matched_results_list = re.findall(r"model\.layers\.(\d+)\.([\w.]+)\.weight", param_name)
                module_name = param_name[:-len(".weight")]
                if len(matched_results_list) != 0 or param_name == "lm_head.weight":
                    assert isinstance(module_dict[module_name], nn.Linear)
                else:
                    assert param_name == "model.embed_tokens.weight" and isinstance(module_dict[module_name], nn.Embedding)

                # Tensor, shape (in_dim, )
                param_magnitude_diff = torch.abs(finetuned_param_magnitude_dict[param_name] - pretrained_param_magnitude_dict[param_name])
                # Tensor, shape (in_dim, )
                param_direction_diff = 1.0 - torch.cosine_similarity(finetuned_param_direction_dict[param_name], pretrained_param_direction_dict[param_name], dim=0)
                param_magnitude_diff_dict[param_name] = param_magnitude_diff
                param_direction_diff_dict[param_name] = param_direction_diff
            return param_magnitude_diff_dict, param_direction_diff_dict

        def rank_per_param_magnitude_or_direction_within_model(models_to_merge_param_diff: torch.Tensor):
            """
            ranke the magnitude or direction within model
            :param models_to_merge_param_diff: Tensor, shape (num_models_to_merge, in_dim),
            parameter magnitude or direction difference within models that needs to be merged
            """
            # Tensor, shape (num_models_to_merge, in_dim)
            sort_indices = torch.argsort(models_to_merge_param_diff, dim=1, descending=False, stable=True)
            # Tensor, shape (num_models_to_merge, in_dim)
            within_model_significance = (torch.arange(models_to_merge_param_diff.shape[1]) / models_to_merge_param_diff.shape[1]).\
                repeat(models_to_merge_param_diff.shape[0]).reshape(models_to_merge_param_diff.shape)
            # Tensor, shape (num_models_to_merge, in_dim)
            models_to_merge_param_within_model_significance = torch.zeros(within_model_significance.shape)

            # Tensor, shape (num_models_to_merge, in_dim)
            models_to_merge_param_within_model_significance = torch.scatter(input=models_to_merge_param_within_model_significance, dim=1, index=sort_indices, src=within_model_significance)
            return models_to_merge_param_within_model_significance

        def compute_importance_scores(input_significance_tensor: torch.Tensor, above_average_value_ratio: float = 1.0, score_calibration_value: float = 1.0):
            """
            compute importance scores for input significance tensor
            :param input_significance_tensor: Tensor, shape (num_models_to_merge, in_dim), input significance tensor
            :param above_average_value_ratio: float, the ratio above average value
            :param score_calibration_value: float, value for score calibration
            """
            # Tensor, shape (num_models_to_merge, in_dim)
            importance_scores = torch.softmax(input_significance_tensor, dim=0)
            # assign scores for important parameters based on above average ratio
            # Tensor, shape (num_models_to_merge, 1)
            avg_input_significance_tensor = torch.mean(input_significance_tensor, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, in_dim)
            mask = input_significance_tensor > (avg_input_significance_tensor * above_average_value_ratio)
            importance_scores[mask] = score_calibration_value

            return importance_scores

        def merge_per_param_magnitude_direction(models_to_merge_delta_param: torch.Tensor, pretrained_param: torch.Tensor,
                                                models_to_merge_param_magnitude_rank: torch.Tensor, models_to_merge_param_direction_rank: torch.Tensor,
                                                above_average_value_ratio: float = 1.0, score_calibration_value: float = 1.0):
            """
            merge the magnitude and direction for each parameter
            :param models_to_merge_delta_param: Tensor, shape (num_models_to_merge, out_dim, in_dim), delta parameter of models that need to be merged
            :param pretrained_param: Tensor, shape (out_dim, in_dim), parameter of pre-trained model
            :param models_to_merge_param_magnitude_rank: Tensor, shape (num_models_to_merge, in_dim), parameter magnitude rank of models that need to be merged
            :param models_to_merge_param_direction_rank: Tensor, shape (num_models_to_merge, in_dim), parameter direction rank of models that need to be merged
            :param above_average_value_ratio: float, the ratio above average value
            :param score_calibration_value: float, value for score calibration
            """
            # Tensor, shape (num_models_to_merge, in_dim)
            magnitude_scores = compute_importance_scores(input_significance_tensor=models_to_merge_param_magnitude_rank,
                                                         above_average_value_ratio=above_average_value_ratio,
                                                         score_calibration_value=score_calibration_value)
            # Tensor, shape (num_models_to_merge, in_dim)
            direction_scores = compute_importance_scores(input_significance_tensor=models_to_merge_param_direction_rank,
                                                         above_average_value_ratio=above_average_value_ratio,
                                                         score_calibration_value=score_calibration_value)

            weight_scores = 0.5 * (magnitude_scores + direction_scores)
            # Tensor, shape (out_dim, in_dim)
            merged_delta_param = (models_to_merge_delta_param * weight_scores.unsqueeze(dim=1)).sum(dim=0)
            merged_param = pretrained_param + merged_delta_param

            return merged_param

        def merge_param_magnitude_direction(models_to_merge_param_magnitude_direction_diff_tuples: list, pretrained_param_dict: dict, finetuned_param_dict_list: list,
                                            models_to_merge_task_vectors: list, exclude_param_names_regex: list, above_average_value_ratio: float = 1.0,
                                            score_calibration_value: float = 1.0):
            """
            merge parameters by magnitudes and directions
            :param models_to_merge_param_magnitude_direction_diff_tuples: list of tuples, each tuple contains parameter magnitude and direction difference
            :param pretrained_param_dict: dict, dictionary of pretrained parameters
            :param finetuned_param_dict_list: list, list of dictionaries of finetuned parameters
            :param models_to_merge_task_vectors: list, list of task vectors
            :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
            :param above_average_value_ratio: float, the ratio above average value
            :param score_calibration_value: float, value for score calibration
            """
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            # models_to_merge_param_magnitude_diff_tuple, tuple of param_magnitude_diff_dict dictionaries, length is number of models to merge
            # models_to_merge_param_direction_diff_tuple, tuple of param_direction_diff_dict dictionaries, length is number of models to merge
            models_to_merge_param_magnitude_diff_tuple, models_to_merge_param_direction_diff_tuple = zip(*models_to_merge_param_magnitude_direction_diff_tuples)
            param_names_merged_by_magnitude_direction = list(models_to_merge_param_magnitude_diff_tuple[0].keys())

            merged_params = {}
            param_merging_importance_dict = {}
            for param_name in param_names_to_merge:
                # parameters that can be merged by magnitudes and directions
                if param_name in param_names_merged_by_magnitude_direction:
                    # Tensor, shape (num_models_to_merge, out_dim, in_dim)
                    models_to_merge_delta_param = torch.stack([models_to_merge_task_vector.task_vector_param_dict[param_name]
                                                               for models_to_merge_task_vector in models_to_merge_task_vectors], dim=0)
                    # Tensor, shape (num_models_to_merge, in_dim)
                    models_to_merge_param_magnitude_diff = torch.stack([model_to_merge_param_magnitude_diff[param_name]
                                                                        for model_to_merge_param_magnitude_diff in models_to_merge_param_magnitude_diff_tuple], dim=0)
                    # Tensor, shape (num_models_to_merge, in_dim)
                    models_to_merge_param_direction_diff = torch.stack([model_to_merge_param_direction_diff[param_name]
                                                                        for model_to_merge_param_direction_diff in models_to_merge_param_direction_diff_tuple], dim=0)

                    # rank magnitudes and directions within each model
                    # Tensor, shape (num_models_to_merge, in_dim)
                    models_to_merge_param_magnitude_rank = rank_per_param_magnitude_or_direction_within_model(models_to_merge_param_diff=models_to_merge_param_magnitude_diff)
                    models_to_merge_param_direction_rank = rank_per_param_magnitude_or_direction_within_model(models_to_merge_param_diff=models_to_merge_param_direction_diff)

                    # Tensor, shape (out_dim, in_dim)
                    merged_params[param_name] = merge_per_param_magnitude_direction(models_to_merge_delta_param=models_to_merge_delta_param,
                                                                                    pretrained_param=pretrained_param_dict[param_name],
                                                                                    models_to_merge_param_magnitude_rank=models_to_merge_param_magnitude_rank,
                                                                                    models_to_merge_param_direction_rank=models_to_merge_param_direction_rank,
                                                                                    above_average_value_ratio=above_average_value_ratio,
                                                                                    score_calibration_value=score_calibration_value)
                # parameters that not required to be merged by magnitudes and directions (vector-like weights like the normalization layers)
                else:
                    # Tensor, shape (num_models_to_merge, in_dim)
                    models_to_merge_param = torch.stack([finetuned_param_dict[param_name] for finetuned_param_dict in finetuned_param_dict_list], dim=0)
                    # Tensor, shape (num_models_to_merge, in_dim)
                    models_to_merge_delta_param = torch.stack([models_to_merge_task_vector.task_vector_param_dict[param_name]
                                                               for models_to_merge_task_vector in models_to_merge_task_vectors], dim=0)
                    param_diff = torch.abs(models_to_merge_param - pretrained_param_dict[param_name])

                    # Tensor, shape (num_models_to_merge, in_dim)
                    param_scores = compute_importance_scores(input_significance_tensor=param_diff, above_average_value_ratio=above_average_value_ratio,
                                                             score_calibration_value=score_calibration_value)

                    # Tensor, shape (in_dim, )
                    merged_delta_param = (models_to_merge_delta_param * param_scores).sum(dim=0)
                    merged_params[param_name] = pretrained_param_dict[param_name] + merged_delta_param

            return merged_params

        module_dict = {module_name: module for module_name, module in merged_model.named_modules()}
        pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
        finetuned_param_dict_list = [{param_name: param_value for param_name, param_value in model_to_merge.named_parameters()} for model_to_merge in models_to_merge]
        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # transpose token embeddings
        transpose_token_embeddings(param_dict=pretrained_param_dict)
        for finetuned_param_dict in finetuned_param_dict_list:
            transpose_token_embeddings(param_dict=finetuned_param_dict)
        for models_to_merge_task_vector in models_to_merge_task_vectors:
            transpose_token_embeddings(param_dict=models_to_merge_task_vector.task_vector_param_dict)

        with torch.no_grad():
            models_to_merge_param_magnitude_direction_diff_tuples = []
            # compute the magnitude vectors, direction matrices and the difference compared to pre-trained model
            pretrained_param_magnitude_dict, pretrained_param_direction_dict = compute_param_magnitude_direction(param_dict=pretrained_param_dict, module_dict=module_dict)
            for finetuned_param_dict in finetuned_param_dict_list:
                finetuned_param_magnitude_dict, finetuned_param_direction_dict = compute_param_magnitude_direction(param_dict=finetuned_param_dict, module_dict=module_dict)
                param_magnitude_direction_diff_tuple = compute_param_magnitude_direction_differences(pretrained_param_magnitude_dict=pretrained_param_magnitude_dict,
                                                                                                     pretrained_param_direction_dict=pretrained_param_direction_dict,
                                                                                                     finetuned_param_magnitude_dict=finetuned_param_magnitude_dict,
                                                                                                     finetuned_param_direction_dict=finetuned_param_direction_dict,
                                                                                                     module_dict=module_dict)
                models_to_merge_param_magnitude_direction_diff_tuples.append(param_magnitude_direction_diff_tuple)

            # merge parameters based on computed difference
            merged_params = merge_param_magnitude_direction(models_to_merge_param_magnitude_direction_diff_tuples=models_to_merge_param_magnitude_direction_diff_tuples,
                                                            pretrained_param_dict=pretrained_param_dict,
                                                            finetuned_param_dict_list=finetuned_param_dict_list,
                                                            models_to_merge_task_vectors=models_to_merge_task_vectors,
                                                            exclude_param_names_regex=exclude_param_names_regex,
                                                            above_average_value_ratio=above_average_value_ratio,
                                                            score_calibration_value=score_calibration_value)
            # transpose token embeddings to recover
            transpose_token_embeddings(param_dict=merged_params)

        return merged_params

    def merging_models(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0,
                       slerp_t: float = 0.5, dot_threshold: float = 0.9995, param_density: float = 0.9, param_value_mask_rate: float = 0.8,
                       weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                       mask_apply_method: str = "average_merging", above_average_value_ratio: float = 1.0, score_calibration_value: float = 1.0):
        """
        model merging methods
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param slerp_t: float, hyperparameter t for slerp merging, where slerp_t=0.0 returns the first model, slerp_t=1.0 returns the first model
        :param dot_threshold: float, threshold for considering the two vectors as colinear
        :param param_density: float, density of retained parameters
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        :return:
        """
        if self.merging_method_name == "average_merging":
            merged_params = self.average_merging(models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "task_arithmetic":
            merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                 scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "slerp_merging":
            merged_params = self.slerp_merging(models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                               slerp_t=slerp_t, dot_threshold=dot_threshold)
        elif self.merging_method_name == "stock_merging":
            merged_params = self.stock_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "breadcrumbs_merging":
            merged_params = self.breadcrumbs_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                     param_density=param_density, param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "ties_merging":
            merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                              param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "widen_merging":
            merged_params = self.widen_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                               above_average_value_ratio=above_average_value_ratio, score_calibration_value=score_calibration_value)
        elif self.merging_method_name == "mask_merging":
            with torch.no_grad():
                for model_to_merge, weight_mask_rate in zip(models_to_merge, weight_mask_rates):
                    # for each individual model, mask its weight
                    masked_param_dict = mask_model_weights(finetuned_model=model_to_merge, pretrained_model=merged_model,
                                                           exclude_param_names_regex=exclude_param_names_regex, weight_format=weight_format,
                                                           weight_mask_rate=weight_mask_rate, use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy)
                    self.copy_params_to_model(params=masked_param_dict, model=model_to_merge)
            if mask_apply_method == "average_merging":
                merged_params = self.average_merging(models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
            elif mask_apply_method == "task_arithmetic":
                merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                     scaling_coefficient=scaling_coefficient)
            elif mask_apply_method == "slerp_merging":
                merged_params = self.slerp_merging(models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                   slerp_t=slerp_t, dot_threshold=dot_threshold)
            elif mask_apply_method == "stock_merging":
                merged_params = self.stock_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
            elif mask_apply_method == "breadcrumbs_merging":
                merged_params = self.breadcrumbs_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                         param_density=param_density, param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
            elif mask_apply_method == "ties_merging":
                merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                  param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
            elif mask_apply_method == "widen_merging":
                merged_params = self.widen_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                   above_average_value_ratio=above_average_value_ratio, score_calibration_value=score_calibration_value)
            else:
                raise NotImplementedError(f"unsupported for mask_apply_method {mask_apply_method}!")
        else:
            raise NotImplementedError(f"unsupported for merging_method_name {self.merging_method_name}!")
        return merged_params

    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0,
                         slerp_t: float = 0.5, dot_threshold: float = 0.9995, param_density: float = 0.9, param_value_mask_rate: float = 0.8,
                         weight_format: str = "delta_weight", weight_mask_rates: list = None, use_weight_rescale: bool = True, mask_strategy: str = "random",
                         mask_apply_method: str = "average_merging", above_average_value_ratio: float = 1.0, score_calibration_value: float = 1.0):
        """
        merge the parameters of models_to_merge to merged_model
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :param slerp_t: float, hyperparameter t for slerp merging, where slerp_t=0.0 returns the first model, slerp_t=1.0 returns the first model
        :param dot_threshold: float, threshold for considering the two vectors as colinear
        :param param_density: float, density of retained parameters
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
        :param weight_mask_rates: list, list of weight mask rates
        :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
        :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
        :param mask_apply_method: str, merging method that the mask strategy applies
        :param above_average_value_ratio: float, the ratio above average value
        :param score_calibration_value: float, value for score calibration
        :return:
        """
        # merged_params, dict of parameters
        merged_params = self.merging_models(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                            scaling_coefficient=scaling_coefficient, slerp_t=slerp_t, dot_threshold=dot_threshold,
                                            param_density=param_density, param_value_mask_rate=param_value_mask_rate, weight_format=weight_format,
                                            weight_mask_rates=weight_mask_rates, use_weight_rescale=use_weight_rescale, mask_strategy=mask_strategy,
                                            mask_apply_method=mask_apply_method, above_average_value_ratio=above_average_value_ratio,
                                            score_calibration_value=score_calibration_value)
        self.copy_params_to_model(params=merged_params, model=merged_model)

        return merged_model
