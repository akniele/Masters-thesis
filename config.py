import time
from GetClusters.differenceMetrics import get_entropy_feature, bucket_diff_top_k, get_top_p_difference

function_feature_dict = dict()
function_list = [(bucket_diff_top_k, 3), (get_entropy_feature, 1), (get_top_p_difference, 1)]

for function in function_list:
    function_feature_dict[f"{function[0].__name__}"] = function[1]


# code taken from https://www.raaicode.com/decorators-in-python/
def timeit(func):
    def wrapper(*args, **kwargs):
        t_start = time.perf_counter()
        temporary_var = func(*args, **kwargs)
        t_end = time.perf_counter()
        print(f"elapsed time:{t_end-t_start}")
        return temporary_var
    return wrapper
