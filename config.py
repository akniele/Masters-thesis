from GetClusters.differenceMetrics import get_entropy_feature
from GetClusters.differenceMetrics import bucket_diff_top_k


function_feature_dict = dict()
function_list = [(bucket_diff_top_k, 3), (get_entropy_feature, 1)]

for function in function_list:
    function_feature_dict[f"{function[0].__name__}"] = function[1]
