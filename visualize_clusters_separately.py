CLUSTER_TO_VISUALIZE = 1  # first check how many clusters were used for clustering algorithm
SHEET_TO_VISUALIZE = 0  # there are 32 sheets

visualization = defaultdict(list)
for i, sheets in enumerate(probs1[:1]):
  for j, samples in enumerate(sheets):
    if labels[j] == CLUSTER_TO_VISUALIZE:  # if distribution pertains to cluster 1
      sorted_probs = sorted(probs1[i+SHEET_TO_VISUALIZE][j], reverse=True)
      sorted_probs = sorted_probs[:30]
      sorted_small = sorted(probs0[i+SHEET_TO_VISUALIZE][j], reverse=True)
      sorted_small = sorted_small[:30]
      for k in range(30):
        visualization["model"].append("big")
        visualization["probs"].append(sorted_probs[k])
        visualization["timestep"].append(j)
        visualization["token_id"].append(k)
      for l in range(30):
        visualization["model"].append("small")
        visualization["probs"].append(sorted_small[l])
        visualization["timestep"].append(j)
        visualization["token_id"].append(l)

full_data = pd.DataFrame.from_dict(visualization)

fig = px.line(full_data, x="token_id", y="probs", animation_frame="timestep", color="model", range_x=[0,29], range_y=[1e-7,1], log_y=True)
fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()