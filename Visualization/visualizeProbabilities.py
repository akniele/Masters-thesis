"""Visualization of the probability distributions, right now sorted by inclusion distribution of the smaller model"""

from collections import defaultdict
import pandas as pd
import plotly.express as px


prob_vis = defaultdict(list)
for i, sheets in enumerate(probs[:1]): # sheets (one of 32 lists, with 64 lists inside of it)
    for j, samples in enumerate(sheets[1:]): # time steps (a list with 64 dictionaries inside of it)
        #  (a dictionary with keys M1-P and M0-P, ...)

        # big model
        for k in range(len(samples["M1-P"])):
            prob_vis["model"].append("big")
            prob_vis["probs"].append(samples["M1-P"][k])
            prob_vis["timestep"].append(j)
            prob_vis["token_id"].append(k)

        # small model
        for l in range(len(samples["M0-P"])):
            prob_vis["model"].append("small")
            prob_vis["probs"].append(samples["M0-P"][l])
            prob_vis["timestep"].append(j)
            prob_vis["token_id"].append(l)

full_data = pd.DataFrame.from_dict(prob_vis)

fig = px.line(full_data, x="token_id", y="probs", animation_frame="timestep", color="model", range_x=[-1, 129],
              range_y=[1e-15, 1.0], log_y=True)

fig["layout"].pop("updatemenus")  # optional, drop animation buttons
fig.show()
