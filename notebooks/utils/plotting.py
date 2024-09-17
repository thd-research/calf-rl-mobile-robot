import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


N_SAMPLE_INTER=40
USE_INTERPOLATE=True
STICK_SIZE=15

def get_min_acc_values(array):
   accum_min = []
   for v in array:
      if len(accum_min) == 0:
         accum_min.append(v)
      elif v < accum_min[-1]:
         accum_min.append(v)
      else:
         accum_min.append(accum_min[-1])

   post_median_acc = np.array(accum_min).squeeze()
   return post_median_acc

def plot_cost_ic_learning_curve(df, ax, low_quantile=0.025, high_quantile=0.975, color=None, use_interpolate=USE_INTERPOLATE):
   def quantile_low(series):
      return series.quantile(low_quantile)

   def quantile_high(series):
      return series.quantile(high_quantile)

   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   ic_95_df = group_df.groupby("iteration_id").agg({"accumulative_objective": [quantile_low, quantile_high]})

   # Plot
   if not use_interpolate:
      X_ = ic_95_df.index
      Y_hi = ic_95_df.accumulative_objective.quantile_high
      Y_lo = ic_95_df.accumulative_objective.quantile_low
   else:
      cubic_interpolation_model_hi = interp1d(ic_95_df.index, 
                                             ic_95_df.accumulative_objective.quantile_high, kind = "cubic")
      
      cubic_interpolation_model_lo = interp1d(ic_95_df.index, 
                                             ic_95_df.accumulative_objective.quantile_low, kind = "cubic")
   
      # Plotting the Graph
      X_=np.linspace(ic_95_df.index.min(), ic_95_df.index.max(), N_SAMPLE_INTER)
      Y_hi=cubic_interpolation_model_hi(X_)
      Y_lo=cubic_interpolation_model_lo(X_)
   
   p = ax.fill_between(X_, 
                       Y_hi, Y_lo, 
                       facecolor=color,
                       alpha=0.5)
   
   ax.set_xlabel("Iterations")
   ax.set_ylabel("Accumulated Objectives")

   return p

def plot_cost_ic_learning_curve_mod(df, ax, low_quantile=0.025, high_quantile=0.975, color=None, use_interpolate=USE_INTERPOLATE):
   def quantile_low(series):
      return series.quantile(low_quantile)

   def quantile_high(series):
      return series.quantile(high_quantile)

   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   ic_95_df = group_df.groupby("iteration_id").agg({"accumulative_objective": [quantile_low, quantile_high]})

   # Plot
   if not use_interpolate:
      X_ = ic_95_df.index
      Y_hi = ic_95_df.accumulative_objective.quantile_high
      Y_lo = ic_95_df.accumulative_objective.quantile_low
   else:
      cubic_interpolation_model_hi = interp1d(ic_95_df.index, 
                                              get_min_acc_values(ic_95_df.accumulative_objective.quantile_high.values), 
                                              kind = "cubic")
      
      cubic_interpolation_model_lo = interp1d(ic_95_df.index, 
                                              get_min_acc_values(ic_95_df.accumulative_objective.quantile_low.values), 
                                              kind = "cubic")
   
      # Plotting the Graph
      X_=np.linspace(ic_95_df.index.min(), ic_95_df.index.max(), N_SAMPLE_INTER)
      Y_hi=cubic_interpolation_model_hi(X_)
      Y_lo=cubic_interpolation_model_lo(X_)
   
   p = ax.fill_between(X_, 
                       Y_hi, Y_lo, 
                       facecolor=color,
                       alpha=0.5)
   
   ax.set_xlabel("Iterations")
   ax.set_ylabel("Accumulated Objectives")

   return p

def plot_cost_median_learning_curve(df, ax, color=None, use_interpolate=USE_INTERPOLATE):
   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   median_df = group_df.groupby("iteration_id").agg({"accumulative_objective": "median"})

   # Plot
   if not use_interpolate:
      X_ = median_df.index
      Y_ = median_df.accumulative_objective
   else:
      cubic_interpolation_model = interp1d(median_df.index, 
                                           median_df.accumulative_objective, 
                                           kind = "cubic")
      
      # Plotting the Graph
      X_ = np.linspace(median_df.index.min(), median_df.index.max(), N_SAMPLE_INTER)
      Y_ = cubic_interpolation_model(X_)

   p = ax.plot(X_, Y_, color=color, lw=2.5)
   ax.set_xlabel("Iterations")
   ax.set_ylabel("Accumulated Objectives")
   return p[0]

def plot_cost_median_top_5_learning_curve(df, ax, color=None, use_interpolate=USE_INTERPOLATE):
   def lowest_5_median(series):
      return series.sort_values()[:5].median()

   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   median_df = group_df.groupby("iteration_id").agg({"accumulative_objective": lowest_5_median})

   X_ = median_df.index
   # Y_ = median_df.accumulative_objective

   #### POST process
   Y_ = get_min_acc_values(median_df.values)

   # Plot
   if use_interpolate:
      # # Exponential
      # cubic_interpolation_model = interp1d(X_, 
      #                                      Y_, 
      #                                      kind = "cubic")
      
      # # Plotting the Graph
      # X_ = np.linspace(X_.min(), X_.max(), N_SAMPLE_INTER)
      # Y_ = cubic_interpolation_model(X_)

      # ### ---------- SMOOTHEN {
      # alpha = 0.5

      # smoothed_data = np.zeros_like(Y_)
      # smoothed_data[0] = Y_[0]
      # for t in range(1, len(Y_)):
      #    smoothed_data[t] = alpha * Y_[t] + (1 - alpha) * smoothed_data[t-1]

      # Y_ = smoothed_data
      # ### ---------- }

      # Poly
      degree = 12
      coeffs = np.polyfit(X_, Y_, degree)
      X_ = np.linspace(X_.min(), X_.max(), N_SAMPLE_INTER)
      Y_ = np.polyval(coeffs, X_)

   p = ax.plot(X_, Y_, color=color, lw=2.5)
   ax.set_xlabel("Iterations")
   ax.set_ylabel("Accumulated Objectives")
   return p[0]

def plot_cost_best_learning_curve(df, ax, color=None):
   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   index = group_df[group_df.accumulative_objective == group_df.accumulative_objective.min()].index
   best_exp = df.query(f"experiment_path == \"{index[0][0]}\"").groupby("iteration_id").last()

   p = ax.plot(best_exp.index, best_exp.accumulative_objective, color=color, lw=2.5)
   ax.set_xlabel("Iterations")
   ax.set_ylabel("Accumulated Objectives")
   return p[0]

def plot_accum_cost_ic(df, ax, 
                       low_quantile=0.025, 
                       high_quantile=0.975, 
                       color=None, 
                       use_interpolate=USE_INTERPOLATE,
                       is_truncate=True):
   def quantile_low(series):
      return series.quantile(low_quantile)

   def quantile_high(series):
      return series.quantile(high_quantile)

   tmp_df = df.copy()
   # tmp_df["approximate_time"] = tmp_df.time.apply(lambda x: np.round(x, 1))
   tmp_df["approximate_time"] = tmp_df.time.apply(lambda x: int(x))
   tmp_df["goal_err"] = tmp_df.apply(lambda x: np.linalg.norm([x["x [m]"], x["y [m]"]]), axis=1)
   tmp_df["goal_diff"] = (tmp_df["goal_err"] - 0.01).abs()

   if is_truncate:   
      ic_95_df = tmp_df.groupby("approximate_time").agg({"accumulative_objective": [quantile_low, quantile_high],
                                                         "goal_diff": "min"})
      ic_95_df = ic_95_df.truncate(after=(ic_95_df["goal_diff"]["min"] - 0.01).abs().argmin())
   else:
      ic_95_df = tmp_df.groupby("approximate_time").agg({"accumulative_objective": [quantile_low, quantile_high]})

   # Plot
   if not use_interpolate:
      X_ = ic_95_df.index
      Y_hi = ic_95_df.accumulative_objective.quantile_high
      Y_lo = ic_95_df.accumulative_objective.quantile_low
   else:
      cubic_interpolation_model_hi = interp1d(ic_95_df.index, 
                                             ic_95_df.accumulative_objective.quantile_high, kind = "cubic")
      
      cubic_interpolation_model_lo = interp1d(ic_95_df.index, 
                                             ic_95_df.accumulative_objective.quantile_low, kind = "cubic")
   
      # Plotting the Graph
      X_=np.linspace(ic_95_df.index.min(), ic_95_df.index.max(), N_SAMPLE_INTER)
      Y_hi=cubic_interpolation_model_hi(X_)
      Y_lo=cubic_interpolation_model_lo(X_)
   
   p = ax.fill_between(X_, 
                       Y_hi, Y_lo, 
                       facecolor=color,
                       alpha=0.5)
   
   ax.set_xlabel("Time [s]")
   ax.set_ylabel("Accumulated Objectives")

   return p

def plot_accum_cost_median(df, ax, 
                           color=None, 
                           linestyle=None, 
                           use_interpolate=USE_INTERPOLATE,
                           is_truncate=True):
   tmp_df = df.copy()
   # tmp_df["approximate_time"] = tmp_df.time.apply(lambda x: np.round(x, 1))
   tmp_df["approximate_time"] = tmp_df.time.apply(lambda x: int(x))

   if is_truncate:
      tmp_df["goal_err"] = tmp_df.apply(lambda x: np.linalg.norm([x["x [m]"], x["y [m]"]]), axis=1)
      tmp_df["goal_diff"] = (tmp_df["goal_err"] - 0.01).abs()
      median_df = tmp_df.groupby("approximate_time").agg({"accumulative_objective": "median", "goal_diff": "min"})
      median_df = median_df.truncate(after=(median_df["goal_diff"] - 0.01).abs().argmin())
   else:
      median_df = tmp_df.groupby("approximate_time").agg({"accumulative_objective": "median"})
   

      # Plot
   if not use_interpolate:
      X_ = median_df.index
      Y_ = median_df.accumulative_objective
   else:
      cubic_interpolation_model = interp1d(median_df.index, 
                                           median_df.accumulative_objective, 
                                           kind = "cubic")
      
      # Plotting the Graph
      X_ = np.linspace(median_df.index.min(), median_df.index.max(), N_SAMPLE_INTER)
      Y_ = cubic_interpolation_model(X_)

   p = ax.plot(X_, Y_, color=color, lw=2.5, linestyle=linestyle)

   ax.set_xlabel("Time [s]")
   ax.set_ylabel("Accumulated Objectives")

   return p[0]

def plot_accum_cost_best(df, ax, color=None):
   group_df = df.groupby(["experiment_path", "iteration_id"]).last()
   index = group_df[group_df.accumulative_objective == group_df.accumulative_objective.min()].index

   tmp_df = df.copy()
   tmp_df["approximate_time"] = tmp_df.time.apply(lambda x: np.round(x, 1))
   best_exp = tmp_df.query(f"experiment_path == \"{index[0][0]}\" and iteration_id == {index[0][1]}")

   p = ax.plot(best_exp.time, best_exp.accumulative_objective, color=color, lw=2.5)
   ax.set_xlabel("Time [s]")
   ax.set_ylabel("Accumulated Objectives")
   return p[0]


def plot_trajectories(df, ax, color="blue", linestyle="solid"):
    paths = df.absolute_path.unique()
    tmp_df = df.set_index("absolute_path")
    for p in paths:
        _X = tmp_df.loc[p]["x [m]"]
        _Y = tmp_df.loc[p]["y [m]"]
        pl = ax.plot(_X, _Y, color=color, lw=2.5, zorder=5, linestyle=linestyle)

    ax.tick_params(axis='both', labelsize=STICK_SIZE)

    return pl[0]


# %matplotlib inline
def plot_chosen_best_checkpoint(chosen_df, name, color, linestyle, 
                                get_cost_map_func,
                                target_r=0.1, 
                                y_lim=[-1.25, 0.25], 
                                x_lim=[-1.25, 0.25]):
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.figure(facecolor='white')  # Set figure background color
    fig.patch.set_facecolor('white')
    plt.rcParams.update({"figure.figsize" : (12, 8),
                        "axes.facecolor" : "white",
                        "axes.edgecolor":  "black"})

    # x_lim = y_lim = xy_lim
    X, Y, Z = get_cost_map_func(x_lim, y_lim)

    cs = ax.contourf(X, Y, Z, alpha=0.8, levels=35, cmap="BuPu")
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel('Cost', fontsize=25)

    chosen_pl = plot_trajectories(chosen_df, ax, color, linestyle)

    legend_info = {
        name: chosen_pl,
    }

    legend = ax.legend(legend_info.values(), 
                    legend_info.keys(),
                    facecolor='whitesmoke', 
                    framealpha=0.5)
    legend.set_frame_on(True)

    ax.annotate(text="Target", 
                xy=(0, 0), 
                ha='center', 
                va='center', 
                zorder=15,
                fontsize=15)
    goal_circle = plt.Circle((0, 0), target_r, color="yellowgreen", zorder=10)
    ax.add_artist(goal_circle)

    ax.grid()
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.tick_params(axis='both', labelsize=STICK_SIZE)
    ax.set_xlabel("x [m]", fontsize=25)
    ax.set_ylabel("y [m]", fontsize=25)

    ax.set_title(f"Trajectories of {name} and references", fontsize=25)
    # plt.tight_layout()

    fig.savefig(f"media/report_{name}_trajectory.svg", 
                facecolor="white",
                bbox_inches = 'tight',
                pad_inches = 0)
    


def plot_distance_goal_hist(df, name, color, n_top=20):
    fig, ax = plt.subplots()

    df["goal_err"] = df.apply(lambda x: np.linalg.norm([x["x [m]"], x["y [m]"]]), axis=1)
    group_df = df.groupby(["absolute_path"]).last()
    top_abs_path = group_df.sort_values(by="goal_err").iloc[:n_top, :].index
    top_df = df[df["absolute_path"].isin(top_abs_path)]
    top_group_df = top_df.groupby(["absolute_path"]).last()

    top_group_df.goal_err.plot(kind="hist", 
                                        ax=ax,
                                        title=f"{name} top {n_top}: Distance of the parking position from goal histogram",
                                        color=color,
                                        edgecolor='white')
    
    ax.set_xlabel("Distance [m]")

    fig.savefig(f"media/{name}_top_{n_top}_distance_from_goal.svg",
                facecolor="white",
                bbox_inches='tight',
                pad_inches=0)
    

def plot_distance_goal_comhist(dfs, names, colors, n_top=20, title=""):
    fig, ax = plt.subplots()
    top_group_dfs = []
    max_goal_err = 0

    for df in dfs:
        df["goal_err"] = df.apply(lambda x: np.linalg.norm([x["x [m]"], x["y [m]"]]), axis=1)
        group_df = df.groupby(["absolute_path"]).last()
        top_abs_path = group_df.sort_values(by="goal_err").iloc[:n_top, :].index
        top_df = df[df["absolute_path"].isin(top_abs_path)]
        top_group_df = top_df.groupby(["absolute_path"]).last()
        top_group_dfs.append(top_group_df)

        max_goal_err = max(max_goal_err, top_group_df.goal_err.max())

    bins = np.linspace(0, max_goal_err, int(max_goal_err/0.02))

    b_heights = []
    b_bins = []
    for idx, top_group_df in enumerate(top_group_dfs): 
        h, b = np.histogram(top_group_df.goal_err, bins=bins)
        b_heights.append(h)
        b_bins.append(b)
    
    ax.set_xlabel("Distance [m]")

    width = (bins[1] - bins[0])/4

    ax.bar(b_bins[0][:-1]-width, b_heights[0], width=width, facecolor=colors[0], label=names[0])
    ax.bar(b_bins[1][:-1], b_heights[1], width=width, facecolor=colors[1], label=names[1])
    ax.bar(b_bins[2][:-1]+width, b_heights[2], width=width, facecolor=colors[2], label=names[2])

    ax.set_title(title)
    legend= ax.legend(facecolor='whitesmoke', 
              framealpha=0.5)
    legend.set_frame_on(True)

    fig.savefig("media/combined_hist_{}.svg".format(title.lower().replace(" ", "_")),
                facecolor="white",
                bbox_inches='tight',
                pad_inches=0)