import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from transformers import AutoModel

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from post_processing import all_but_the_top, remove_mean

PLOTS_DIR = "..experiments/plots/"
plot_extension_format = "pdf"

# Global Plots Config
sns.set_style('whitegrid')
sns.set_context("paper")

mpl.use(plot_extension_format)
# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=10)  # family="Times New Roman",
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=10)

# Some esthetics constants
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = 3.0315  # width in inches
fig_height = fig_width * golden_mean  # height in inches
FIG_SIZE = [fig_width, fig_height]


def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def max_normalize(data):
    return data / np.max(data)


def similarity_unused_others_binned(model_name, embeddings, lower_idx=0, upper_idx=1000, step=1000):
    print(f"Computing similarities between unused/rare for {model_name}")

    unused = embeddings[lower_idx:upper_idx]
    exclude = [0, 100, 101, 102, 103]
    mask = np.ones(len(unused), dtype=bool)
    mask[exclude] = False
    unused = unused[mask]

    cosines = []
    inners = []
    upper = step
    lim = len(embeddings)
    while upper <= lim:
        target = embeddings[np.arange(upper - step, upper, step=1)]
        cosines.append(np.average(cosine_similarity(unused, target)))
        inners.append(np.average(np.matmul(unused, target.T)))
        upper += step
        if upper % 5000 == 0:
            print(f"Completed {round(((upper / float(lim)) * 100), 2)}%")

    # process the leftover tokens
    target = embeddings[np.arange(lim - (lim % step), lim, step=1)]

    cosines.append(np.average(cosine_similarity(unused, target)))
    inners.append(np.average(np.matmul(unused, target.T)))

    return inners, cosines


def _plot_unused_others(
        cosines,
        cosines_centered,
        step=1000,
        x_ticks=(0, 10000, 20000, 30000),
        x_tick_labels=('0', '10K', '20K', '30K'),
        label_raw='BERT-base-unc.',
        label_centered='Centered BERT-base-unc.',
        out_file=""
):
    print("Plotting similarity of unused tokens to all other tokens.")
    cpallete = sns.color_palette('colorblind')

    fig, ax = plt.subplots(constrained_layout=True)
    plt.grid(True, which='major')
    fig.set_size_inches(fig_width, fig_height)

    ax.set_xlabel("Index in the Vocabulary")
    ax.set_ylabel('Cosine Similarity')
    ax.set_xlim(0, ((len(cosines) + 1) * step))
    x = np.arange(0, len(cosines) * step, step=step)

    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(list(x_tick_labels))
    # ax.set_yticks(np.arange(0, max(sims), step=0.1), minor=True)
    # x.margins(x=0.2)

    ax.plot(x, cosines, linewidth=1., color=cpallete[0], label=label_raw)
    ax.plot(x, cosines_centered, linewidth=1., linestyle='--', color=cpallete[1], label=label_centered)
    plt.legend(fontsize=7, frameon=False)

    out_file = f"unused_others.{plot_extension_format}" if len(out_file) == 0 else out_file
    fig.savefig(
        out_file,
        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label]
    )


def get_similarities_unused_others_binned(model_name, lower_idx=0, upper_idx=1000, step=1000):
    embeddings = get_embeddings(model_name=model_name)
    embeddings_centered = embeddings - np.mean(embeddings, axis=0)
    inner_products, cosines = similarity_unused_others_binned(
        model_name,
        embeddings,
        lower_idx=lower_idx,
        upper_idx=upper_idx,
        step=step
    )
    inner_products_centered, cosines_centered = similarity_unused_others_binned(
        model_name,
        embeddings_centered,
        lower_idx=lower_idx,
        upper_idx=upper_idx,
        step=step
    )
    return cosines, cosines_centered


def get_embeddings(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model.get_input_embeddings().weight.detach().cpu().numpy()


def compute_singular_values(embeddings: np.ndarray):
    embeddings_centered = remove_mean(embeddings, scale=False)

    # raw
    svd_raw = TruncatedSVD(n_components=embeddings.shape[1] - 1,
                           algorithm='arpack', n_iter=15, tol=0.)
    singular_values_raw = svd_raw.fit(embeddings).singular_values_
    singular_values_raw = max_normalize(singular_values_raw)

    # centered
    svd_centered = TruncatedSVD(n_components=embeddings_centered.shape[1] - 1,
                                algorithm='arpack', n_iter=15, tol=0.)
    singular_values_centered = svd_centered.fit(embeddings_centered).singular_values_
    singular_values_centered = max_normalize(singular_values_centered)

    return singular_values_raw, singular_values_centered


def plot_singular_value_decay(model_name: str,
                              singular_values_raw: np.ndarray,
                              singular_values_centered: np.ndarray,
                              out_file: str):
    fig, ax = plt.subplots(constrained_layout=True)
    plt.grid(True)
    fig.set_size_inches(FIG_SIZE)

    ax.plot(
        np.arange(0, len(singular_values_raw)),
        singular_values_raw,
        linestyle='--',
        label='original'
    )

    ax.plot(
        np.arange(0, len(singular_values_centered)),
        singular_values_centered,
        linestyle='-',
        label='centered',
        color="crimson"
    )

    # annotate
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.annotate(
        model_name.upper(),
        xy=(0.9, 0.9),
        xytext=(0.9, 0.9),
        textcoords='axes fraction',
        fontsize=8,
        horizontalalignment='right',
        verticalalignment='top'
    )

    plt.legend(fontsize=8, loc='center right', frameon=False)

    plt.savefig(
        out_file,
        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label]
    )


def plot_spectral_distribution2d(model_name: str, embeddings: np.ndarray, out_file: str):
    cm = plt.cm.get_cmap('viridis')
    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(FIG_SIZE)

    markers_scale = [0.5 for _ in range(len(embeddings))]

    plot_out = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=[i for i in range(len(embeddings))],
        cmap=cm,
        edgecolor='none',
        alpha=0.5,
        label=model_name.upper(),
        s=markers_scale
    )

    plot_out_bar = plt.colorbar(plot_out)
    ticks_loc = plot_out_bar.ax.get_yticks()
    plot_out_bar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc.tolist()))
    # plot_out_bar.ax.yaxis.set_major_locator(plot_out_bar.ax.get_yticks().tolist())
    bar_labels = ['{:,.1f}'.format(x) + 'K' for x in ticks_loc / 1000]
    plot_out_bar.ax.set_yticklabels(bar_labels)
    plot_out_bar.ax.get_yaxis().labelpad = 15
    plot_out_bar.ax.set_ylabel('Index in Vocabulary', rotation=270, fontsize=8)

    ax.set_xlabel(r'$u_1$')
    ax.set_ylabel(r'$u_2$')

    plt.legend(
        fontsize=8,
        frameon=False,
        loc='best',
        bbox_to_anchor=(0.1, 0., 0.7, 0.3),
        handletextpad=0.2
    )

    # bbox_to_anchor=(0., 0., 0.5, 0.3),
    # bbox_inches='tight',

    plt.savefig(
        out_file,
        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label]
    )


def get_projected(embeddings):
    svd = TruncatedSVD(n_components=2, algorithm='arpack', n_iter=30, tol=0.)
    svd_fit = svd.fit(embeddings)
    components = svd_fit.components_
    projected = embeddings @ components.T
    return projected


def plot_spectral(
        model_name,
        run_raw=True,
        run_centered=True,
        run_post_proc=False,
        top_d=2,
        out_dir="../experiments/plots"
):
    embeddings = get_embeddings(model_name)
    out_file = os.path.join(out_dir, model_name)

    # Plot singular value decay uncentered & centered
    print(f"Generating singular value decay plot for {model_name}")
    singular_values_raw, singular_values_centered = compute_singular_values(embeddings=embeddings)
    plot_singular_value_decay(
        model_name=model_name,
        singular_values_raw=singular_values_raw,
        singular_values_centered=singular_values_centered,
        out_file=f"{out_file}_sv_decay.{plot_extension_format}"
    )

    if run_raw:
        print(f"Generating spectral distribution plot for {model_name}")
        plot_spectral_distribution2d(
            model_name=model_name,
            embeddings=get_projected(embeddings),
            out_file=f"{out_file}_spectral_distribution.{plot_extension_format}"
        )

    if run_centered:
        print(f"Generating [centered] spectral distribution plot for {model_name}")
        plot_spectral_distribution2d(
            model_name=model_name,
            embeddings=get_projected(remove_mean(embeddings, scale=False)),
            out_file=f"{out_file}_spectral_distribution_centered.{plot_extension_format}"
        )

    if run_post_proc:
        print(f"Generating [post-processed] spectral distribution plot for {model_name}")
        plot_spectral_distribution2d(
            model_name=model_name,
            embeddings=all_but_the_top(embeddings=embeddings, top_d=top_d, use_fix=True, scale=False),
            out_file=f"{out_file}_spectral_distribution_post_proc.{plot_extension_format}"
        )


def plot_unused_others(plots_dir):
    bbu = 'bert-base-uncased'
    print(f"Plotting unused to other tokens similarity for {bbu}.")
    cosines, cosines_centered = get_similarities_unused_others_binned(
        model_name=bbu, lower_idx=0, upper_idx=1000, step=1000
    )
    _plot_unused_others(
        cosines=cosines,
        cosines_centered=cosines_centered,
        out_file=f"{os.path.join(plots_dir, bbu)}_unused_others.{plot_extension_format}"
    )


def main():
    models_list = ['gpt2', 'gpt2-medium',
                   'bert-base-cased', 'bert-large-cased',
                   'roberta-base', 'roberta-large']
    plots_dir = os.path.join(os.getcwd(), 'experiments', 'plots')

    plot_unused_others(plots_dir)
    for model_name in models_list:
        plot_spectral(model_name=model_name, out_dir=plots_dir)


if __name__ == "__main__":
    main()
