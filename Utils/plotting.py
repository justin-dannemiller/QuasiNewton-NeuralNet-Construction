
###############################################################################
## Description: Defines the utility function for plotting the loss/error     ##
##              function across iteration and across increasing number of    ##
##              hidden units used                                            ##
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def plot_loss_profile(training_results: dict, plot_title: str, output_plot_name: str):
    """
        Description: Creates a loss plot showing how the training loss/error
                     decreases across time in the given optimization problem across
                     increasing numbers of hidden units
        Args:
            training_results (dict): Dictionary storing the training losses across iterations
                                     and hidden units. Stores number of hidden units as keys
                                     and the list of training losses and iterations for that 
                                     number of hidden units as values. Ex
                                     {2: 
                                        "training_loss": [0.3, ...],
                                        "iterations": [1, 2, ...] 
                                     }
            plot_title (str): Title of plot
            output_plot_name (str): Name of file in which to save the output plot
        Returns:
            None
    """
    plt.figure(figsize=(8,5), facecolor="white")
    for n_hidden, results in training_results.items():
        losses, iterations = results["training_loss"], results["iterations"]
        plt.plot(iterations, losses, label=f"$h={n_hidden}$", linewidth=2)
        
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.title(plot_title, fontsize=16, weight='bold')
    plt.legend(title="Number of Hidden Units", title_fontsize=12, fontsize=11,
               loc="upper right")

    # Format tickmarks
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Add grid
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8)

    plt.tight_layout()

    plt.savefig(output_plot_name, bbox_inches='tight', dpi=300)
    plt.close()

