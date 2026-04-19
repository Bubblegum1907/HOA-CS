import matplotlib.pyplot as plt

def plot_convergence(curves, labels, title="Algorithm Convergence Comparison"):
    """
    Plots multiple convergence curves on a single graph.
    :param curves: List of lists (each inner list is a convergence curve)
    :param labels: List of strings for the legend
    :param title: String for the plot title
    """
    plt.figure(figsize=(10, 6))
    
    styles = ['--', '-.', '-']
    colors = ['#7f8c8d', '#2980b9', '#e74c3c'] # Gray, Blue, Red
    
    for i in range(len(curves)):
        plt.plot(curves[i], label=labels[i], 
                 linestyle=styles[i % len(styles)], 
                 color=colors[i % len(colors)],
                 linewidth=2 if i == len(curves)-1 else 1.5)
    
    plt.yscale('log') 
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig('convergence_comparison.png', dpi=300)
    plt.show()

def plot_accuracy_comparison(accuracies, labels):
    """
    Simple bar chart to compare final test accuracy.
    """
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracies, color=['#95a5a6', '#3498db', '#e67e22'])
    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)')
    plt.title('Final Model Performance')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center')
        
    plt.show()