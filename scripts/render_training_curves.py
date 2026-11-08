"""Render recorded values only, including the metric used for model selection."""
import json
from backend.catalog import ROOT

def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    history=json.loads((ROOT/'reports/training_history.json').read_text());epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    axes[0].plot(epochs,[r['loss'] for r in history],label='Training weighted CE + 0.05 MSE',color='#244f41')
    axes[0].plot(epochs,[r['val_loss'] for r in history],label='Validation unweighted CE',color='#b18757');axes[0].set_title('Different objectives; compare trends')
    axes[1].plot(epochs,[r['accuracy'] for r in history],label='Training accuracy',color='#244f41')
    axes[1].plot(epochs,[r['val_accuracy'] for r in history],label='Validation accuracy',color='#b18757');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_title('Accuracy')
    axes[2].plot(epochs,[r['val_macro_f1'] for r in history],label='Validation macro F1',color='#244f41');axes[2].set_title('Checkpoint selection metric')
    for ax in axes:
        ax.set_xlabel('Completed epoch');ax.legend(fontsize=7);ax.grid(alpha=.15)
        if len(history)>5:ax.axvline(5.5,color='#7a8177',linestyle=':',linewidth=1)
    fig.suptitle('Actual FER2013 training; dotted line marks optimizer restart after epoch 5',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/'reports/training_curves.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
