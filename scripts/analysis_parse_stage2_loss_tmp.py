import re, json, csv
from pathlib import Path
log = Path('/data3/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank/logs/stage2_hgb_ckpt9000_6gpu_6datasets_bs16_acc1_max9271_resume_from3000/train.log')
outdir = Path('/data3/yeyuanhao/sp_re_cbp/GeoThinker_zenview_vggt_bank/analysis/stage2_hgb_bs16_final')
outdir.mkdir(parents=True, exist_ok=True)
text = log.read_text(errors='ignore')
metrics=[]
pat = re.compile(r"\{'loss': ([0-9.eE+-]+), 'grad_norm': ([0-9.eE+-]+), 'learning_rate': ([0-9.eE+-]+), 'epoch': ([0-9.eE+-]+)\}")
for m in pat.finditer(text):
    prefix = text[max(0, m.start()-4000):m.start()]
    steps = re.findall(r'(\d+)/(9271)', prefix)
    step = int(steps[-1][0]) if steps else None
    metrics.append({'step': step, 'loss': float(m.group(1)), 'grad_norm': float(m.group(2)), 'learning_rate': float(m.group(3)), 'epoch': float(m.group(4))})
with (outdir/'train_metrics.json').open('w') as f:
    json.dump(metrics, f, indent=2)
with (outdir/'train_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['step','loss','grad_norm','learning_rate','epoch'])
    w.writeheader(); w.writerows(metrics)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs=[m['step'] for m in metrics]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(xs, [m['loss'] for m in metrics], marker='o', linewidth=1.5, label='loss')
    ax1.set_xlabel('optimizer step')
    ax1.set_ylabel('loss')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [m['grad_norm'] for m in metrics], color='orange', marker='x', linewidth=1.0, label='grad_norm')
    ax2.set_ylabel('grad_norm')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='best')
    fig.tight_layout()
    fig.savefig(outdir/'loss_curve.png', dpi=160)
except Exception as e:
    (outdir/'plot_error.txt').write_text(repr(e))
summary={'n':len(metrics), 'first': metrics[:5], 'last': metrics[-10:], 'outdir': str(outdir)}
print(json.dumps(summary, indent=2))
