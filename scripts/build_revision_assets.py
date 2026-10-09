"""Build publication tables, the system diagram, and a supplemental timeline."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'.cache'))
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
RESULTS=ROOT/'results/reviewer_revision'

def system_and_selection(physical_only=False):
    """Schematic of the implemented physical resources and shared plan search.

    The paths illustrate supported capabilities, not measured geometry, timing,
    bandwidth, or the number of satellites in a particular instance.
    """
    if physical_only:
        fig, ax = plt.subplots(figsize=(3.5,1.95))
        axes = [ax]
    else:
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.55))
    fig.subplots_adjust(left=.01,right=.99,top=.87,bottom=.04,wspace=.16)
    ink='#243746';blue='#dceaf5';green='#e2efe8';orange='#fff0d8'
    def box(ax,x,y,w,h,label,color=blue,size=8):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
                     boxstyle='round,pad=0.03,rounding_size=0.10',
                     facecolor=color,edgecolor=ink,linewidth=.8))
        ax.text(x,y,label,ha='center',va='center',fontsize=size,color=ink)
    def arrow(ax,start,end,color=ink):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',
                    color=color,lw=.9,shrinkA=1,shrinkB=1))
    for ax in axes:ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    a=axes[0]
    if not physical_only:
        a.set_title('(a) Observation and delivery resources',fontsize=9,pad=8)
    box(a,1.5,5.25,2.4,.68,'EO tile A',orange)
    box(a,5.1,5.25,2.4,.68,'EO tile B',orange)
    box(a,1.5,3.5,2.4,.68,'Satellite 1')
    box(a,5.1,3.5,2.4,.68,'Satellite 2')
    box(a,8.55,3.5,2.4,.68,'Satellite 3')
    arrow(a,(1.5,4.88),(1.5,3.87));a.text(1.7,4.35,'observe',fontsize=7.5)
    arrow(a,(5.1,4.88),(5.1,3.87));a.text(5.3,4.35,'observe',fontsize=7.5)
    arrow(a,(6.35,3.5),(7.3,3.5),'#ae554b')
    a.text(8.1,4.60,'One-hop ISL\nfallback',ha='center',va='center',fontsize=7.5,color='#ae554b')
    box(a,1.5,1.1,2.8,.68,'Ground station 1',green,7)
    box(a,8.55,1.1,2.8,.68,'Ground station 2',green,7)
    arrow(a,(1.5,3.13),(1.5,1.47),'#257450')
    a.text(1.75,2.27,'Data delivery over\ntwo downlink windows',va='center',fontsize=7,color='#257450')
    arrow(a,(8.55,3.13),(8.55,1.47),'#257450')
    a.text(8.3,2.27,'Relay\ndownlink',ha='right',va='center',fontsize=7.5,color='#257450')
    a.text(5,.17,'All observation data must reach ground by the request deadline.',
           ha='center',fontsize=7)
    if physical_only:
        for ext in ['pdf', 'png']:
            fig.savefig(FIG / f'racs_observation_delivery.{ext}', dpi=260,
                        bbox_inches='tight', pad_inches=.02)
        plt.close(fig)
        return
    a=axes[1];a.set_title('(b) Implemented plan generation and selection',fontsize=9,pad=8)
    box(a,5,5.48,9.2,.60,'Requests + contacts + current resource ledger',orange,8)
    box(a,5,4.40,9.2,.82,'Shared generator: 3 observation rules\n'+r'$\times$ 2 delivery rules; direct first, then relay',blue,8)
    box(a,5,3.13,9.2,.78,'Check windows, deadlines, conflicts and buffers;\nretain distinct feasible complete plans',blue,8)
    box(a,3.6,1.83,6.4,.87,'Nonempty pool: select a plan\nRACS: highest score',green,7.6)
    box(a,8.65,1.83,2.5,.87,'Empty pool:\nreject request',orange,7.5)
    box(a,3.6,.51,6.4,.66,'Commit plan; next request',green,7.5)
    for b,e in [((5,5.14),(5,4.85)),((5,3.94),(5,3.55)),
                ((3.6,2.70),(3.6,2.3)),((8.65,2.70),(8.65,2.3)),
                ((3.6,1.36),(3.6,.87))]:arrow(a,b,e)
    for ext in ['pdf','png']:fig.savefig(FIG/f'racs_system_and_selection.{ext}',dpi=260)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    system_and_selection()
    system_and_selection(physical_only=True)
    fig,ax=plt.subplots(figsize=(3.5,1.8),layout='constrained')
    colors={'obs':'#4477AA','dl':'#228833','isl':'#CC6677'}
    for y,b,e,label,kind in [(2,10,20,'Tile A','obs'),(1,10,20,'Tile B','obs'),
                            (2,30,40,'0.3','dl'),(2,50,60,'0.3','dl'),
                            (1,20,28,'ISL','isl'),(0,20,28,'Rx','isl'),(0,40,50,'0.6 GB','dl')]:
        ax.broken_barh([(b,e-b)],(y-.27,.54),facecolors=colors[kind])
        ax.text((b+e)/2,y,label,ha='center',va='center',color='white',fontsize=7)
    ax.annotate('',xy=(24,.27),xytext=(24,.73),arrowprops=dict(arrowstyle='->',lw=1,color=colors['isl']))
    ax.text(29,.48,'relay',fontsize=7,color=colors['isl'])
    ax.axvline(60,color='black',linestyle='--',linewidth=1)
    ax.set_xlim(0,64);ax.set_ylim(-.55,2.55);ax.set_yticks([0,1,2],['Satellite 3','Satellite 2','Satellite 1'])
    ax.set_xticks([0,10,20,30,40,50,60]);ax.set_xlabel('Time (s); deadline = 60 s')
    ax.set_title('One request: two tiles, 0.6 GB each',fontsize=9)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    fig.savefig(FIG/'full_chain_example.pdf');fig.savefig(FIG/'full_chain_example.png',dpi=240);plt.close(fig)
    frame=pd.read_csv(RESULTS/'comparison_summary.csv')
    names={'single_margin':'Single margin','full_earliest':'Full earliest','full_margin':'Full margin',
           'full_congestion':'Full congestion','full_random':'Full random','full_learned':'Full learned','full_racs':'RACS default'}
    selectors={'Single margin': 'Max. $R_q$', 'Full earliest': 'Min. $t_q$', 'Full margin': 'Max. $R_q$', 'Full congestion': 'Min. $\\rho_q$', 'Full random': 'Uniform', 'Full learned': 'Pred. reward', 'RACS default': 'Default score'}
    rows=[]
    for policy,name in names.items():
        base=frame.query('stress=="base" and policy==@policy').iloc[0]
        joint=frame.query('stress=="combined" and policy==@policy').iloc[0]
        rows.append(f"{name} & {selectors[name]} & {base.completion:.3f} & {joint.completion:.3f} & {joint.reward:.3f} & {joint.congestion:.3f} & {joint.fairness:.3f} & {joint.runtime_ms:.1f} \\\\")
    (OUT/'comparison_rows.tex').write_text('\\newcommand{\\ComparisonRows}{%\n'+'\n'.join(rows)+'%\n}\n')
    stats=pd.read_csv(RESULTS/'paired_statistics.csv');rows=[]
    for policy in ['single_margin','full_earliest','full_margin','full_congestion','full_learned','full_random']:
        r=stats[(stats.stress=='combined') & (stats.comparison=='full_racs vs '+policy)].iloc[0]
        rows.append(f"{names[policy]} & {100*r.delta:+.2f} & [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}] & {r.p:.3f} & {r.p_holm:.2f} \\\\")
    (OUT/'statistics_rows.tex').write_text('\\newcommand{\\StatisticsRows}{%\n'+'\n'.join(rows)+'%\n}\n')
    from plot_matched_results import plot_matched_results
    plot_matched_results(pd.read_csv(RESULTS/'instance_results.csv'), stats, FIG)
    for name in ['weight_sensitivity.pdf']:
        (FIG/name).write_bytes((RESULTS/'figures'/name).read_bytes())

if __name__=='__main__':main()
