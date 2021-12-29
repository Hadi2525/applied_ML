import pandas as pd
import seaborn as sns

datacorr = {'16:00': [1,0.403,0.150,0.045,-0.002,-0.031,-0.035,0.009],
            '17:00': [0.403,1,0.534,0.2,0.033,-0.013,-0.001,-0.021],
            '18:00':[0.150, 0.534,1,0.575,0.251,0.096,0.016,0.001],
            '19:00':[0.045,0.2,0.575,1,0.671,0.354,0.163,0.059],
            '20:00':[-0.002,0.033,0.251,0.671,1,0.68,0.41,0.118],
            '21:00':[-0.031,-0.013,0.096,0.354,0.68,1,0.58,0.213],
            '22:00':[-0.035,-0.001,0.016,0.163,0.41,0.58,1,0.444],
            '23:00':[0.009,-0.021,0.001,0.059,0.118,0.213,0.444,1]}

datacorr = pd.DataFrame(datacorr,columns=['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'])
hours = ['16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
i = 0
for ind in datacorr.index:
    datacorr = datacorr.rename(index={ind:hours[i]})
    i+=1

ax = sns.heatmap(
    datacorr,
    vmin=-0.2, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=45,
    horizontalalignment='right'
)
