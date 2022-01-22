import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('/nrs/funke/funkl/results/one_vs_control/test_accuracy.csv',sep='\t',header=None)
df.columns = ['gene','accuracy','checkpoint']
print(f'{df["gene"].nunique()} gene images')

df = df.sort_values('checkpoint').drop_duplicates(subset=['gene'],keep='last')

plt.rcParams['font.size'] = 6

fig,ax = plt.subplots(figsize=(18,9),gridspec_kw=dict(left=0.1,right=0.9,top=0.9,bottom=0.1))

ax = sns.barplot(data=df.sort_values('accuracy',ascending=False),x='gene',y='accuracy', color=sns.color_palette()[0], ax=ax)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,rotation_mode='anchor',ha='right')

ax.tick_params(axis='x',pad=7)
ax.set_ylim([df['accuracy'].min()-0.1,df['accuracy'].max()+0.1])

plt.savefig('/nrs/funke/funkl/figures/one_vs_control_test_accuracy.pdf')