import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('white',{'font.sans-serif':['simhei','Arial']})

data=sns.load_dataset("iris")
print(data.head())

data.rename(columns={"sepal_length":"萼片长",
                     "sepal_width":"萼片宽",
                     "petal_length":"花瓣长",
                     "petal_width":"花瓣宽",
                     "species":"种类"},inplace=True)
print(data.head())

kind_dict = {
    "setosa":"山鸢尾",
    "versicolor":"杂色鸢尾",
    "virginica":"维吉尼亚鸢尾"
}
data["种类"] = data["种类"].map(kind_dict)
print(data.head(6))
#exit()

sns.pairplot(data)
plt.show()

#markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
sns.pairplot(data,hue="种类", markers=["s", "o", "D"], palette="husl")
plt.show()

# kind: "scatter"与"reg"; diag_kind:"hist"与"kde"
sns.pairplot(data,vars=["萼片长","花瓣长"],  kind="reg")
plt.show()

sns.pairplot(data,x_vars=["萼片长","花瓣宽"], y_vars=["萼片宽","花瓣长"])
plt.show()

sns.pairplot(data, diag_kind="kde",
             plot_kws=dict(s=50, edgecolor="w",color="g",alpha=.5),
             diag_kws=dict(shade=True, color="r"))
plt.show()