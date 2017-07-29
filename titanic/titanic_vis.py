import numpy as np
import matplotlib.pyplot as plt
import cDTLearner as cDTL
from dataHelperTitanic import Batch, read_data, split_sets
from collections import Counter



np.random.seed(10)


# read in data
X,Y,feature_names,codeing = read_data()


# bar plot of survival rate of men vs. women
m_idx = X[:,2] == 'male'
survival_men = Y[m_idx]
survival_women = Y[np.invert(m_idx)]
p_men = np.mean(survival_men)
p_women = np.mean(survival_women)
plt.bar([.9,1.9],[p_men,1-p_men],width=.19)
plt.bar([1.1,2.1],[p_women,1-p_women], width = .19)
plt.legend(['male','female'])
plt.xticks([1,2],['died','survived'])
plt.ylim([0,1])
plt.show()


# bar plot of original distribution of titles:
title_idx = np.where(feature_names == 'Title')[0]
title_idx = title_idx[0]
titles = X[:,title_idx]
title_counter = Counter(titles)
x_titles = np.arange(0,len(title_counter),1)
y_titles = np.array(list(title_counter.values()))
ticks_titles = np.array(list(title_counter.keys()))
idx = np.flipud(np.argsort(y_titles))
y_titles = y_titles[idx]
ticks_titles = ticks_titles[idx]
plt.bar(x_titles,y_titles,width = .8)
plt.xticks(x_titles, ticks_titles, rotation = 45)
plt.show()


# bar plot of distribution of fare prices vs titles
u_titles = np.unique(titles)
fares_per_title = []
std_in_fares_per_titles = []
groups = []
max_per_title = []
min_per_title = []
f_idx = np.where(feature_names == 'Fare')
f_idx = f_idx[0][0]
for t in u_titles:
	idx = X[:,title_idx] == t
	fares = X[idx,f_idx]
	fares = [float(f) for f in fares if not f == '']
	groups.append(fares)

plt.boxplot(groups)
plt.xticks(np.arange(1,len(groups)+1,1), u_titles, rotation = 45)
plt.show()


# bar plot of distribution of ticket class and price
c_idx = np.where(feature_names == 'Pclass')
c_idx = c_idx[0][0]
u_classes = np.unique(X[:,c_idx])
class_groups = []
for c in u_classes:
	idx = X[:,c_idx] == c
	clss = X[idx,f_idx]
	clss = [float(f) for f in clss if not f == '']
	class_groups.append(clss)

plt.boxplot(class_groups)
plt.xticks(np.arange(1,len(class_groups)+1,1), u_classes)
plt.show()

# bar plot of distribution of age and price
a_idx = np.where(feature_names == 'Age')
ages = X[:,a_idx]
age_groups = np.unique([np.ceil(float(x[0][0])) for x in ages if not x == ''])
age_helper = np.copy(ages)
age_helper[age_helper==''] = 0
age_helper = [float(x) for x in np.squeeze(age_helper)]
fares_per_age_group = []
for age in age_groups:
	idx = age_helper==age 
	fares = X[idx,f_idx]
	fares_per_age_group.append(np.mean([float(f) for f in fares]))

