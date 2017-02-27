import numpy as np
import pandas
import matplotlib.pyplot as plt
import codecs
from matplotlib.gridspec import GridSpec
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, f1_score

train = codecs.open('train.csv', 'r', 'utf-8')
info = []
for t in train:
    el = t.split(',')
    i = {}
    i['PassengerId'] = el[0]
    i['Survived'] = el[1]
    i['Pclass'] = el[2]
    i['Name'] = el[3] + ' ' + el[4]
    i['Sex'] = el[5]
    i['Age'] = el[6]
    i['SibSp'] = el[7]
    i['Parch'] = el[8]
    i['Ticket'] = el[9]
    i['Fare'] = el[10]
    i['Cabin'] = el[11]
    #i['Embarked'] = el[12]
    info.append(i)
train.close()

del info[0]

# 1) а) Сравнение по полу

male = {'Survived': 0, 'NotSurvived': 0}
female = {'Survived': 0, 'NotSurvived': 0}
for i in info:
    if i['Sex'] == 'male' and i['Survived'] == '1':
        male['Survived'] += 1
    if i['Sex'] == 'male' and i['Survived'] == '0':
        male['NotSurvived'] += 1
    if i['Sex'] == 'female' and i['Survived'] == '1':
        female['Survived'] += 1
    if i['Sex'] == 'female' and i['Survived'] == '0':
        female['NotSurvived'] += 1

maleperc = {'Survived': round(male['Survived']/(male['Survived'] + male['NotSurvived']), 2),
        'NotSurvived': round(male['NotSurvived']/(male['Survived'] + male['NotSurvived']), 2)}
femaleperc = {'Survived': round(female['Survived']/(female['Survived'] + female['NotSurvived']), 2),
              'NotSurvived': round(female['NotSurvived']/(female['Survived'] + female['NotSurvived']), 2)}

labels = 'Survived', 'Passed Away'
sizes1 = [maleperc['Survived'], maleperc['NotSurvived']]
sizes2 = [femaleperc['Survived'], femaleperc['NotSurvived']]
colors = ['gold', 'lightskyblue']
explode = (0.1, 0)

the_grid = GridSpec(2, 1)
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(sizes1, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Chance of Survival for Males')
plt.axis('equal')
plt.subplot(the_grid[1, 0], aspect=1)
plt.pie(sizes2, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Chance of Survival for Females')
plt.axis('equal')
 
plt.show()

print('1) a)')
print('')
print('On the Titanic, men had a ' + str(maleperc['Survived'])[2:] + '% chance of survival which is a lot less than the chance of survival for women - '
      + str(femaleperc['Survived'])[2:] + '%.')
print('')

# 1) б) Сравнение по классам

class1 = {'Survived': 0, 'NotSurvived': 0}
class2 = {'Survived': 0, 'NotSurvived': 0}
class3 = {'Survived': 0, 'NotSurvived': 0}

for i in info:
    if i['Pclass'] == '1' and i['Survived'] == '1':
        class1['Survived'] += 1
    if i['Pclass'] == '1' and i['Survived'] == '0':
        class1['NotSurvived'] += 1
    if i['Pclass'] == '2' and i['Survived'] == '1':
        class2['Survived'] += 1
    if i['Pclass'] == '2' and i['Survived'] == '0':
        class2['NotSurvived'] += 1
    if i['Pclass'] == '1' and i['Survived'] == '1':
        class3['Survived'] += 1
    if i['Pclass'] == '3' and i['Survived'] == '0':
        class3['NotSurvived'] += 1

perc1 = {'Survived': round(class1['Survived']/(class1['Survived'] + class1['NotSurvived']), 2),
        'NotSurvived': round(class1['NotSurvived']/(class1['Survived'] + class1['NotSurvived']), 2)}
perc2 = {'Survived': round(class2['Survived']/(class2['Survived'] + class2['NotSurvived']), 2),
        'NotSurvived': round(class2['NotSurvived']/(class2['Survived'] + class2['NotSurvived']), 2)}
perc3 = {'Survived': round(class3['Survived']/(class3['Survived'] + class3['NotSurvived']), 2),
        'NotSurvived': round(class3['NotSurvived']/(class3['Survived'] + class3['NotSurvived']), 2)}

labels = 'Survived', 'Passed Away'
sizesb1 = [perc1['Survived'], perc1['NotSurvived']]
sizesb2 = [perc2['Survived'], perc2['NotSurvived']]
sizesb3 = [perc3['Survived'], perc3['NotSurvived']]
colors = ['gold', 'lightskyblue']
explode = (0.1, 0)

the_grid = GridSpec(3, 1)
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(sizesb1, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Chance of Survival for Class 1 Passengers')
plt.axis('equal')
plt.subplot(the_grid[1, 0], aspect=1)
plt.pie(sizesb2, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Chance of Survival for Class 2 Passengers')
plt.axis('equal')
plt.subplot(the_grid[2, 0], aspect=1)
plt.pie(sizesb3, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Chance of Survival for Class 3 Passengers')
plt.axis('equal')
 
plt.show()

print('1) б)')
print('')
print('On the Titanic, first class passengers had a ' + str(perc1['Survived'])[2:]
      + '% chance of survival which is more than the chance of survival for second class passengers  - '
      + str(perc2['Survived'])[2:] + '%. Third class passengers had the least chance of survival - ' + str(perc3['Survived'])[2:] + '%.')
print('')

# 1) в) Сравнение по цене билета в зависимости от социально-экономического класса

df = pandas.read_csv('train.csv', index_col = 'PassengerId')

class1 = df.query('Pclass == 1')['Fare']
class2 = df.query('Pclass == 2')['Fare']
class3 = df.query('Pclass == 3')['Fare']
data = [class1, class2, class3]

c = plt.figure()
a = c.add_subplot(111)
b = a.boxplot(data, showfliers=True)
a.set_xticklabels(['Class 1', 'Class 2', 'Class 3'])
a.set_ylabel('Fare')
 
plt.show()

print('1) в)')
print('')
print('In general, tickets for the first class are the most expensive, tickets for second class cabins are somewhere in the middle and tickets for third class cabins are the cheapest')
print('Though the difference between second class tickets and third class tickets isn\'t that drastic. But the fist class tickets are noticeably more expensive.')
print('Also three passengers bought their ticket for a first class cabin for $5123292. They all survived.')
print('')

# 2)    Сравнение по полу и классу

class1 = {'SurvivedF': 0, 'NotSurvivedF': 0, 'SurvivedM': 0, 'NotSurvivedM': 0}
class2 = {'SurvivedF': 0, 'NotSurvivedF': 0, 'SurvivedM': 0, 'NotSurvivedM': 0}
class3 = {'SurvivedF': 0, 'NotSurvivedF': 0, 'SurvivedM': 0, 'NotSurvivedM': 0}

for i in info:
    if i['Pclass'] == '1' and i['Sex'] == 'male' and i['Survived'] == '1':
        class1['SurvivedM'] += 1
    if i['Pclass'] == '1' and i['Sex'] == 'female' and i['Survived'] == '1':
        class1['SurvivedF'] += 1
    if i['Pclass'] == '1' and i['Sex'] == 'male' and i['Survived'] == '0':
        class1['NotSurvivedM'] += 1
    if i['Pclass'] == '1' and i['Sex'] == 'female' and i['Survived'] == '0':
        class1['NotSurvivedF'] += 1
    if i['Pclass'] == '2' and i['Sex'] == 'male' and i['Survived'] == '1':
        class2['SurvivedM'] += 1
    if i['Pclass'] == '2' and i['Sex'] == 'female' and i['Survived'] == '1':
        class2['SurvivedF'] += 1
    if i['Pclass'] == '2' and i['Sex'] == 'male' and i['Survived'] == '0':
        class2['NotSurvivedM'] += 1
    if i['Pclass'] == '2' and i['Sex'] == 'female' and i['Survived'] == '0':
        class2['NotSurvivedF'] += 1
    if i['Pclass'] == '3' and i['Sex'] == 'male' and i['Survived'] == '1':
        class3['SurvivedM'] += 1
    if i['Pclass'] == '3' and i['Sex'] == 'female' and i['Survived'] == '1':
        class3['SurvivedF'] += 1
    if i['Pclass'] == '3' and i['Sex'] == 'male' and i['Survived'] == '0':
        class3['NotSurvivedM'] += 1
    if i['Pclass'] == '3' and i['Sex'] == 'female' and i['Survived'] == '0':
        class3['NotSurvivedF'] += 1

menperc = [round(class1['SurvivedM']/(class1['SurvivedM'] + class1['NotSurvivedM']), 2),
           round(class2['SurvivedM']/(class2['SurvivedM'] + class2['NotSurvivedM']), 2),
           round(class3['SurvivedM']/(class3['SurvivedM'] + class3['NotSurvivedM']), 2)]
womenperc = [round(class1['SurvivedF']/(class1['SurvivedF'] + class1['NotSurvivedF']), 2),
             round(class2['SurvivedF']/(class2['SurvivedF'] + class2['NotSurvivedF']), 2),
             round(class3['SurvivedF']/(class3['SurvivedF'] + class3['NotSurvivedF']), 2)]
    
fig = plt.figure()
ax = fig.add_subplot(111)
n = 3
ind = np.arange(n)
width = 0.35
men = ax.bar(ind, menperc, width,
                color='blue')
women = ax.bar(ind+width, womenperc, width,
                    color='red')

ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_ylabel('Chance of Survival')
ax.set_title('Chance of Survival based on Sex and Class')
xTickMarks = ['Class'+str(i) for i in range(1,4)]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

ax.legend( (men[0], women[0]), ('Men', 'Women') )
plt.show()

# 3)    Очистите данные

df = pandas.read_csv('train.csv', index_col = 'PassengerId')
df2 = pandas.read_csv('test.csv', index_col = 'PassengerId')

df["Age"] = df["Age"].fillna(df["Age"].median())
df2["Age"] = df2["Age"].fillna(df2["Age"].median())

columns = ['Age', 'Sex', 'Pclass' , 'Fare', 'SibSp', 'Parch']
df2['Survived'] = ''
         
x = df[columns]
x['Sex'] = x['Sex'].map(lambda sex: 1 if sex == 'male' else 0)
x = x.dropna()
y = df['Survived']

# 4)    Разделить данные на обучающую и проверочную выборки (или использовать кросс-валидацию). Будем строить дерево решений.
#       Нужно выбрать параметр модели, который, на ваш взгляд, может повлиять на результат, и выбрать для него возможные значения.
#       Прокомментировать свой выбор. Изменяя в цикле значения параметра, посчитать для каждого случая точноть, полноту, F-меру (может быть, другие метрики?).
#       Изобразить результаты на диаграмме/-ах. Интерпретировать результаты. Нарисовать лучшее дерево.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf = DecisionTreeClassifier(min_samples_split=5)
clf.fit(np.array(x_train), np.array(y_train))
importances = pandas.Series(clf.feature_importances_, index=x_labels)
print(importances)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))

# 5)    Проделать аналогичные операции для модели Random Forest. Сравнить результаты.

model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
scores = []\n",
for t in range(1,100):
    rfc = RandomForestClassifier(n_estimators=t)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
     rfc.fit(X_train, y_train)
     y_pred = rfc.predict(X_test)
plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()
