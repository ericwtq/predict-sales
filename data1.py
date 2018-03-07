import csv
import pandas as pd

holiday = csv.reader(open('/Users/ericwtq/Desktop/Machine Learning/final/data/holidays_events.csv', encoding='utf-8'))
next(holiday)
a = dict()
for row in holiday:
    date = row[0]
    locale_name = row[3]
    transferred = row[5]
    if transferred == 'False':
        if date not in a:
            b = []
            b.append(locale_name)
            a[date] = b
        else:
            a[date].append(locale_name)

s = dict()
c = dict()
stores = csv.reader(open('/Users/ericwtq/Desktop/Machine Learning/final/data/stores.csv', encoding='utf-8'))
next(stores)
for row in stores:
    store_id = row[0]
    city = row[1]
    state = row[2]
    if city not in c :
        n = []
        n.append(store_id)
        c[city] = n
    else:
        c[city].append(store_id)
    if state not in s:
        n = []
        n.append(store_id)
        s[state] = n
    else:
        s[state].append(store_id)

    f = []
    for key in a:
        temp = a[key]
        for i in temp:
            if i == 'Ecuador':
                for j in range(1,55):
                    k = []
                    k.append(key)
                    k.append(j)
                    k.append(1)
                    f.append(k)
            if i in s:
                for n in s[i]:
                    k = []
                    k.append(key)
                    k.append(n)
                    k.append(1)
                    f.append(k)
            if i in c:
                for m in c[i]:
                    k = []
                    k.append(key)
                    k.append(m)
                    k.append(1)
                    f.append(k)
print(f)
new_data = pd.DataFrame(f,columns=["date","store_nbr","holiday"])
new_data.to_csv('iv_ass2_data.csv',index=False)
