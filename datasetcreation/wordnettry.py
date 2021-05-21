import pickle5 as pickle
from xml.dom import minidom

sountracks9000 = pickle.load(open('/home/pdbanet/Vionlabs/datasets/sountracks9000.pkl', 'rb'))

mydoc = minidom.parse('/home/pdbanet/Vionlabs/datasets/a-hierarchy.xml')
items = mydoc.getElementsByTagName('categ')

mydoc2 = minidom.parse('/home/pdbanet/Vionlabs/datasets/wn-affect-1.1/a-synsets.xml')
nouns = mydoc2.getElementsByTagName('noun-syn')

mas = []

check = []
for noun in nouns:
    check.append(noun.attributes['categ'].value)
print(check)

for elem in items:
    if elem.attributes['name'].value in check:
        continue
    else:
        mas.append(elem.attributes['name'].value)

print('mas', mas)
print(len(mas))
