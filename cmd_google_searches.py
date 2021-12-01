import os

sample = 'start www.google.com/search?q=SEARCHTERM'
#
# carDict = {
#     'toyota': 'prius  corolla'
# }
#
# for make in carDict:
#     for model in carDict[make].split():
#         for year in range(2007, 2022):
#             #print(r'start www.google.com/search?q={}+{}+{}'.format(year, make, model))
#             os.system(r'start www.google.com/search?q={}+{}+{}'.format(year, make, model))


for year in range(2013, 2022):
    # os.system(r'start www.google.com/search?q={}+toyota+prius'.format(year))
    os.system(r'start www.google.com/search?q={}+honda+accord'.format(year))
