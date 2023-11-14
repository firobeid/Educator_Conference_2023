

names = ['St. Albans',
        'St. Albans', 
        'St Albans', 
        'St.Ablans',
        "St.albans", 
        "St. Alans", 'S.Albans',
        'St..Albans', 'S.Albnas', 
        'St. Albnas', "St.Al bans", 'St.Algans',
        "Sl.Albans", 'St. Allbans', "St, Albans", 'St. Alban', 'St. Alban']

new_names = [' '.join(name.replace('.', ' ').split()).title() for name in names]
new_names

