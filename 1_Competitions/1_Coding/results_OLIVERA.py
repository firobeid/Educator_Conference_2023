def clean_names(names):
    return [' '.join(name.replace(",", " ").split()).title() for name in names]

names = ['St. Albans', 'St. Albans', 'St Albans', 'St.Ablans', "St.albans", "St. Alans", 'S.Albans',
        'St..Albans', 'S.Albnas', 'St. Albnas', "St.Al bans", 'St.Algans',
        "Sl.Albans", 'St. Allbans', "St, Albans", 'St. Alban', 'St. Alban']

cleaned_names = clean_names(names)
print(cleaned_names)