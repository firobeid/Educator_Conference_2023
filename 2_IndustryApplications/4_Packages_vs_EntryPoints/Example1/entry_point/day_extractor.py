from datetime import date

def diff_date():
    # s = '3/19/2021'
    s = input("Please give the first date (i.e. MM/DD/YYYY)")
    # n = '3/27/2021'
    n = input("Please give the second date (i.e. MM/DD/YYYY)") 
    s = list(map(int,s.split("/"))) 
    n = list(map(int,n.split("/")))
    d = date(s[-1],s[0],s[1]) - date(n[-1],n[0],n[1]) 
    print(f'The difference bw the 2 dates is {d.days}days')
