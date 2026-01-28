def computerLPS(pattenrn):
    n = len(pattenrn)
    lps = [0]*n 
    len_ = 0
    i = 1

    while i< n:
        if pattenrn[i] == pattenrn[len_]:
            len_+=1
            lps[i] = len_
            i+=1
        else:
            if len_ != 0:
                len_ = lps[len_-1]
            else:
                lps[i] = 0
                i+=1
    return lps

def stringmatch(s, pat):
    lps = computerLPS(pat)
    n = len(s)
    m = len(pat) 

    i = 0
    j = 0
    matches = []
    while i<n:
        if s[i] == pat[j]:
            i+=1
            j+=1
            if j == m:
                matches.append((i,i-j))
                j = lps[j-1]
        else:
            if j != 0:
                j = lps[j-1]
            else:
                i+=1
    return matches
    

s = "ABC ABCDAB ABCDABCDABDE"
pat = "ABCDABD"

if __name__ =="__main__":
    print(stringmatch(s, pat))