from numpy import zeros, flipud, double




def build(method):
    if method == 'LT':
        s = 1
        a = zeros(s)
        b = zeros(s)
        a[0] = 1.
        b[0] = 1.
    elif method =='S2':
        s= 2
        a = zeros(s)
        b = zeros(s)
        a[1] = 1.
        b[0] = 0.5
        b[1] = 0.5
    elif method == 'SS':
        s = 2
        a = zeros(s)
        b = zeros(s)
        a[0] = 0.5
        a[1] = 0.5
        b[0] = 1.
    elif method == 'BM42': # Order 4 (BM4-2, Blanes/Moan, Table 3, SRKNb6)
        s = 7
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.245298957184271
        a[2] = 0.604872665711080
        a[3] = 0.5 - a[:3].sum()
        a[4:] = flipud(a[1:4])
        b[0] = 0.0829844064174052
        b[1] = 0.396309801498368
        b[2] = - 0.0390563049223486
        b[3] = 1. - 2.*b[:3].sum()
        b[4:] = flipud(b[:3])            
    elif method == 'Y4': # Order 4 (Yoshida, see Hairer/Lubich/Wanner, p. 40, (4.4))
        s = 4
        a = zeros(s)
        b = zeros(s)
        pp = 3.
        theta = 1./(2.-2**(1./pp))
        vi = -2**(1./pp)*theta
        a[0] = 0.
        a[1] = theta
        a[2] = vi
        a[3] = theta
        b[0] = 0.5*theta
        b[1] = 0.5*(vi+theta)
        b[2:] = flipud(b[:2])
    elif method == 'Y61': #Order 6 (Yoshida, see Hairer/Lubich/Wanner, p. 144, (3.11), s = 8)
        s = 8
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.78451361047755726381949763
        a[2] = 0.23557321335935813368479318
        a[3] = -1.17767998417887100694641568
        a[4] = 1. - 2.*a[1:4].sum()
        a[5:] = flipud(a[1:4])
        #
        b[0] = 0.5*a[1]
        b[1] = 0.5*a[1:3].sum()
        b[2] = 0.5*a[2:4].sum()
        b[3] = 0.5*(1-4*b[1]-a[3])
        b[4:] = flipud(b[0:4])
    elif method == 'BM63': # Order 4... (BM6-3, Blanes/Moan, Table 3, SRKNa14)
        s = 15
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.09171915262446165
        a[2] = 0.183983170005006
        a[3] = -0.05653436583288827
        a[4] = 0.004914688774712854
        a[5] = 0.143761127168358
        a[6] = 0.328567693746804
        a[7] = 0.5 - a[:7].sum()
        a[8:] = flipud(a[1:8])
        #
        b[0] = 0.0378593198406116
        b[1] = 0.102635633102435
        b[2] = -0.0258678882665587
        b[3] = 0.314241403071447
        b[4] = -0.130144459517415
        b[5] = 0.106417700369543
        b[6] = -0.00879424312851058
        b[7] = 1. - 2.*b[:7].sum()
        b[8:] = flipud(b[:7])
        #
    elif method == 'KL6': #Order 6 (Kahan/Li, see Hairer/Lubich/Wanner, p. 144, (3.12))
        s = 10
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.39216144400731413927925056
        a[2] = 0.33259913678935943859974864
        a[3] = -0.70624617255763935980996482
        a[4] = 0.08221359629355080023149045
        a[5] = 1. - 2.*a[1:5].sum()
        a[6:] = flipud(a[1:5])
        #
        b[0] = 0.5*a[1]
        b[1] = 0.5*a[1:3].sum()
        b[2] = 0.5*a[2:4].sum()
        b[3] = 0.5*a[3:5].sum()
        b[4] = 0.5*(1-2*a[1:4].sum()-a[4])
        b[5:] = flipud(b[0:5])

    elif method == 'KL8': #Order 8 (Kahan/Li, see see Hairer/Lubich/Wanner (3.14)
        s = 18
        a = zeros(s)
        b = zeros(s)
        a[0] = 0.
        a[1] = 0.13020248308889008087881763
        a[2] = 0.56116298177510838456196441
        a[3] = -0.38947496264484728640807860
        a[4] = 0.15884190655515560089621075
        a[5] = -0.39590389413323757733623154
        a[6] = 0.18453964097831570709183254
        a[7] = 0.25837438768632204729397911
        a[8] = 0.29501172360931029887096624
        a[9] = -0.60550853383003451169892108
        a[10:] = flipud(a[1:9])
        #
        b[0:-1] = 0.5*(a[:-1]+a[1:])
        b[-1] = 1.*b[0]
        
        
    else: print 'method ' + method + ' not implemented'


    return a, b



def intsplit(psi1,psi2,a,b,tspan,N,args1=(),args2=()):
    if type(args1) != type(()): args1 = (args1,)
    if type(args2) != type(()): args2 = (args2,)
    s = a.shape[0]
    h = double(tspan[1]-tspan[0])/N
    
    for k in xrange(N):
        for j in xrange(s):
            #
            psi1(a[j]*h,*args1)
            psi2(b[j]*h,*args2)
            #
        

    
    
