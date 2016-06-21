import numpy as np
import math
import matplotlib.pyplot as plt
import pyfits
import random
from random import *
import glob
from matplotlib.figure import Figure
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import batman
from scipy.stats import norm
import matplotlib.mlab as mlab

def Log_Like_GP(t,FT,c,rp,a,inc,alfas,S,err):
    a_k=3300
    tau=math.sqrt(3)
    PC=range(len(t))
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    #print PC
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 0.78884
    params.rp = rp
    params.a = a
    params.inc=inc
    params.ecc = 0.
    params.w = 90.     
    params.u = [0.1,0.3]
    params.limb_dark = "quadratic"
    #t2 = np.linspace(-2.0558955633,1.9441044367 , 100)
    t=np.array(t)
    m = batman.TransitModel(params,t)
    flux = m.light_curve(params)
    n=len(flux)
    C=create_constant_array(c,n)
    logFT=[]
    logflux=[]
    for i in xrange(0,n):
        logflux.append(np.log(flux[i]))
    
    model=np.array(C)+np.array(logflux)+np.array(PC)
    exp_mod=[]
    for i in xrange(0,len(model)):
        exp_mod.append(np.e**model[i])
    exp_mod=np.array(exp_mod)
    r=FT-exp_mod
    
    rt=r.T
    
    K=MatrixK(t,err,a_k,tau)
    
    Kinv=np.linalg.inv(K)
    #print np.dot(rt,np.dot(Kinv,r))
    
    #print np.linalg.det(K)
    return -0.5*(np.dot(rt,np.dot(Kinv,r)))-0.5*np.log(np.linalg.det(K))-0.5*len(r)*np.log(2.*np.pi)
    
   # return n*np.sum(np.log(1./(np.sqrt(2.*np.pi)*err)))+np.sum(-0.5*(FT-exp_mod)**2/err**2)

def delta_k(i,j):
    if i==j:
        return 1
    else:
        return 0
def Kernel(r,a_k,tau):
    return (a_k**2)*(1.+(math.sqrt(3.)*abs(r))/tau)*np.e**(-math.sqrt(3.)*abs(r)/tau)
def MatrixK(t,err,a_k,tau):
    N=len(t) 
    M=np.dot(np.identity(N),err**2)
    for i in xrange(0,N):
        for j in xrange(0,N):
            r=abs(t[i]-t[j])
            M[i][j]+=Kernel(r,a_k,tau)
    #print M
    return M

    

def t_to_days(t):
    n=len(t)
    t_out=range(n)
    for i in xrange(0,n):
        t_out[i]=0.0416667*t[i]
    return t_out

def Design_Matrix_a(Z,k):
    n=len(Z[0])
    M=Z[0].reshape(n,1)
    for i in xrange(1,k):
        M=np.hstack((M,Z[i].reshape(n,1)))
    return M
def Get_Coef_a(Z,Flux,k):
    M=Design_Matrix_a(Z,k)
    Mt=np.transpose(M)
    C=np.dot(Mt,M)
    inv=np.linalg.inv(C)
    D=np.dot(inv,Mt)
    logFlux=[]
    for i in xrange(0,len(Flux)):
        logFlux.append(np.log(Flux[i]))
    return np.dot(D,logFlux)
     
    
     
     
    
def Design_Matrix(Z,k): #Z en cada componente tiene una de las k PCA
        n=len(Z[0])
        print n
        C=np.array(create_constant_array(1,n)).reshape(n,1)
           
        M = np.hstack((C,Z[0].reshape(n,1)))
        for i in xrange(1,k):
            M=np.hstack((M,Z[i].reshape(n,1)))
        return M
def Get_Coef(Z,Flux,k):
    M=Design_Matrix(Z,k)
    Mt=np.transpose(M)
    C=np.dot(Mt,M)
    inv=np.linalg.inv(C)
    D=np.dot(inv,Mt)
    logFlux=[]
    for i in xrange(0,len(Flux)):
        logFlux.append(np.log(Flux[i]))
    return np.dot(D,logFlux)

def columnas(filee,x,indicador):
##      Si indicador es 1, columna es float, si es 0, columna es string,si es 2, integer
        n=0
        file=open(filee)        
        read1=file.readline()
        while len(read1)>1:
            n=n+1
            read1=file.readline()              
        c=range(n)
        file.close()
        file=open(filee)        
        read=file.readline()
        i=0
        if indicador==1:
            while i<n:                    
                ID=read.split("\t")
                c[i]=float(ID[x-1])
                i=i+1
                read=file.readline()
        if indicador==0:
            while i<n:
                ID=read.split("\t")
                c[i]=ID[x-1]
                if c[i][len(c[i])-1]=='\n':
                    c[i]=substring(c[i],0,len(c[i])-2)
                i=i+1
                read=file.readline()
        if indicador==2:
            while i<n:
                ID=read.split("\t")
                c[i]=int(float(ID[x-1]))
                i=i+1
                read=file.readline()               
        file.close()
        return c
def Get_Data(filee,char):
    if char=='t':
        return columnas(filee,1,1)
    if char=='FT':
        return columnas(filee,2,1)
    if char.isdigit():
        Comp_n=int(char)
        col=Comp_n+2 ##El flujo de la primera estrella esta en la columna 3
        return columnas(filee,col,1)

def Get_Mean(F):# F son arreglos en los que cada coordenada F[i] corresponden a una curva de luz
    all_F=[]
    for i in xrange(0,len(F)):
        all_F=np.concatenate((all_F,F[i]),axis=1)
    muF=np.mean(all_F)
    return muF,all_F

def cosorting(array1,array2):#Ordena array1 primero, y despues ordena array 2 de la misma forma
    array1Ord,array2Ord=zip(*sorted(zip(array1,array2)))
    return array1Ord,array2Ord
def Get_All_dataset(filee): #Obtiene las curvas de luz de todas las estrellas no variables
    n=9
    Out=range(n)
    for i in xrange(0,9):
        Out[i]=Get_Data(filee,str(i+1))
    return Out
def Test(cov,eig_vec,eig_val):
    print np.dot(cov,eig_vec)
    print eig_val*eig_vec
    return np.dot(cov,eig_vec)-eig_val*eig_vec<0.00000000001
def Matrix_W(eig_pairs,k):#Se eligen k autovalores
    out=eig_pairs[0][1].reshape(9,1)
    if k==1:
        return out
    if k>1:
        for i in xrange(1,k):
            out=np.hstack((out,eig_pairs[i][1].reshape(9,1)))
    if k==0:
        print 'Error, K no puede ser cero'
        return -1
    return out
def biggest_values(vect,k):
    out=range(k)
    for i in xrange(0,k):
        out[i]=vect[i]
    return out
    
def PCA(M,k):#M es una matriz donde en cada columna tiene una curva de luz, solo flujos
    Mt=np.transpose(M)
    cov=np.cov(M)
    eig_val,eig_vec=np.linalg.eig(cov)
    eigvec=[]
    for i in range(len(eig_val)):
        eigvec.append(eig_vec[:,i].reshape(1,9).T)
    #print Test(cov,eigvec[8],eig_val[8])
    eigval_ord,eigvec_ord=cosorting(eig_val,eigvec)
    SUM=np.sum(eig_val)
    eig_pairs=[]
    for i in xrange(0,len(eigval_ord)):
        eig_pairs.append([eigval_ord[i],eigvec_ord[i]])
    eig_pairs.reverse() #eig_pairs es una lista con autovalores/autovectores ordenadas de mayor AV a menor autovalor
    #print eig_pairs
    
    
    matrix_w= Matrix_W(eig_pairs,k)
    K_evalues=biggest_values(eig_pairs,k)
    K_evalues_out=[]##TRansformarlos en un arreglo simple
    for i in xrange(0,len(K_evalues)):
        K_evalues_out.append(K_evalues[i][0])
    transformed=matrix_w.T.dot(M)
    #print transformed.shape()
    return transformed,K_evalues_out
def create_constant_array(c,n):
    out=range(n)
    for i in xrange(0,n):
        out[i]=c
    return out
    
    
def Log_Like(t,FT,c,rp,a,inc,alfas,S,err):
    PC=range(len(t))
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    #print PC
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 0.78884
    params.rp = rp
    params.a = a
    params.inc=inc
    params.ecc = 0.
    params.w = 90.     
    params.u = [0.1,0.3]
    params.limb_dark = "quadratic"
    #t2 = np.linspace(-2.0558955633,1.9441044367 , 100)
    t=np.array(t)
    m = batman.TransitModel(params,t)
    flux = m.light_curve(params)
    n=len(flux)
    C=create_constant_array(c,n)
    logFT=[]
    logflux=[]
    for i in xrange(0,n):
        logFT.append(np.log(FT[i]))
        logflux.append(np.log(flux[i]))
    
    model=np.array(C)+np.array(logflux)+np.array(PC)
    exp_mod=[]
    for i in xrange(0,len(model)):
        exp_mod.append(np.e**model[i])
    exp_mod=np.array(exp_mod)
    #print (np.sum(np.log(1./(np.sqrt(2.*np.pi)*err)))+np.sum(-0.5*(FT-model)**2/err**2))
    #print 'ssssssss'
    #print len(np.log(1./(np.sqrt(2.*np.pi)*err)))
    return n*np.sum(np.log(1./(np.sqrt(2.*np.pi)*err)))+np.sum(-0.5*(FT-exp_mod)**2/err**2)
    out=range(len(t))
    alfas=[]
    PC=range(len(t))
    for i in xrange(1,len(coefs)):
        alfas.append(coefs[i])
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    for i in xrange(0,len(t)):
        out[i]=coefs[0]+PC[i]
    exp_out=[]
    for i in xrange(0,len(out)):
        exp_out.append(np.e**(out[i]))
    return exp_out
    
    
    
def Log_Prior(c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup):
    if c<cinf or c>csup:
        log_c_prior=-np.inf
    else:
        log_c_prior=1./(csup-cinf)
    

    if rp<rpinf or rp>rpsup:
        log_rp_prior=-np.inf
    else:
        log_rp_prior=1./(rpsup-rpinf)
   

    if a<ainf or a>asup:
        log_a_prior=-np.inf
    else:
        log_a_prior=1./(asup-ainf)
    
    
    if inc<incinf or inc>incsup:
        log_inc_prior=-np.inf
    else:
        log_inc_prior=1./(incsup-incinf)
    
    log_prior_alfas=range(len(alfas))
    for i in xrange(0,len(alfas)):
        if alfas[i]<alfasinf or alfas[i]>alfassup:
            log_prior_alfas[i]=-np.inf
        else:
            log_prior_alfas[i]=1./(alfassup-alfasinf)
        
    if err<errinf or err>errsup:
        log_err_prior=-np.inf
    else:
        log_err_prior=1./(errsup-errinf)
    
    return log_c_prior+log_rp_prior+log_a_prior+log_inc_prior+sum(log_prior_alfas)+log_err_prior
    
def Log_Post(t,FT,S,c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup):
    return Log_Like(t,FT,c,rp,a,inc,alfas,S,err)+Log_Prior(c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup)

def Metro(t,FT,S,c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup):
    nsteps=40000
    stepsize_c=(csup-cinf)/100.
    stepsize_a=(asup-ainf)/100.
    stepsize_rp=(rpsup-rpinf)/100.
    stepsize_inc=(incsup-incinf)/100.
    stepsize_alfas=0.1
    stepsize_err=(errsup-errinf)/100.
    output=range(nsteps)
    logpost=[]
    
    for j in xrange(0,nsteps):
        #print j
        post_previa= Log_Post(t,FT,S,c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup)
        stepc=np.random.normal(0,stepsize_c)
        stepa=np.random.normal(0,stepsize_a)
        steprp=np.random.normal(0,stepsize_rp)
        stepinc=np.random.normal(0,stepsize_inc)
        steperr=np.random.normal(0,stepsize_err)
        stepalfas=range(len(alfas))
        for i in xrange(len(alfas)):
            stepalfas[i]=np.random.normal(0,stepsize_alfas)
        alfas2=range(len(alfas))
        c2=c+stepc
        a2=a+stepa
        rp2=rp+steprp
        inc2=inc+stepinc
        err2=err+steperr
        
        for i in xrange(0,len(alfas)):
            alfas2[i]=alfas[i]+stepalfas[i]
        post_sgt=Log_Post(t,FT,S,c2,rp2,a2,inc2,alfas2,err2,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup)
        criterio=post_sgt-post_previa
        print 'posteriores =' +str(post_previa)+' '+str(post_sgt)
        if criterio>=0.:
            output[j]=[c2,a2,rp2,inc2,err2,alfas2]
            c=c2
            a=a2
            rp=rp2
            inc=inc2
            err=err2
            alfas=alfas2
            logpost.append(post_sgt)
        else:
            u=random()
            #print np.log(u)
            if criterio>np.log(u):
                output[j]=[c2,a2,rp2,inc2,err2,alfas2]
                c=c2
                a=a2
                rp=rp2
                inc=inc2
                err=err2
                alfas=alfas2
                logpost.append(post_sgt)
            else:
                output[j]=[c,a,rp,inc,err,alfas]
                logpost.append(post_previa)
                
     
    return output,logpost
    
def Recover_out(A,index):
    out=[]
    for i in xrange(0,len(A)):
        out.append(A[i][index])
    return out
     
   
    
def Get_MCMC_Coefs(A):
    log_post=A[1]
    #A[0]=[c,a,rp,inc,err,alfas]
    n=len(Recover_out(A[0],0))
    c=Recover_out(A[0],0)[n-1]
    a=Recover_out(A[0],1)[n-1]
    rp=Recover_out(A[0],2)[n-1]
    inc=Recover_out(A[0],3)[n-1]
    err=Recover_out(A[0],4)[n-1]
    out=Recover_out(A[0],5)[0]
    n_alphas=len(out)
    print n_alphas
    alfas=[]
    for i in xrange(0,n_alphas):
        out=Recover_out(A[0],5)
        out2=Recover_out(out,i)[n-1]
        alfas.append(out2)
    return c,a,rp,inc,err,alfas
def plot_chains(A,ind):
    plt.close()
    if ind==1:
        name='GP'
    if ind==0:
        name='noGP'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    out_c=Recover_out(A[0],0)
    out_a=Recover_out(A[0],1)
    out_rp=Recover_out(A[0],2)
    out_inc=Recover_out(A[0],3)
    out_err=Recover_out(A[0],4)
    out_alfas=Recover_out(A[0],5)
    out_alfa1=Recover_out(out_alfas,0)
    plt.figure(1)
    plt.subplot(3,2,1)
    plt.title('MC para c')
    plt.plot(out_c)
    plt.subplot(3,2,2)
    plt.title('MC para a')
    plt.plot(out_a)
    plt.subplot(3,2,3)
    plt.plot(out_rp)
    plt.title('MC para rp')
    plt.subplot(3,2,4)
    plt.title('MC para inc')
    plt.plot(out_inc)
    plt.subplot(3,2,5)
    plt.title('MC para err')
    plt.plot(out_err)
    plt.subplot(3,2,6)
    plt.title('MC para alfa_1')
    plt.plot(out_alfa1)
    plt.savefig('MC_params_'+name)
    plt.figure(2)
    plt.title('Posterioris de la MC')
    plt.plot(A[1])
    plt.savefig('Posterioris_MC_'+name)
    #plt.show()
    plt.close()
    
    
def linear_model_a(t,f,COEF,S):#Los COEFS son los alfas
    alfas=COEF
    PC=range(len(t))
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    PC=np.array(PC)
    PC2=np.exp(PC)
    plt.plot(t,f)
    plt.plot(t,PC2)
    plt.show()
    return
def linear_model(t,f,COEF,S):#Los COEFS son el C en la primera columna + los alfas
    plt.close()
    c=COEF[0]
    alfas=[]
    n=len(t)
    for i in xrange(1,len(COEF)):
        alfas.append(COEF[i])
    PC=range(len(t))
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    C=create_constant_array(c,n)
    mod=[]
    for i in xrange(0,n):
        mod.append(C[i]+PC[i])
    mod=np.array(mod)
    mod=np.exp(mod)
    residual=np.array(mod)-np.array(f)
    plt.subplot(2,1,1)
    plt.title('Modelo del que se obtuvieron los priors para c y para los alfas')
    plt.plot(t,f,'o')
    plt.plot(t,mod)
    plt.subplot(2,1,2)
    plt.title('Residuos')
    plt.plot(t,residual)
    plt.savefig('linearMod')
    plt.close()
    return residual
    

def MCMC_model(A,t,FT,S,ind):
    plt.close()
    if ind==0:
        name='noGP'
    if ind==1:
        name='GP'
    B=Get_MCMC_Coefs(A)
    c=B[0]
    a=B[1]
    rp=B[2]
    inc=B[3]
    err=B[4]
    alfas=B[5]
    PC=range(len(t))
    for i in xrange(0,len(t)):
        PC[i]=0
        for j in xrange(0,len(alfas)):
            ###print str(i)+' '+str(j)
            PC[i]+=alfas[j]*S[j][i]
    #print PC
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 0.78884
    params.rp = rp
    params.a = a
    params.inc=inc
    params.ecc = 0.
    params.w = 90.     
    params.u = [0.1,0.3]
    params.limb_dark = "quadratic"
    #t2 = np.linspace(-2.0558955633,1.9441044367 , 100)
    t=np.array(t)
    m = batman.TransitModel(params,t)
    flux = m.light_curve(params)
    n=len(flux)
    C=create_constant_array(c,n)
    logFT=[]
    logflux=[]
    for i in xrange(0,n):
        logFT.append(np.log(FT[i]))
        logflux.append(np.log(flux[i]))
    model=np.array(C)+np.array(logflux)+np.array(PC)
    exp_mod=[]
    for i in xrange(0,len(model)):
        exp_mod.append(np.e**model[i])
    residual=np.array(exp_mod)-np.array(FT)
    plt.close()
    plt.subplot(3,1,1)
    plt.title('Transito Ajustado')
    plt.plot(t,flux)
    plt.subplot(3,1,2)
    plt.title('Modelo Completo ajustado a los datos')
    plt.plot(t,FT,'o')
    plt.plot(t,exp_mod)
    plt.subplot(3,1,3)
    plt.title('Residuos')
    plt.plot(t,residual)
    plt.savefig('MCMC_model'+name)
    plt.close()
    #plt.show()
    return flux
def BIC(filee):
    plt.close()
    nparams=[]
    bic=[]
    n=9 #Son 9 PCA asi que habra que probar cuantas incluir en el modelo
    rp=0.5
    a=5
    inc=90
    err=0.001
    cinf=-5
    csup=5
    rpinf=0.0001
    rpsup=0.9
    ainf=1
    asup=15
    incinf=87
    incsup=93
    alfasinf=-30
    alfassup=30
    errinf=0.0000000000001
    errsup=1
    FT=Get_Data(filee,'FT')
    t=Get_Data(filee,'t')
    F1=Get_Data(filee,'1')
    t=t_to_days(t)
    M=Get_All_dataset(filee)
    N=len(t)
    ###############################
    for i in xrange(0,n):
        k=i+1
        Z=PCA(M,k)
        COEF=Get_Coef(Z[0],F1,k)
        c=COEF[0]
        alfas=[]
        for i in xrange(1,len(COEF)):
            alfas.append(COEF[i])
        #print alfas
        S=Z[0]
        A=Metro(t,FT,S,c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup)
        B=Get_MCMC_Coefs(A)
        c=B[0]
        a=B[1]
        rp=B[2]
        inc=B[3]
        err=B[4]
        alfas=B[5]
        L_k=Log_Like(t,FT,c,rp,a,inc,alfas,S,err)
        nparams.append(k)
        bic.append(-2.*L_k+(k+5)*np.log(N))
    ###################################
    plt.plot(nparams,bic)
    plt.title('BIC Testing')
    plt.xlabel('Number of PCAs')
    plt.ylabel('BIC')
    plt.savefig('BIC')
    plt.close()
    #plt.show()
    return nparams,bic
def a_posterioris(A,nbins,ind):# si ind=0, sin procesos GP, si ind=1, con GP
    plt.close()
    c=Recover_out(A[0],0)
    a=Recover_out(A[0],1)
    rp=Recover_out(A[0],2)
    inc=Recover_out(A[0],3)
    err=Recover_out(A[0],4)    
    if ind==0:
        consin='sin'
        name='noGP'
    if ind==1:
        consin='con'
        name='GP'
    plt.hist(c,nbins)
    plt.title('Distribucion a-posteriori de c '+consin+'  GP')
    plt.savefig(name+'_c')
    plt.close()
    plt.hist(a,nbins)
    (mu_a, sigma_a) = norm.fit(a)
    print 'mu_a='+str(mu_a)+' sigma a= '+str(sigma_a)
    plt.title('Distribucion a-posteriori de a '+consin+' GP')
    plt.savefig(name+'_a')
    plt.close()
    plt.hist(rp,nbins)
    (mu_rp, sigma_rp) = norm.fit(rp)
    print 'mu_rp='+str(mu_rp)+' sigma rp= '+str(sigma_rp)
    plt.title('Distribucion a-posteriori de rp '+consin+' GP')
    plt.savefig(name+'_rp')
    plt.close()
    plt.hist(inc,nbins)
    (mu_inc, sigma_inc) = norm.fit(inc)
    print 'mu_inc='+str(mu_inc)+' sigma inc= '+str(sigma_inc)
    plt.title('Distribucion a-posteriori de inc '+consin+' GP')
    plt.savefig(name+'_inc')
    plt.close()
    plt.hist(err,nbins)
    plt.title('Distribucion a-posteriori de err '+consin+' GP')
    plt.savefig(name+'_err')          
    plt.close()
    return np.mean(c),np.mean(a),np.mean(rp),np.mean(inc),np.mean(err),np.median(c),np.median(a),np.median(rp),np.median(inc),np.median(err)
    
    
def Obtener_priors(filee,k):
    t=Get_Data(filee,'t')
    F1=Get_Data(filee,'1')
    t=t_to_days(t)
    M=Get_All_dataset(filee)
    Z=PCA(M,k)
    COEF=Get_Coef(Z[0],F1,k)
    S=Z[0]
    linear_model(t,F1,COEF,S)

def Plot_Best_Model(A,t,FT,S,ind):    
    MCMC_model(A,t,FT,S,ind)
    plot_chains(A,ind)
    
    
    
    
def Examen_a(filee):
    plt.close()
    FT=Get_Data(filee,'FT')
    t=Get_Data(filee,'t')
    M=Get_All_dataset(filee)
    k=9 #Todas las Senales
    Z=PCA(M,k)
    plt.figure(1)
    plt.title('PCA')
    plt.subplot(2,2,1)
    plt.title('Curva de luz')
    plt.plot(t,F1,label='Estrella Sin transito')
    plt.subplot(2,2,2)
    plt.title('AV='+str(Z[1][0]))
    plt.plot(t,Z[0][0],label='Componente 1')
    plt.subplot(2,2,3)
    plt.title('AV='+str(Z[1][1]))
    plt.plot(t,Z[0][1],label='Componente 2')
    plt.subplot(2,2,4)
    plt.title('AV='+str(Z[1][2]))
    plt.plot(t,Z[0][2],label='Componente 3')
    plt.savefig('PCA_1')
    #plt.legend()
    plt.close()
    plt.figure(2)
    plt.title('PCA')
    plt.subplot(3,2,1)
    plt.plot(t,Z[0][3],label='Componente 4')
    plt.subplot(3,2,2)
    plt.plot(t,Z[0][4],label='Componente 5')
    plt.subplot(3,2,3)
    plt.plot(t,Z[0][5],label='Componente 6')
    plt.subplot(3,2,4)
    plt.plot(t,Z[0][6],label='Componente 7')
    plt.subplot(3,2,5)
    plt.plot(t,Z[0][7],label='Componente 8')
    plt.subplot(3,2,6)
    plt.plot(t,Z[0][8],label='Componente 9')
    #plt.legend()
    plt.savefig('PCA_2')
    plt.close()


    

  
    


    
k=2
filee='dataset_8.dat'
FT=Get_Data(filee,'FT')
t=Get_Data(filee,'t')
F1=Get_Data(filee,'1')
t=t_to_days(t)
M=Get_All_dataset(filee)
Z=PCA(M,k)
#print Z[0]
print len(t)
print len(F1)
#plt.plot(t,F1,label='F1')
#plt.plot(t,Z[0][0],label='Componente 1')
#plt.plot(t,Z[0][1],label='Componente 2')
#plt.plot(t,Z[0][2],label='Componente 3')
#plt.plot(t,Z[0][3],label='Componente 4')
#plt.legend()
#plt.show()
#print Z[0]


COEF=Get_Coef(Z[0],F1,k)
print COEF

c=COEF[0]
rp=0.5
a=5
inc=90

alfas=[]
for i in xrange(1,len(COEF)):
    alfas.append(COEF[i])
print alfas
S=Z[0]
err=0.001
cinf=-5
csup=5
rpinf=0.0001
rpsup=0.9
ainf=1
asup=15
incinf=87
incsup=93
alfasinf=-30
alfassup=30
errinf=0.0000000000001
errsup=1

#BIC(filee)

#output[j]=[c,a,rp,inc,err,alfas]
A=Metro(t,FT,S,c,rp,a,inc,alfas,err,cinf,csup,rpinf,rpsup,ainf,asup,incinf,incsup,alfasinf,alfassup,errinf,errsup)
out=A[0]
logpost=A[1]

B=Get_MCMC_Coefs(A)
ind=0

nbins=30

Examen_a(filee)
Obtener_priors(filee,k)
Plot_Best_Model(A,t,FT,S,ind)
a_posterioris(A,nbins,ind)



               

#a=1/0

#print Log_Like(t,FT,c,rp,a,inc,alfas,S,err)    



