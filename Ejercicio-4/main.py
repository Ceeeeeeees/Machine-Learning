from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles

def graficar(gamma_resp):

    X,y = make_moons(n_samples=100, random_state=123)
    XKPCA = KernelPCA(n_components=2, kernel='rbf', gamma=gamma_resp).fit_transform(X)

    plt.figure(figsize=(16,6))

    #Figura 1:
    plt.subplot(1,2,1)
    plt.scatter(X[y==0,0], X[y==0,1], color='green', alpha=0.5)
    plt.scatter(X[y==1,0], X[y==1,1], color='purple', alpha=0.5)

    plt.xlabel("$x_1$",fontsize=16)
    plt.ylabel("$x_2$",fontsize=16)
    #plt.show()

    #-----------------------------------------------------------------

    #plt.figure(figsize=(16,6))

    plt.subplot(1,2,2)
    plt.scatter(XKPCA[y==0,0], XKPCA[y==0,1], color='green', alpha=0.5)
    plt.scatter(XKPCA[y==1,0], XKPCA[y==1,1], color='purple', alpha=0.5)

    plt.xlabel("$x_1$",fontsize=16)
    plt.ylabel("$x_2$",fontsize=16)
    plt.show()

    #-----------------------------------------------------------------

    X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    XKPCA = KernelPCA(n_components=2, kernel='rbf', gamma=gamma_resp).fit_transform(X)

    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    plt.scatter(X[y==0,0], X[y==0,1], color='green', alpha=0.5)
    plt.scatter(X[y==1,0], X[y==1,1], color='purple', alpha=0.5)

    plt.xlabel("$x_1$",fontsize=16)
    plt.ylabel("$x_2$",fontsize=16)
    #plt.show()

    #-----------------------------------------------------------------

    #plt.figure(figsize=(16,6))

    plt.subplot(1,2,2)
    plt.scatter(XKPCA[y==0,0], XKPCA[y==0,1], color='green', alpha=0.5)
    plt.scatter(XKPCA[y==1,0], XKPCA[y==1,1], color='purple', alpha=0.5)

    plt.xlabel("$x_1$",fontsize=16)
    plt.ylabel("$x_2$",fontsize=16)
    plt.show()


respuesta_gamma = input("Deseas cambiar el valor de gamma? (1:Gamma = 15) (2:Gamma=1.5): ")

if respuesta_gamma == "1":
    graficar(15)
elif respuesta_gamma == "2":
    graficar(1.5)
    

