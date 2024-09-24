from functools import lru_cache
from numba import njit 
import numpy as np
import matplotlib.pyplot as plt


def get_activation_function(name):
    """
    Returns the activation function based on the provided name.

    Parameters:
    name (str): The name of the activation function. 
                Options are 'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 
                'leaky_relu', 'elu', 'selu', 'celu', 'gelu', 'swish'.

    Returns:
    function: A function that computes the specified activation.

    Example use:
    
    f = get_activation_function('relu',)
    coefs = compute_hermite_coefs(f, coefs_len=20)
    f2 = hermite_expansion(coefs)
    print(coefs)
    x = np.linspace(-3,3,100)
    """
    @njit
    def relu(x):
        return np.maximum(0, x)
    @njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @njit
    def tanh(x):
        return np.tanh(x)
    @njit
    def exp(x):
        return np.exp(x)
    @njit
    def softmax(x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    @njit
    def linear(x):
        return x
    @njit
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    @njit
    def elu(x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    @njit
    def selu(x, alpha=1.67326, scale=1.0507):
        return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    @njit
    def celu(x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1))
    @njit
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    @njit
    def swish(x, beta=1.0):
        return x / (1 + np.exp(-beta * x))
    
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'exp': exp,
        'softmax': softmax,
        'linear': linear,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'selu': selu,
        'celu': celu,
        'gelu': gelu,
        'swish': swish
    }
    
    return activations.get(name.lower(), None)

@lru_cache(maxsize=1000)
def factorial(k):
    assert k >= 0 and k == int(k)
    if k<=1:
        return 1
    return k * factorial(k-1)

def coefs_derivative(coefs, r):
    if r == 0:
        return coefs
    coefs = coefs.copy()
    # print(f'derivative 0 = ', ', '.join(f"c{i}={c}" for i,c in enumerate(coefs)))
    for i in range(r):
        for k in range(1,len(coefs)):
            coefs[k-1] = k * coefs[k]
        coefs[len(coefs)-1-i] = 0
        # print(f'derivative {i+1} = ', ', '.join(f"c{i}={c}" for i,c in enumerate(coefs)))
    return coefs

def coefs_to_poly(coefs):
    @njit
    def f(x):
        K = len(coefs)
        x_powers = np.zeros((K,len(x)),dtype=np.float64)
        for i in range(K):
            x_powers[i] = x**i
        return coefs @ x_powers
    return f

def hermite_poly_coefs(k):
    if k == 0:
        return np.array([1,])
    elif k == 1:
        return np.array([0,1])
    else:
        H_k_minus_1 = hermite_poly_coefs(k-1)
        H_k_minus_2 = hermite_poly_coefs(k-2)
        return np.concatenate([[0], H_k_minus_1]) - (k-1) * np.concatenate([H_k_minus_2, [0, 0]])

def hermite_poly_coefs_norm(k):
    coefs = hermite_poly_coefs(k)
    c = factorial(k)**0.5
    return coefs / c

def hermite_expansion(coefs, norm=True, return_coefs=False):
    K = len(coefs)
    cs = np.zeros(len(coefs),dtype=np.float64)
    for k,c in enumerate(coefs):
        if norm:
            cs += c * np.concatenate([hermite_poly_coefs_norm(k), np.zeros(K-1-k)])
        else:
            cs += c * np.concatenate([hermite_poly_coefs(k), np.zeros(K-1-k)])
    if return_coefs:
        return coefs_to_poly(cs), cs
    return coefs_to_poly(cs)

def hermite_poly(k,norm=True):
    coefs = np.zeros(k+1,dtype=np.float64)
    coefs[k] = 1
    return hermite_expansion(coefs, norm=norm)

@lru_cache(maxsize=1000)
def compute_hermite_coefs(f, order, norm=True, num_samples=10**7):
    X = np.random.randn(num_samples)
    hermite_coefs = np.zeros(order)
    for k in range(len(hermite_coefs)):
        hermite_coefs[k] = np.mean(f(X) * hermite_poly(k,norm=norm)(X))
        if not norm:
            hermite_coefs[k] /= factorial(k)
    return hermite_coefs


def kernel_map_emp(f, num_bins=100, num_samples=10**6,atol=1e-2,rtol=1e-2):
    rhos = np.linspace(-1,1,num_bins)
    vals = np.zeros(len(rhos))
    (x,y,z) = np.random.randn(3,num_samples)
    for i,rho in enumerate(rhos):
        ryz = np.sqrt(abs(rho))
        rxz = np.sign(rho) * ryz
        r = np.sqrt(1-abs(rho))
        X = rxz * z + r * y
        Y = ryz * z + r * x
        # test if rho = E[X Y], and variances are 1
        assert(np.allclose(np.mean(X * Y),rho,atol=atol,rtol=rtol))
        assert(np.allclose(np.var(X),1,atol=atol,rtol=rtol))
        assert(np.allclose(np.var(Y),1,atol=atol,rtol=rtol))
        vals[i] = np.mean(f(X) * f(Y))
    @njit
    def kernel(x):
        closest_indices = np.abs(rhos[:, np.newaxis] - x).argmin(axis=0)
        return vals[closest_indices]
    
    return kernel


# compute kernel map from Hermite coefficients
def kernel_map(coefs,r=0, norm=True):
    # cross terms dissapear, since E[He_k He_l] = 0 for k != l, leaving squared terms 
    coefs = coefs ** 2
    # if not normalized, E[He_k^2] = k!, if normalized E[He_k^2] = 1
    if not norm:
        c = 1
        for k in range(1,len(coefs)):
            c *= k
            coefs[k] = coefs[k] * c
    coefs = coefs_derivative(coefs, r)
    def kappa(x):
        return np.sum([(coefs[k]) * x**k for k in range(len(coefs))], axis=0)
    return kappa



def fixed_point_iteration(func, rho0, eps=1e-5, max_iterations=1000):
    x_values = [rho0]
    for _ in range(max_iterations):
        x_values.append(func(x_values[-1]))
        if len(x_values)>10 and abs(x_values[-1] - x_values[-2]) < eps:
            break
    return x_values

def plot_fixed_point_iteration(func, rho0, a = 0, b=1, eps=1e-5, max_iterations=1000, kernel_name='\\kappa(\\rho)'):
    x_values = fixed_point_iteration(func, rho0, eps=eps, max_iterations=max_iterations)
    x = np.linspace(a, b, 100)
    
    plt.plot(x, func(x), label=f'${kernel_name}$', color='blue')
    plt.plot(x, x, label='identity', color='black', linestyle='--')
    
    for i in range(1, len(x_values)):
        plt.plot([x_values[i-1], x_values[i-1]], [x_values[i-1], x_values[i]], 'r',linewidth=0.5)
        plt.plot([x_values[i-1], x_values[i]], [x_values[i], x_values[i]], 'r',linewidth=0.5)
        
    plt.scatter(x_values[-1], x_values[-1], marker='*', color='red', zorder=5, label=f'Fixed $\\rho^*={x_values[-1]:.2f}$')
    plt.scatter(x_values[0], func(x_values[0]), marker='.',color='red', zorder=5, label=f'Initial $\\rho_0={rho0:.2f}$')

    plt.axhline(0, color='black',linewidth=0.5)  # Add x-axis
    plt.axvline(0, color='black',linewidth=0.5)  # Add y-axis
    
    plt.scatter(x_values[0], func(x_values[0]), color='red', zorder=5)
    plt.title(f'Kernel map iterations $\\kappa(\\rho)$')
    plt.xlabel('pre-act $\\rho_\\ell$')
    plt.ylabel('post-act $\\rho_{\\ell+1}=\\kappa(\\rho)$')
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()




def convergence_bound(kappa, kappa_prime, rho0,eps=1e-4):
    rhos = fixed_point_iteration(kappa, rho0=rho0)
    rho_star = rhos[-1]
    rhos = rhos[:30]
    ls = np.arange(len(rhos))


    k0 = kappa(0)
    k_prime_star = kappa_prime(rho_star)
    k_prime_1 = kappa_prime(1)
    k_prime_0 = kappa_prime(0)

    print(f"k(0) = {k0:.4f}, k'(1) = {k_prime_1:.4f}, k'(rho*) = {k_prime_star:.4f}, k'(0) = {k_prime_0:.4f}")

    if abs(k0) < eps: # instead of 0, to avoid numerical issues
        case = ('case 1')
        x0 = abs(rho0-rho_star)
        # alpha =  np.exp(-ls*(1-k_prime_0))
        # bound = (x0 * alpha ) / (1-x0 + x0 * alpha )
        alpha = (1.0/k_prime_1)
        alpha = 1/(2-k_prime_0)
        bound = x0 * alpha **ls / (1-x0 + x0 * alpha**ls)
    else:
        if k_prime_1 < 1-eps: # k_prime_1 < 1
            case = ('case 2')
            alpha = (k_prime_1)
            bound = abs(rho0-rho_star) * alpha **ls
        elif abs(k_prime_1-1) < eps: # k_prime_1 = 1
            case = ('case 3')
            alpha = k0+k_prime_0
            bound = abs(rho0-rho_star) / (ls * (1-alpha) * abs(rho0-rho_star) + 1)
        else: # k_prime_1 > 1
            case = ('case 4')
            alpha1 = (1-rho_star)/(2-k_prime_star)
            alpha2 = k_prime_star
            alpha3 = 1-k0
            alpha = max([alpha1,alpha2,alpha3])
            bound = (abs(rho0-rho_star) * alpha**ls) / (1 - abs(rho0) + abs(rho0) * alpha**ls)

    print(case)
    # print(f"alpha = {alpha:.4f}, rho* = {rho_star:.4f}, k'* = {k_prime_star:.4f}, k0 = {k0:.4f}")
    l = np.arange(len(rhos))
    plt.plot(np.abs(rhos-rho_star), marker='.',label='empirical $|\\rho_\\ell - \\rho^*|$')
    plt.plot(bound, '--', color='red',marker='.',label=f'theory upper bound ({case})')
    plt.ylabel('$|\\rho_ \\ell-\\rho^*|$')
    plt.xlabel('$\\ell$')
    plt.title('Convergence of $\\rho_\\ell$ towards fixed point $\\rho^*$')
    plt.legend()
    # plt.yscale('log')
    plt.tight_layout()

def plot_activation(f, act_name, a=-3,b=3):
    x = np.linspace(a,b,100)
    plt.plot(x,f(x), label=f'${act_name}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Activation $\\phi(x)$')
    plt.legend()
    plt.tight_layout()

def coefs2name(poly_coefs):
    def sgn(c):
        if c==0:
            return ''
        if c>0:
            return '+'
        else:
            return '-'
    act_name = (f"\\phi(x)={''.join(f'{sgn(c)}{abs(c):.1f}{f"x^{i}" if i>0 else ""}' for i,c in enumerate(poly_coefs))}")
    return act_name 

def coefs2kernel_name(coefs):
    def term(c,i):
        if c==0:
            return ''
        elif i==0:
            return f'+{c:.1f}'
        else:
            return f'+{c:.1f}\\rho^{i}'
    kernel_name = (f"\\kappa(\\rho) ={''.join(term(c**2,i) for i,c in enumerate(coefs))}")
    return kernel_name

def test_kernel_map_from_coefs(coefs, norm, atol=1e-2, rtol=1e-2, plot=True):
    f = hermite_expansion(coefs, norm=norm)
    kernel_emp = kernel_map_emp(f)
    kernel_theory = kernel_map(coefs, norm=norm)
    x = np.linspace(-1,1,30)
    emp = kernel_emp(x)
    theory = kernel_theory(x)
    if plot:
        plt.figure()
        plt.plot(x, kernel_emp(x), label='empirical kappa(x)',marker='o')
        plt.plot(x, kernel_theory(x), label='theoretical kappa(x)')
        plt.xlabel('$\\rho$')
        plt.ylabel('$\\kappa(\\rho)$')
        plt.title(f'Kernel map in {"normalized (he)" if norm else "unnormalized (He)"} Hermite basis')
        plt.legend()
    if not np.allclose(emp, theory, atol=atol, rtol=rtol):
        print("Success: Kernel map theory and empirical values are close")
    else:
        print("Failed: Kernel map theory and empirical values are not close")


def test_kernel_map_properties_from_coefs(coefs,norm, num_samples=10**7,atol=1e-2, rtol=1e-2):
    f = hermite_expansion(coefs, norm=norm)
    kappa = kernel_map(coefs, norm=norm)
    kappa_prime = kernel_map(coefs, r=1, norm=norm)
    X = np.random.randn(num_samples)
    c0 = coefs[0]
    c1 = coefs[1]
    if norm:
        c2_sum = np.sum(coefs**2)
    else:
        c2_sum = np.sum([factorial(k) * c**2 for k,c in enumerate(coefs)])
    Ef = np.mean(f(X))
    Efx = np.mean(X * f(X))
    Ef2 = np.mean(f(X)**2)
    k0 = kappa(0)
    kprime_0 = kappa_prime(0)
    k1 = kappa(1)
    np.testing.assert_allclose(c0, Ef, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c0**2, k0, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c1, Efx, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c1**2, kprime_0, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c2_sum, Ef2, atol=atol,rtol=rtol)
    np.testing.assert_allclose(c2_sum, k1, atol=atol,rtol=rtol)
    print(f"Success: Kernel map properties in {'normalized (he)' if norm else 'unnormalized (He)'} basis are satisfied")

# test orthogonality of polynomials 
def test_orthogonality(K=4, eps = 1e-2):
    X = np.random.randn(10**7)
    for norm in [True, False]:
        poly_name = "he" if norm else "He"
        print(f"Testing orthogonality of {"normalized" if norm else ""} Hermite polynomials ({poly_name}(x))")
        for k in range(K):
            for l in range(k,K):
                f = hermite_poly(k,norm)
                g = hermite_poly(l,norm)
                theory = float(k==l)
                if not norm:
                    theory *= factorial(k)
                emp = np.mean(f(X) * g(X))
                error = np.abs(theory - emp)
                if error > eps:
                    message = "WARNING: "
                else:
                    message = ""
                print(f"{message} E [{poly_name}_{k}(X) {poly_name}_{l}(X)], theory = {theory:5.4f}, emp =  {emp:5.4f}, error = {error:5.4f}")


        # test if we can recover the coefficients of the expansion
def test_recovery(coefs, eps = 5e-2):
    for norm in [True, False]:
        poly_name = "he" if norm else "He"
        print(f"Testing recovery of {"normalized" if norm else ""} Hermite coefficients ({poly_name}(x))")
        # test with He_k (not normalized)
        f = hermite_expansion(coefs,norm=norm)
        hermite_coefs = compute_hermite_coefs(f, len(coefs)+3, norm=norm)

        for k,c in enumerate(hermite_coefs):
            c_org = coefs[k] if k < len(coefs) else 0
            err = np.abs(c_org - c) 
            if err > eps:
                message = "WARNING: "
            else:
                message = ""
            print(f"{message} c_k: original = {c_org:5.4f}, recovered = {c:5.4f}, error = {err:5.4f}")

