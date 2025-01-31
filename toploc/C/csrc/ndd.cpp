#include <torch/torch.h>

// TODO: Optimize the NDD memory usage
// TODO: Optimize the modular arithmetic operations to be minimum required to keep results positive

namespace py = pybind11;

constexpr int MOD_N = 65497;

inline int safeMod(long long v) {
    v = v % MOD_N;
    if (v < 0) v += MOD_N;
    return static_cast<int>(v);
}

// Current modInverse cause each iter q = a /m; then m - a %m; & a = 5. Basically a m is can siliently used in wrong roles
// More standart eea where we keep track of old_r and ols. So keep track of the current reamainder and coefficient.
// eea a^-1 (mod m)
int modInverse(int a, int m) {
    a = safeMod(a);
    if (m <= 1) {
        return 0;
    }
    
    // old_r, old_s track the "previous" remainder and coefficient
    // r, s track the "current" remainder and coefficient
    int old_r = a, r = m;
    int old_s = 1, s = 0;

    while (r != 0) {
        int q = old_r / r;     
        int temp_r = old_r - q*r;
        old_r = r;
        r = temp_r;

        int temp_s = old_s - q*s;
        old_s = s;
        s = temp_s;
    }

    // gcd(a, m) should always be 1 if a != 0 mod m. So better throw an error jic.
    if (old_r != 1) {
        throw std::runtime_error("No modular inverse: gcd(a, m) != 1.");
    }

    // Bezout coefficient for a.
    return safeMod(old_s);
}

// Helper function to calculate divided differences
std::vector<std::vector<int>> calculate_divided_differences(
    const std::vector<int>& x,
    const std::vector<int>& y
) {
    int n = x.size();
    std::vector<std::vector<int>> dd(n, std::vector<int>(n, 0));
    
    // Fill in the y-values in the first column
    for (int i = 0; i < n; i++) {
        dd[i][0] = y[i];
    }
    
    long long numer, denom, inv_denom;
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < n - j; i++) {
            numer = (dd[i + 1][j - 1] - dd[i][j - 1]) % MOD_N;
            denom = (x[i + j] - x[i]) % MOD_N;
            // denom = (denom + MOD_N) % MOD_N; should not be needed
            inv_denom = modInverse(denom, MOD_N);
            dd[i][j] = (numer * inv_denom) % MOD_N;
            dd[i][j] = (dd[i][j] + MOD_N) % MOD_N;
        }
    }
    
    return dd;
}

// Main function to compute polynomial coefficients
std::vector<int> compute_newton_coefficients(
    const std::vector<int>& x,
    const std::vector<int>& y
) {
    TORCH_CHECK(x.size() == y.size(), "Input vectors must have the same size");
    TORCH_CHECK(!x.empty(), "Input vectors must not be empty");

    int n = x.size();
    std::vector<std::vector<int>> dd = calculate_divided_differences(x, y);
    std::vector<int> coefficients(n, 0);
    
    // Convert divided differences to polynomial coefficients
    coefficients[0] = dd[0][0];
    
    for (int i = 1; i < n; i++) {
        // Compute term: dd[0][i] * (x - x[0])(x - x[1])...(x - x[i-1])
        std::vector<int> term(i + 1, 0);
        term[0] = dd[0][i];
        
        for (int j = 0; j < i; j++) {
            // Multiply by (x - x[j])
            std::vector<int> temp = term;
            for (int k = i; k > 0; k--) {
                term[k] = temp[k - 1];
            }
            term[0] = 0;
            
            for (int k = 0; k <= i; k++) {
                term[k] -= (temp[k] * x[j]) % MOD_N;
                term[k] = (term[k] % MOD_N + MOD_N) % MOD_N;
            }
        }
        
        // Add term to coefficients
        for (int j = 0; j <= i; j++) {
            coefficients[j] += term[j];
            coefficients[j] %= MOD_N;
        }
    }
    
    return coefficients;
}

// Fix interpolation test case error
int evaluate_polynomial(const std::vector<int>& coefficients, int x)
{
    // Start with highest degree coefficient 
    long long result = coefficients.back();

    // Apply Horner's method 
    for (int i = static_cast<int>(coefficients.size()) - 2; i >= 0; i--) {
        result = (result * x + coefficients[i]) % MOD_N;
    }

    return static_cast<int>((result + MOD_N) % MOD_N);
}

// Python module definition
PYBIND11_MODULE(ndd, m) {
    m.doc() = "Newton's divided difference interpolation for polynomial congruences"; 

    m.def("compute_newton_coefficients", &compute_newton_coefficients, 
          "Compute coefficients of interpolating polynomial using Newton's divided difference method",
          py::arg("x"),
          py::arg("y"));
    
    m.def("evaluate_polynomial", &evaluate_polynomial,
          "Evaluate polynomial at given point using computed coefficients",
          py::arg("coefficients"),
          py::arg("x"));
}
