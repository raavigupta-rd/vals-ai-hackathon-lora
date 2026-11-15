#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

using Mat = array<array<int, 20>, 20>;

Mat mat_mul(const Mat& A, const Mat& B, int k) {
    Mat C{};
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            __int128 acc = 0;
            for (int t = 0; t < k; t++) {
                acc += (__int128)A[i][t] * B[t][j];
            }
            C[i][j] = (int)(acc % MOD);
        }
    }
    return C;
}

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

long long euler_phi(long long x) {
    long long res = x;
    for (long long p = 2; p * p <= x; p++) {
        if (x % p == 0) {
            while (x % p == 0) x /= p;
            res -= res / p;
        }
    }
    if (x > 1) res -= res / x;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    long long n;
    int k;
    if (!(cin >> n >> k)) return 0;
    Mat M{};
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            cin >> M[i][j];
            M[i][j] %= MOD;
        }
    }

    array<Mat, 31> pw;
    pw[0] = M;
    for (int b = 1; b <= 30; b++) pw[b] = mat_mul(pw[b - 1], pw[b - 1], k);

    auto trace_pow = [&](long long d) -> long long {
        Mat R{};
        for (int i = 0; i < k; i++) R[i][i] = 1;
        for (int b = 0; b <= 30; b++) {
            if ((d >> b) & 1LL) R = mat_mul(R, pw[b], k);
        }
        long long tr = 0;
        for (int i = 0; i < k; i++) tr = (tr + R[i][i]) % MOD;
        return tr;
    };

    vector<long long> divs;
    for (long long d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            divs.push_back(d);
            if (d * d != n) divs.push_back(n / d);
        }
    }

    long long total = 0;
    for (long long d : divs) {
        long long phi = euler_phi(n / d) % MOD;
        long long tr = trace_pow(d);
        total = (total + phi * tr) % MOD;
    }
    long long ans = total * mod_pow(n % MOD, MOD - 2) % MOD;
    cout << ans << '\n';
    return 0;
}
