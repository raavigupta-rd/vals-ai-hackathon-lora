#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;
static const int MAXV = 300000;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    int vmax = 1;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        vmax = max(vmax, a[i]);
    }

    vector<int> inv(vmax + 1, 1), mu(vmax + 1, 0), lp(vmax + 1, 0), primes;
    mu[1] = 1;
    for (int i = 2; i <= vmax; i++) {
        inv[i] = (int)(1LL * (MOD - MOD / i) * inv[MOD % i] % MOD);
        if (!lp[i]) {
            lp[i] = i;
            primes.push_back(i);
            mu[i] = -1;
        }
        for (int p : primes) {
            if (p > lp[i] || 1LL * i * p > vmax) break;
            lp[i * p] = p;
            if (p == lp[i]) {
                mu[i * p] = 0;
                break;
            } else {
                mu[i * p] = -mu[i];
            }
        }
    }

    // f(n) = sum_{d | n} mu(n / d) * d^{-1} mod MOD
    vector<int> f(vmax + 1, 0);
    for (int d = 1; d <= vmax; d++) {
        int id = inv[d];
        for (int m = d, k = 1; m <= vmax; m += d, k++) {
            if (mu[k] == 1) {
                f[m] += id;
                if (f[m] >= MOD) f[m] -= MOD;
            } else if (mu[k] == -1) {
                f[m] -= id;
                if (f[m] < 0) f[m] += MOD;
            }
        }
    }

    // Precompute divisors only for values present in array a
    vector<char> present(vmax + 1, 0);
    for (int i = 1; i <= n; i++) present[a[i]] = 1;
    vector<vector<int>> divs(vmax + 1);
    for (int d = 1; d <= vmax; d++) {
        for (int m = d; m <= vmax; m += d) {
            if (present[m]) divs[m].push_back(d);
        }
    }

    vector<int> mult_sum(vmax + 1, 0);
    vector<char> active(n + 1, 0);
    long long ordered_sum = 0; // sum_{d} f[d] * mult_sum[d]^2
    long long diag_sum = 0;    // sum_{i active} a[i]
    const long long INV2 = (MOD + 1) / 2;

    while (q--) {
        int x;
        cin >> x;
        int v = a[x];
        if (!active[x]) {
            active[x] = 1;
            diag_sum = (diag_sum + v) % MOD;
            for (int d : divs[v]) {
                long long old_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                mult_sum[d] += v;
                if (mult_sum[d] >= MOD) mult_sum[d] -= MOD;
                long long new_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                long long diff = new_sq - old_sq;
                if (diff < 0) diff += MOD;
                ordered_sum = (ordered_sum + diff * f[d]) % MOD;
            }
        } else {
            active[x] = 0;
            diag_sum = (diag_sum - v + MOD) % MOD;
            for (int d : divs[v]) {
                long long old_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                mult_sum[d] -= v;
                if (mult_sum[d] < 0) mult_sum[d] += MOD;
                long long new_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                long long diff = new_sq - old_sq;
                if (diff < 0) diff += MOD;
                ordered_sum = (ordered_sum + diff * f[d]) % MOD;
            }
        }
        long long ans = (ordered_sum - diag_sum + MOD) % MOD * INV2 % MOD;
        cout << ans << '\n';
    }
    return 0;
}
