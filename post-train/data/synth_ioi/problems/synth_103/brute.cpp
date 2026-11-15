#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    long long n;
    int k;
    cin >> n >> k;
    vector<vector<int>> M(k, vector<int>(k)), cur(k, vector<int>(k, 0)), nxt(k, vector<int>(k, 0));
    for (int i = 0; i < k; i++) {
        cur[i][i] = 1;
        for (int j = 0; j < k; j++) cin >> M[i][j];
    }

    // Step-by-step matrix powers O(N * K^3) time, O(K^2) memory
    long long total = 0;
    for (long long step = 1; step <= n; step++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                long long s = 0;
                for (int t = 0; t < k; t++) s = (s + 1LL * cur[i][t] * M[t][j]) % MOD;
                nxt[i][j] = (int)s;
            }
        }
        cur.swap(nxt);
        if (n % step == 0) {
            long long tr = 0;
            for (int i = 0; i < k; i++) tr = (tr + cur[i][i]) % MOD;
            long long x = n / step, phi = x;
            for (long long p = 2; p * p <= x; p++) {
                if (x % p == 0) {
                    while (x % p == 0) x /= p;
                    phi -= phi / p;
                }
            }
            if (x > 1) phi -= phi / x;
            total = (total + (phi % MOD) * tr) % MOD;
        }
    }
    cout << total * mod_pow(n % MOD, MOD - 2) % MOD << '\n';
    return 0;
}
