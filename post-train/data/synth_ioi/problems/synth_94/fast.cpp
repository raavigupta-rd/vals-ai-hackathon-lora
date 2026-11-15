#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

long long modpow(long long a, long long e) {
    a %= MOD;
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
    int v, e;
    long long K;
    cin >> v >> e >> K;
    vector<int> B(v + 1);
    for (int i = 1; i <= v; i++) cin >> B[i];
    vector<vector<int>> g(v + 1);
    for (int i = 0; i < e; i++) {
        int x, y; cin >> x >> y;
        g[x].push_back(y);
    }

    vector<int> gr(v + 1, 0), seen(v + 5, 0);
    int stamp = 0;
    for (int u = v; u >= 1; u--) {
        ++stamp;
        for (int nb : g[u]) if (gr[nb] < (int)seen.size()) seen[gr[nb]] = stamp;
        int mex = 0;
        while (seen[mex] == stamp) mex++;
        gr[u] = mex;
    }

    int max_val = 0;
    for (int u = 1; u <= v; u++) max_val = max(max_val, gr[u] ^ B[u]);
    int SZ = 1;
    while (SZ <= max_val) SZ <<= 1;

    vector<long long> poly(SZ, 0);
    for (int u = 1; u <= v; u++) poly[gr[u] ^ B[u]]++;

    // XOR Fast Walsh-Hadamard Transform
    for (int len = 1; len < SZ; len <<= 1) {
        for (int i = 0; i < SZ; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                long long x = poly[i + j], y = poly[i + j + len];
                poly[i + j] = (x + y) % MOD;
                poly[i + j + len] = (x - y + MOD) % MOD;
            }
        }
    }

    for (int i = 0; i < SZ; i++) poly[i] = modpow(poly[i], K);

    // Inverse XOR FWHT: only need poly[0] = (1 / SZ) * sum_i poly[i]!
    long long sum_hat = 0;
    for (int i = 0; i < SZ; i++) sum_hat = (sum_hat + poly[i]) % MOD;
    long long zero_tuples = sum_hat * modpow(SZ, MOD - 2) % MOD;
    long long total_tuples = modpow(v, K);
    cout << (total_tuples - zero_tuples + MOD) % MOD << '\n';
    return 0;
}
