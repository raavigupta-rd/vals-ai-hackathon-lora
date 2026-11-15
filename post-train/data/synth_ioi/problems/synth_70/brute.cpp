#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> primes;
    for (int p = 2; p <= 300; p++) {
        bool ok = true;
        for (int d = 2; d * d <= p; d++) if (p % d == 0) ok = false;
        if (ok) primes.push_back(p);
    }
    const int P = (int)primes.size();

    int n, q;
    cin >> n >> q;
    vector<uint64_t> mask(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        unsigned long long x; cin >> x;
        uint64_t m = 0;
        for (int b = 0; b < P && x > 1; b++) {
            unsigned long long pr = primes[b];
            int parity = 0;
            while (x % pr == 0) { x /= pr; parity ^= 1; }
            if (parity) m |= (1ULL << b);
        }
        mask[i] = m;
    }

    vector<int> pow2(n + 1, 1);
    for (int i = 1; i <= n; i++) pow2[i] = (pow2[i - 1] * 2) % MOD;

    while (q--) {
        int l, r; cin >> l >> r;
        uint64_t basis[62] = {};
        int dim = 0;
        for (int i = l; i <= r; i++) {
            uint64_t x = mask[i];
            for (int b = P - 1; b >= 0 && x; b--) {
                if (!((x >> b) & 1ULL)) continue;
                if (!basis[b]) { basis[b] = x; dim++; break; }
                x ^= basis[b];
            }
        }
        int ways = pow2[(r - l + 1) - dim] - 1;
        if (ways < 0) ways += MOD;
        cout << ways << '\n';
    }
    return 0;
}
