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
    const int P = (int)primes.size(); // 62

    int n, q;
    cin >> n >> q;
    vector<uint64_t> mask(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        unsigned long long x;
        cin >> x;
        uint64_t m = 0;
        for (int b = 0; b < P && x > 1; b++) {
            unsigned long long pr = primes[b];
            if (x % pr == 0) {
                int parity = 0;
                while (x % pr == 0) { x /= pr; parity ^= 1; }
                if (parity) m |= (1ULL << b);
            }
        }
        mask[i] = m;
    }

    vector<int> pow2(n + 1, 1);
    for (int i = 1; i <= n; i++) {
        pow2[i] = pow2[i - 1] * 2;
        if (pow2[i] >= MOD) pow2[i] -= MOD;
    }

    vector<vector<pair<int, int>>> qs(n + 1);
    for (int id = 0; id < q; id++) {
        int l, r;
        cin >> l >> r;
        qs[r].push_back({l, id});
    }

    uint64_t bvec[62] = {};
    int bpos[62] = {};
    vector<int> ans(q);

    for (int r = 1; r <= n; r++) {
        uint64_t x = mask[r];
        int pos = r;
        for (int b = P - 1; b >= 0; b--) {
            if (!((x >> b) & 1ULL)) continue;
            if (!bvec[b]) {
                bvec[b] = x;
                bpos[b] = pos;
                break;
            }
            if (bpos[b] < pos) {
                swap(bvec[b], x);
                swap(bpos[b], pos);
            }
            x ^= bvec[b];
        }
        for (auto [l, id] : qs[r]) {
            int dim = 0;
            for (int b = 0; b < P; b++) if (bpos[b] >= l) dim++;
            int free_vars = (r - l + 1) - dim;
            int ways = pow2[free_vars] - 1;
            if (ways < 0) ways += MOD;
            ans[id] = ways;
        }
    }

    for (int id = 0; id < q; id++) cout << ans[id] << '\n';
    return 0;
}
