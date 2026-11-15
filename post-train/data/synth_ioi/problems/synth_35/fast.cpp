#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 2;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }

    vector<int> spf(mx + 1, 0), primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }

    vector<vector<pair<int, int>>> by_r(n + 1);
    for (int i = 0; i < q; i++) {
        int l, r; cin >> l >> r;
        by_r[r].push_back({l, i});
    }

    vector<int> bit(n + 2, 0), last_pos(mx + 1, 0), ans(q, 0);
    auto add = [&](int idx, int d) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += d;
    };
    auto sum = [&](int idx) {
        int s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    int total_active = 0;
    for (int r = 1; r <= n; r++) {
        int x = a[r];
        while (x > 1) {
            int p = spf[x];
            while (x % p == 0) x /= p;
            if (last_pos[p]) add(last_pos[p], -1);
            else total_active++;
            last_pos[p] = r;
            add(r, +1);
        }
        for (auto [l, id] : by_r[r]) {
            ans[id] = total_active - sum(l - 1);
        }
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
