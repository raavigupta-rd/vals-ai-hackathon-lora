#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long w;
        cin >> l >> r >> w;
        by_r[r].push_back({l, w});
    }
    vector<vector<pair<int, int>>> qs(n + 1);
    for (int i = 0; i < q; i++) {
        int a, b;
        cin >> a >> b;
        qs[b].push_back({a, i});
    }
    vector<long long> bit(n + 1, 0), ans(q);
    auto add = [&](int idx, long long v) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };
    long long total = 0;
    for (int r = 1; r <= n; r++) {
        for (auto [l, w] : by_r[r]) { add(l, w); total += w; }
        for (auto [a, id] : qs[r]) ans[id] = total - sum(a - 1);
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
