#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> col(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> tin(n + 1), tout(n + 1), flat(n + 1), par(n + 1, 0);
    int timer = 0;
    vector<pair<int, int>> st = {{1, 0}};
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) {
            tin[u] = ++timer;
            flat[timer] = u;
        }
        if (idx < (int)adj[u].size()) {
            int v = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            st.push_back({v, 0});
        } else {
            tout[u] = timer;
            st.pop_back();
        }
    }

    vector<long long> acc(n + 1, 0);
    vector<int> touched;
    touched.reserve(n);
    while (q--) {
        int u, l, r;
        cin >> u >> l >> r;
        touched.clear();
        for (int t = tin[u]; t <= tout[u]; t++) {
            int v = flat[t];
            int c = col[v];
            if (l <= c && c <= r) {
                if (acc[c] == 0) touched.push_back(c);
                acc[c] = (acc[c] + w[v]) % MOD;
            }
        }
        long long res = 0;
        for (int c : touched) {
            res = (res + acc[c] * acc[c]) % MOD;
            acc[c] = 0;
        }
        cout << res << '\n';
    }
    return 0;
}
