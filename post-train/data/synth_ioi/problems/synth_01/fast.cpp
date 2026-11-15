#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int sz;
    vector<int> mn;
    SegTree(int n) {
        sz = 1;
        while (sz <= n + 1) sz <<= 1;
        mn.assign(2 * sz, 0);
    }
    void update(int idx, int val) {
        int u = idx + sz;
        mn[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) mn[u] = min(mn[2 * u], mn[2 * u + 1]);
    }
    int query_first_lt(int thresh) const {
        int u = 1;
        while (u < sz) {
            if (mn[2 * u] < thresh) u = 2 * u;
            else u = 2 * u + 1;
        }
        return u - sz;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<pair<int, int>>> by_r(n + 1);
    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;
        by_r[r].push_back({l, i});
    }
    SegTree st(n + 1);
    vector<int> ans(q);
    for (int r = 1; r <= n; r++) {
        if (a[r] <= n + 1) st.update(a[r], r);
        for (auto [l, id] : by_r[r]) {
            ans[id] = st.query_first_lt(l);
        }
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
